# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "agentx_result", ROOT / "agentx_result.py"
)
RESULT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESULT)


class ManifestTest(unittest.TestCase):
    def test_manifest_identifies_run_snapshots_and_auditor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auditor = root / "agentx_result.py"
            auditor.write_bytes((ROOT / "agentx_result.py").read_bytes())
            for name in (
                "agentx.slurm",
                "harness.slurm",
                "server.sh",
                "source-server.sh",
                "traces.jsonl",
                "dataset/traces.jsonl",
                "adapter.py",
                "timing/phase/runner.py",
                "credit/callback_handler.py",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            (root / "scenario.json").write_text('{"name":"agentx","mode":"smoke"}')
            environment = dict.fromkeys(
                (
                    "CONTAINER_IMAGE",
                    "CONTAINER_MOUNTS",
                    "CLIENT_PYTHON",
                    "MODEL_NAME",
                    "TOKENIZER_PATH",
                    "CONCURRENCY",
                    "SEED",
                    "API_PORT",
                    "READINESS_PATH",
                    "READINESS_TIMEOUT",
                    "HOLD_AFTER_RUN",
                    "SLURM_JOB_ID",
                ),
                "test",
            )
            environment.update(
                SOURCE_ROOT=str(root),
                SERVER_SCRIPT=str(root / "source-server.sh"),
                DATASET_PATH=str(root),
                DURATION="60",
                CLIENT_TIMEOUT="120",
            )
            evalscope = MagicMock()
            adapter = evalscope.perf.scenarios.agentx
            adapter.__file__ = str(root / "adapter.py")
            scenario = adapter.AgentXScenario.model_validate_json.return_value
            scenario.mode = "smoke"
            scenario.model_dump.return_value = {"name": "agentx", "mode": "smoke"}
            aiperf = MagicMock()
            aiperf.__file__ = str(root / "__init__.py")
            modules = {
                "evalscope": evalscope,
                "evalscope.perf": evalscope.perf,
                "evalscope.perf.scenarios": evalscope.perf.scenarios,
                "evalscope.perf.scenarios.agentx": adapter,
                "aiperf": aiperf,
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, modules),
                patch.object(RESULT, "__file__", str(auditor)),
                patch.object(
                    RESULT.subprocess, "check_output", return_value="same-commit"
                ),
                patch.object(RESULT.importlib.metadata, "version", return_value="test"),
            ):
                RESULT.prepare(root)
                before = json.loads((root / "manifest.json").read_text())
                self.assertEqual(
                    before["harness_sha256"],
                    hashlib.sha256((root / "harness.slurm").read_bytes()).hexdigest(),
                )
                self.assertNotEqual(
                    before["harness_sha256"],
                    hashlib.sha256((root / "agentx.slurm").read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    before["server_script_sha256"],
                    hashlib.sha256((root / "server.sh").read_bytes()).hexdigest(),
                )
                self.assertEqual(before["server_script_path"], str(root / "server.sh"))
                (root / "source-server.sh").write_text("Changed server source")
                (root / "agentx.slurm").write_text("Changed checkout after submission")
                self.assertEqual(
                    before["dataset_sha256"],
                    hashlib.sha256(
                        (root / "dataset/traces.jsonl").read_bytes()
                    ).hexdigest(),
                )
                self.assertEqual(before["dataset_path"], str(root / "dataset"))
                (root / "traces.jsonl").write_text(
                    "Shared dataset replaced after preparation"
                )
                self.assertEqual(
                    before["auditor_sha256"],
                    hashlib.sha256(auditor.read_bytes()).hexdigest(),
                )
                with auditor.open("a") as stream:
                    stream.write("\n# Local auditor modification without a commit.\n")
                RESULT.prepare(root)
                after = json.loads((root / "manifest.json").read_text())
                self.assertEqual(
                    before["server_script_sha256"], after["server_script_sha256"]
                )
                (root / "harness.slurm").write_text("Different executed script")
                RESULT.prepare(root)
                different_launcher = json.loads((root / "manifest.json").read_text())
                self.assertNotEqual(
                    after["harness_sha256"], different_launcher["harness_sha256"]
                )
                self.assertEqual(
                    different_launcher["harness_sha256"],
                    hashlib.sha256((root / "harness.slurm").read_bytes()).hexdigest(),
                )
            self.assertEqual(before["harness_commit"], after["harness_commit"])
            self.assertEqual(before["harness_sha256"], after["harness_sha256"])
            self.assertEqual(before["dataset_sha256"], after["dataset_sha256"])
            self.assertNotEqual(before["auditor_sha256"], after["auditor_sha256"])
            self.assertEqual(
                after["auditor_sha256"],
                hashlib.sha256(auditor.read_bytes()).hexdigest(),
            )

    def test_server_parameters_are_recorded_without_mandating_a_launcher(self):
        environment = dict(
            MODEL_DIR="/models/target",
            DRAFT_DIR="/models/draft",
            MODEL_REVISION="target-revision",
            DRAFT_REVISION="draft-revision",
            SERVER_VENV="/opt/server-venv",
            GPU_MEMORY_UTILIZATION="0.9",
            SERVER_SEED="7",
        )
        baseline = RESULT.server_configuration(environment)
        self.assertEqual(baseline, environment)
        for field, changed in (
            ("MODEL_DIR", "/models/other-target"),
            ("DRAFT_DIR", "/models/other-draft"),
            ("MODEL_REVISION", "other-target-revision"),
            ("DRAFT_REVISION", "other-draft-revision"),
            ("SERVER_SEED", "8"),
            ("GPU_MEMORY_UTILIZATION", "0.8"),
            ("SERVER_VENV", ""),
        ):
            with self.subTest(field=field):
                updated = RESULT.server_configuration(
                    dict(environment, **{field: changed})
                )
                self.assertNotEqual(baseline, updated)
                self.assertEqual(updated[field], changed)
        for field in environment:
            with self.subTest(missing=field):
                incomplete = dict(environment)
                del incomplete[field]
                self.assertIsNone(RESULT.server_configuration(incomplete)[field])


class ResultTest(unittest.TestCase):
    def test_phase_and_request_evidence(self):
        summary = {
            "status": "completed",
            "error_summary": [],
            "scenario": {"mode": "smoke"},
            "submission_valid": False,
            "metrics": {"request_count": {"avg": 1}},
        }
        records = [
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                    "context_overflow_skip": False,
                },
                "metrics": {"output_sequence_length": {"value": 7}},
            }
        ]
        for case in (
            "success",
            "hidden_cancel_smoke",
            "hidden_cancel_benchmark",
            "error",
            "overflow",
            "missing_phase",
            "grace_timeout",
            "count_mismatch",
            "summary_error",
            "phase_error",
        ):
            with self.subTest(case=case):
                s, rows = copy.deepcopy(summary), copy.deepcopy(records)
                cancelled = int(case.startswith("hidden_cancel"))
                log = f"Phase profiling (profiling) complete | completed=1, cancelled={cancelled}, errors=0"
                if case == "hidden_cancel_benchmark":
                    s["scenario"]["mode"] = "benchmark"
                elif case == "error":
                    rows[0]["error"] = {"message": "failed"}
                elif case == "overflow":
                    rows[0]["metadata"]["context_overflow_skip"] = True
                elif case == "grace_timeout":
                    s["scenario"]["mode"] = "benchmark"
                    log += " | elapsed=2400.00s | grace_period_timeout=True"
                elif case == "missing_phase":
                    log = ""
                elif case == "summary_error":
                    s["error_summary"] = ["failed request"]
                elif case == "phase_error":
                    log = log.replace("errors=0", "errors=1")
                elif case == "count_mismatch":
                    s["metrics"]["request_count"]["avg"] = 2
                if case in ("success", "hidden_cancel_benchmark"):
                    self.assertEqual(RESULT.audit(s, rows, log)["cancelled"], cancelled)
                else:
                    with self.assertRaises(ValueError):
                        RESULT.audit(s, rows, log)
                with tempfile.TemporaryDirectory() as directory:
                    run = Path(directory)
                    artifact = run / "client/run/aiperf"
                    (artifact / "logs").mkdir(parents=True)
                    (artifact.parent / "agentx_summary.json").write_text(json.dumps(s))
                    (artifact / "profile_export.jsonl").write_text(
                        "\n".join(json.dumps(row) for row in rows)
                    )
                    (artifact / "logs/aiperf.log").write_text(log)
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "agentx_result.py"),
                            "audit",
                            str(run),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    result = json.loads((run / "audit.json").read_text())
                    accepted = case in ("success", "hidden_cancel_benchmark")
                    self.assertEqual(process.returncode == 0, accepted, process.stderr)
                    if not accepted:
                        self.assertEqual(result["status"], "rejected")
                        self.assertTrue(result["reason"])
                        self.assertEqual(result["error_type"], "ValueError")
                        self.assertEqual(
                            result["cancelled"],
                            None if case == "missing_phase" else cancelled,
                        )
                        self.assertEqual(
                            result["errors"],
                            (
                                None
                                if case == "missing_phase"
                                else int(case == "phase_error")
                            ),
                        )

    def test_missing_or_invalid_artifacts_leave_failure_record(self):
        for case in (
            "missing_summary",
            "invalid_summary",
            "invalid_export",
            "ambiguous_phases",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                if case != "missing_summary":
                    artifact = run / "client/run/aiperf"
                    (artifact / "logs").mkdir(parents=True)
                    (artifact.parent / "agentx_summary.json").write_text("invalid")
                    log = "Phase profiling (profiling) complete | completed=1, cancelled=2, errors=3"
                    if case == "ambiguous_phases":
                        log += "\n" + log
                    (artifact / "logs/aiperf.log").write_text(log)
                    if case == "invalid_export":
                        (artifact.parent / "agentx_summary.json").write_text("{}")
                        (artifact / "profile_export.jsonl").write_text("invalid")
                with self.assertRaises(ValueError):
                    RESULT.main("audit", run)
                result = json.loads((run / "audit.json").read_text())
                self.assertEqual(result["status"], "rejected")
                self.assertTrue(result["reason"])
                if case in ("missing_summary", "ambiguous_phases"):
                    self.assertIsNone(result["completed"])
                    self.assertEqual(
                        len(result["profiling_phase_counts"]),
                        0 if case == "missing_summary" else 2,
                    )
                else:
                    self.assertEqual(
                        (result["completed"], result["cancelled"], result["errors"]),
                        (1, 2, 3),
                    )


class LifecycleTest(unittest.TestCase):
    def test_cleanup_and_hold(self):
        for case in (
            "success",
            "spool_copy",
            "dataset_replaced",
            "auditor_replaced",
            "server_replaced",
            "copy_sigterm",
            "copy_sigint",
            "copy_failure",
            "prepare_sigterm",
            "prepare_sigint",
            "prepare_failure",
            "audit_sigterm",
            "audit_sigint",
            "audit_failure",
            "client_failure",
            "client_sigterm",
            "client_sigint",
            "client_timeout",
            "hold",
            "hold_marker",
            "hold_sigterm",
            "hold_sigint",
            "occupied",
            "startup_failure",
            "startup_sigterm",
            "startup_sigint",
            "startup_timeout",
            "preflight_sigterm",
            "preflight_sigint",
            "preflight_failure",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binaries = root / "bin"
                binaries.mkdir()
                scripts = {
                    "cp": '''if [ "$2" = "$DATASET_PATH/traces.jsonl" ]; then
case "$CASE" in
copy_sigterm|copy_sigint) exec "$TEST_PYTHON" "$TEST_ROOT/pending_srun.py" copy ;;
copy_failure) exit 29 ;;
esac
fi
exec /bin/cp "$@"''',
                    "squeue": """case "$*" in
*"%u"*) id -un ;;
*"%T"*) echo RUNNING ;;
*"%N"*) echo testnode ;;
*"%i %j"*) if [ -f "$TEST_ROOT/step" ]; then cat "$TEST_ROOT/step"; else touch "$TEST_ROOT/step-lookup-missed"; fi; echo "123.99 unrelated_server" ;;
esac""",
                    "scontrol": """if [ "$1" = show ] && [ "$2" = hostnames ]; then echo testnode; else echo allocation; fi""",
                    "srun": """for arg in "$@"; do
case "$arg" in --job-name=*)
printf '%s' "${@: -1}" > "$TEST_ROOT/server-script-path"
/bin/cp "${@: -1}" "$TEST_ROOT/executed-server"
case "$CASE" in startup_sigterm|startup_sigint|startup_timeout) exec "$TEST_PYTHON" "$TEST_ROOT/pending_srun.py" "${arg#--job-name=}" ;; esac
echo "123.1 ${arg#--job-name=}" > "$TEST_ROOT/step"; echo $$ > "$TEST_ROOT/server-pid"; exec /usr/bin/sleep 30 ;; esac
done
case "$CASE" in
preflight_sigterm|preflight_sigint) exec "$TEST_PYTHON" "$TEST_ROOT/pending_srun.py" ;;
preflight_failure) exit 23 ;;
dataset_replaced) echo "Replaced shared trace" > "$DATASET_PATH/traces.jsonl" ;;
server_replaced) echo "Changed server source" > "$SERVER_SCRIPT" ;;
esac
exit 0""",
                    "scancel": '''echo "$*" >> "$TEST_ROOT/cancelled"
case "$CASE" in client_sigterm|client_sigint|client_timeout)
    [ -f "$TEST_ROOT/client-stopped" ] || touch "$TEST_ROOT/premature-server-cleanup" ;;
audit_sigterm|audit_sigint)
    [ -f "$TEST_ROOT/audit-stopped" ] || touch "$TEST_ROOT/premature-server-cleanup" ;;
esac
[ "$1" = 123.1 ] && kill "$(cat "$TEST_ROOT/server-pid")"''',
                    "curl": """if [ "$CASE" = occupied ]; then exit 0; fi
if [ ! -f "$TEST_ROOT/step" ] || [ "$CASE" = startup_failure ]; then exit 7; fi
echo "{}"
""",
                    "sleep": "exec /usr/bin/sleep 0.05",
                    "python": """if [ "$1" = -c ]; then
 touch "$TEST_ROOT/client-called"
 printf '%s\\0' "$@" > "$TEST_ROOT/client-arguments"
 case "$CASE" in client_sigterm|client_sigint|client_timeout) exec "$TEST_PYTHON" "$TEST_ROOT/client.py" ;; esac
 if [ "$CASE" = hold_marker ]; then touch "$RUN_ROOT/hold-after-client"; fi
 if [ "$CASE" = auditor_replaced ]; then echo '# Changed checkout auditor' > "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx_result.py"; fi
 [ "$CASE" != client_failure ]
else
 printf '%s' "$1" > "$TEST_ROOT/$2-path"
 cp "$1" "$TEST_ROOT/$2-auditor"
 case "$2:$CASE" in
 prepare:prepare_sigterm|prepare:prepare_sigint|audit:audit_sigterm|audit:audit_sigint) exec "$TEST_PYTHON" "$TEST_ROOT/pending_srun.py" "$2" ;;
 prepare:prepare_failure|audit:audit_failure) exit 29 ;;
 esac
 exit 0
fi""",
                }
                for name, body in scripts.items():
                    path = binaries / name
                    path.write_text("#!/bin/bash\n" + body + "\n")
                    path.chmod(0o755)
                (root / "client.py").write_text("""import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ["TEST_ROOT"])
role = "worker" if len(sys.argv) > 1 else "client"
received = None

def stop(signum, frame):
    global received
    received = signum

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
(root / (role + "-pid")).write_text(str(os.getpid()))
if role == "client":
    (root / "client-pgid").write_text(str(os.getpgrp()))
    child = subprocess.Popen([sys.executable, __file__, "worker"])
    while not (root / "worker-ready").exists():
        time.sleep(0.01)
(root / (role + "-ready")).touch()
while received is None:
    time.sleep(0.01)
if role == "client":
    child.wait(timeout=5)
(root / (role + "-stopped")).write_text(str(received))
""")
                (root / "pending_srun.py").write_text("""import os
import signal
import sys
import time
from pathlib import Path

root = Path(os.environ["TEST_ROOT"])
role = "server" if len(sys.argv) > 1 else "preflight"
if len(sys.argv) > 1 and sys.argv[1] in ("prepare", "audit", "copy"):
    role = sys.argv[1]

def stop(signum, frame):
    (root / (role + "-stopped")).write_text(str(signum))
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
(root / (role + "-pid")).write_text(str(os.getpid()))
(root / (role + "-ready")).touch()
if role == "server":
    # Make cleanup miss this step deterministically, then register it late.
    while not (root / "step-lookup-missed").exists():
        time.sleep(0.01)
    (root / "step").write_text("123.1 " + sys.argv[1])
while True:
    time.sleep(0.01)
""")
                server = root / "server.sh"
                server.touch()
                scenario = root / "scenario.json"
                scenario_config = {
                    "name": "agentx",
                    "mode": "smoke",
                    "benchmark_grace_period": 30,
                }
                scenario.write_text(json.dumps(scenario_config))
                trace_content = b'{"trace": "original"}\n'
                (root / "traces.jsonl").write_bytes(trace_content)
                run = root / "output"
                source_root = ROOT.parents[3]
                if case == "auditor_replaced":
                    source_root = root / "checkout"
                    source_auditor = (
                        source_root
                        / "test/agentic_benchmark/kimi_k3/tokenspeed/agentx_result.py"
                    )
                    source_auditor.parent.mkdir(parents=True)
                    source_auditor.write_bytes((ROOT / "agentx_result.py").read_bytes())
                env = dict(
                    os.environ,
                    PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                    TEST_ROOT=str(root),
                    TEST_PYTHON=sys.executable,
                    CASE=case,
                    SLURM_JOB_ID="123",
                    SOURCE_ROOT=str(source_root),
                    SERVER_SCRIPT=str(server),
                    CONTAINER_IMAGE="test-image",
                    CONTAINER_MOUNTS="/tmp:/tmp",
                    CLIENT_PYTHON=str(binaries / "python"),
                    SCENARIO_FILE=str(scenario),
                    MODEL_NAME="model",
                    TOKENIZER_PATH="tokenizer",
                    DATASET_PATH=str(root),
                    CONCURRENCY="1",
                    DURATION="60",
                    SEED="1",
                    API_PORT="8000",
                    READINESS_PATH="/readiness",
                    READINESS_TIMEOUT=(
                        "1" if case in ("startup_failure", "startup_timeout") else "30"
                    ),
                    CLIENT_TIMEOUT="2" if case == "client_timeout" else "90",
                    HOLD_AFTER_RUN=(
                        "1"
                        if case
                        in (
                            "hold",
                            "hold_sigterm",
                            "hold_sigint",
                            "client_sigterm",
                            "client_sigint",
                            "startup_sigterm",
                            "startup_sigint",
                            "preflight_sigterm",
                            "preflight_sigint",
                            "preflight_failure",
                            "copy_sigterm",
                            "copy_sigint",
                            "copy_failure",
                            "prepare_sigterm",
                            "prepare_sigint",
                            "prepare_failure",
                            "audit_sigterm",
                            "audit_sigint",
                        )
                        else "0"
                    ),
                    RUN_ROOT=str(run),
                )
                executing_script = ROOT / "agentx.slurm"
                if case == "spool_copy":
                    executing_script = root / "slurm spool" / "slurm_script"
                    executing_script.parent.mkdir()
                    executing_script.write_bytes(
                        (ROOT / "agentx.slurm").read_bytes()
                        + b"\n# Submitted spool copy.\n"
                    )
                process = subprocess.Popen(
                    ["bash", str(executing_script)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    if case in (
                        "copy_sigterm",
                        "copy_sigint",
                        "prepare_sigterm",
                        "prepare_sigint",
                        "audit_sigterm",
                        "audit_sigint",
                    ):
                        role = case.split("_")[0]
                        deadline = time.monotonic() + 5
                        while (
                            not (root / (role + "-ready")).exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertTrue((root / (role + "-ready")).exists())
                        process.send_signal(
                            signal.SIGTERM
                            if case.endswith("sigterm")
                            else signal.SIGINT
                        )
                    if case in ("preflight_sigterm", "preflight_sigint"):
                        deadline = time.monotonic() + 5
                        while (
                            not (root / "preflight-ready").exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertTrue((root / "preflight-ready").exists())
                        process.send_signal(
                            signal.SIGTERM
                            if case.endswith("sigterm")
                            else signal.SIGINT
                        )
                    if case in ("startup_sigterm", "startup_sigint"):
                        deadline = time.monotonic() + 5
                        while (
                            not (root / "server-ready").exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertTrue((root / "server-ready").exists())
                        self.assertFalse((root / "step").exists())
                        process.send_signal(
                            signal.SIGTERM
                            if case.endswith("sigterm")
                            else signal.SIGINT
                        )
                    if case in ("client_sigterm", "client_sigint"):
                        deadline = time.monotonic() + 5
                        while (
                            not (root / "client-ready").exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertTrue((root / "client-ready").exists())
                        process.send_signal(
                            signal.SIGTERM
                            if case == "client_sigterm"
                            else signal.SIGINT
                        )
                    if case.startswith("hold"):
                        deadline = time.monotonic() + 5
                        while (
                            not (run / "allocation-held.txt").exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.05)
                        self.assertTrue((run / "allocation-held.txt").exists())
                        self.assertIsNone(process.poll())
                        if case in ("hold_sigterm", "hold_sigint"):
                            process.send_signal(
                                signal.SIGTERM
                                if case == "hold_sigterm"
                                else signal.SIGINT
                            )
                        else:
                            (run / "release-requested").touch()
                    stdout, stderr = process.communicate(timeout=8)
                    self.assertEqual(
                        process.returncode == 0,
                        case
                        in (
                            "success",
                            "spool_copy",
                            "dataset_replaced",
                            "auditor_replaced",
                            "server_replaced",
                            "hold",
                            "hold_marker",
                        ),
                        stderr,
                    )
                    if case.endswith(("sigterm", "sigint")) or case == "client_timeout":
                        expected = (
                            124
                            if case == "client_timeout"
                            else 143 if case.endswith("sigterm") else 130
                        )
                        self.assertEqual(process.returncode, expected)
                        self.assertEqual(
                            int((run / "client-exit-code.txt").read_text()), expected
                        )
                    if case in ("client_sigterm", "client_sigint", "client_timeout"):
                        expected_signal = (
                            signal.SIGTERM
                            if case == "client_sigterm"
                            else signal.SIGINT
                        )
                        for role in ("client", "worker"):
                            self.assertEqual(
                                int((root / (role + "-stopped")).read_text()),
                                expected_signal,
                            )
                            with self.assertRaises(ProcessLookupError):
                                os.kill(int((root / (role + "-pid")).read_text()), 0)
                        self.assertFalse((run / "allocation-held.txt").exists())
                        self.assertFalse((root / "premature-server-cleanup").exists())
                    if case in ("startup_sigterm", "startup_sigint", "startup_timeout"):
                        if case == "startup_timeout":
                            self.assertEqual(process.returncode, 1)
                            self.assertEqual(
                                (run / "client-exit-code.txt").read_text().strip(), "1"
                            )
                        self.assertTrue((root / "step-lookup-missed").exists())
                        self.assertEqual(
                            int((root / "server-stopped").read_text()), signal.SIGTERM
                        )
                        with self.assertRaises(ProcessLookupError):
                            os.kill(int((root / "server-pid").read_text()), 0)
                        self.assertFalse((root / "cancelled").exists())
                        self.assertFalse((root / "client-called").exists())
                        self.assertFalse((run / "allocation-held.txt").exists())
                    elif case.startswith(("prepare", "copy")):
                        self.assertFalse((root / "server-pid").exists())
                        self.assertFalse((root / "step").exists())
                        self.assertFalse((root / "cancelled").exists())
                        self.assertFalse((root / "client-called").exists())
                    elif case.startswith("preflight"):
                        if case == "preflight_failure":
                            self.assertEqual(process.returncode, 23)
                            self.assertEqual(
                                (run / "client-exit-code.txt").read_text().strip(), "23"
                            )
                        else:
                            self.assertEqual(
                                int((root / "preflight-stopped").read_text()),
                                signal.SIGTERM,
                            )
                            with self.assertRaises(ProcessLookupError):
                                os.kill(int((root / "preflight-pid").read_text()), 0)
                        self.assertFalse((root / "server-pid").exists())
                        self.assertFalse((root / "step").exists())
                        self.assertFalse((root / "cancelled").exists())
                        self.assertFalse((root / "client-called").exists())
                        self.assertFalse((run / "allocation-held.txt").exists())
                    elif case == "occupied":
                        self.assertFalse((root / "step").exists())
                    else:
                        self.assertEqual(
                            (root / "cancelled").read_text().splitlines(), ["123.1"]
                        )
                    if (
                        case.startswith(("prepare", "audit", "copy"))
                        and case != "auditor_replaced"
                    ):
                        role = case.split("_")[0]
                        if case.endswith("failure"):
                            self.assertEqual(process.returncode, 29)
                            self.assertEqual(
                                (run / "client-exit-code.txt").read_text().strip(), "29"
                            )
                        else:
                            self.assertEqual(
                                int((root / (role + "-stopped")).read_text()),
                                signal.SIGTERM,
                            )
                            with self.assertRaises(ProcessLookupError):
                                os.kill(int((root / (role + "-pid")).read_text()), 0)
                        self.assertFalse((run / "allocation-held.txt").exists())
                        self.assertFalse((root / "premature-server-cleanup").exists())
                    if case == "auditor_replaced":
                        self.assertEqual(
                            (root / "prepare-path").read_text(),
                            str(run / "agentx_result.py"),
                        )
                        self.assertEqual(
                            (root / "audit-path").read_text(),
                            str(run / "agentx_result.py"),
                        )
                        self.assertEqual(
                            (root / "prepare-auditor").read_bytes(),
                            (root / "audit-auditor").read_bytes(),
                        )
                        self.assertNotEqual(
                            source_auditor.read_bytes(),
                            (root / "audit-auditor").read_bytes(),
                        )
                    if (root / "server-script-path").exists():
                        self.assertEqual(
                            (root / "server-script-path").read_text(),
                            str(run / "server.sh"),
                        )
                        self.assertEqual(
                            (root / "executed-server").read_bytes(),
                            (run / "server.sh").read_bytes(),
                        )
                        if case == "server_replaced":
                            self.assertNotEqual(
                                server.read_bytes(),
                                (root / "executed-server").read_bytes(),
                            )
                    self.assertEqual(
                        (run / "harness.slurm").read_bytes(),
                        executing_script.read_bytes(),
                    )
                    if case == "spool_copy":
                        self.assertNotEqual(
                            (run / "harness.slurm").read_bytes(),
                            (ROOT / "agentx.slurm").read_bytes(),
                        )
                    if (root / "client-called").exists():
                        arguments = (
                            (root / "client-arguments")
                            .read_bytes()
                            .decode()
                            .split("\0")
                        )
                        self.assertEqual(
                            json.loads(arguments[arguments.index("--scenario") + 1]),
                            scenario_config,
                        )
                        dataset_path = Path(
                            arguments[arguments.index("--dataset-path") + 1]
                        )
                        self.assertEqual(dataset_path, run / "dataset")
                        self.assertEqual(
                            (dataset_path / "traces.jsonl").read_bytes(), trace_content
                        )
                        if case == "dataset_replaced":
                            self.assertNotEqual(
                                (root / "traces.jsonl").read_bytes(), trace_content
                            )
                    if case in ("occupied", "startup_failure"):
                        self.assertFalse((root / "client-called").exists())
                finally:
                    if (root / "client-pgid").exists():
                        try:
                            os.killpg(
                                int((root / "client-pgid").read_text()), signal.SIGKILL
                            )
                        except ProcessLookupError:
                            pass
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=5)
                    # Also reap the fake server if the behavior under test leaked it.
                    for role in ("server", "preflight", "prepare", "audit", "copy"):
                        if (root / (role + "-pid")).exists():
                            try:
                                os.kill(
                                    int((root / (role + "-pid")).read_text()),
                                    signal.SIGKILL,
                                )
                            except ProcessLookupError:
                                pass


if __name__ == "__main__":
    unittest.main()
