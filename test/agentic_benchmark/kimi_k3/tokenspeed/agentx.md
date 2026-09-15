# Kimi-K3 AgentX on Slurm

`agentx.slurm` starts one server in a named Pyxis container, waits for readiness,
runs EvalScope AgentX, and audits both request exports and final phase counts.
Unlike `agentic_bench.slurm`, this uses duration-based AgentX replay, not SWE-Smith.
No packages are installed on the GPUs by the harness.

## Prepare the client and server

Use an EvalScope version containing PR #1745 for custom K3 tokenizers:

```bash
python3.12 -m venv /absolute/path/to/client-venv
source /absolute/path/to/client-venv/bin/activate
python -m pip install 'evalscope[agentx] @ git+https://github.com/modelscope/evalscope.git@0fc9b81bc5824a9f9a33a01ce59d675814b2e99e'
python -m pip check
python -m pip freeze --all > /absolute/path/to/client-requirements.txt
```

The client runs on the **orchestrating host**. On ARM GPU nodes, `sbatch` requires
an ARM client venv. To reuse an x86 client on a login host, first obtain a held
allocation and then invoke this script there with `SLURM_JOB_ID` set. A Python
path on shared storage does not make that Python executable cross-architecture.

Prepare a local `traces.jsonl` from the fixed EvalScope 256K dataset; retain its
revision and SHA256. Use the same file for both engines. A smoke run still uses
long context: the server must support 262144 tokens. The supplied
`configs/agentx_tp8.sh` follows the existing TP8/Eagle3 configuration, with 256K
context, an explicit seed and GPU memory fraction, and KVStore disabled.

Create `/absolute/path/to/scenario.json`:

```json
{
  "name": "agentx",
  "mode": "smoke",
  "tokenizer_trust_remote_code": true,
  "num_gpus": 8,
  "engine": "tokenspeed",
  "engine_version": "RECORD_YOUR_SERVER_COMMIT",
  "hardware": "GB300"
}
```

## Configure a run

All choices are explicit. Use paths visible inside the container for server
assets and paths visible on the orchestrating host for client assets. Mount the
output directory into the server container. Do not place private experiment
artifacts in a public commit.

```bash
export SOURCE_ROOT=/absolute/path/to/tokenspeed
export SERVER_SCRIPT="$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/configs/agentx_tp8.sh"
export CONTAINER_IMAGE=/absolute/path/to/pinned-server.sqsh
export CONTAINER_MOUNTS=/shared:/shared
export SERVER_VENV=/opt/server-venv
export MODEL_DIR=/shared/pinned-target-checkpoint
export DRAFT_DIR=/shared/pinned-eagle3-checkpoint
export MODEL_NAME=kimi-k3
export GPU_MEMORY_UTILIZATION=0.9
export SERVER_SEED=20260707
export CLIENT_PYTHON=/absolute/path/to/client-venv/bin/python
export SCENARIO_FILE=/absolute/path/to/scenario.json
export TOKENIZER_PATH="$MODEL_DIR"
export DATASET_PATH=/shared/agentx-dataset
export CONCURRENCY=1
export DURATION=60
export SEED=20260707
export API_PORT=8000
export READINESS_PATH=/readiness
export READINESS_TIMEOUT=7500
export CLIENT_TIMEOUT=3300
export HOLD_AFTER_RUN=1
export RUN_ROOT=/shared/new-agentx-output
```

`RUN_ROOT` must not exist and its parent must exist. `mkdir` reserves it atomically.
The harness rejects an occupied HTTP port or GPUs before launching its server.
Use a dedicated allocation and do not start concurrent independent clients on it.
For an sbatch launch, set `SOURCE_ROOT` explicitly: the script runs from Slurm's
spool directory, which is not the checkout or necessarily the submission directory.

For a native client on the allocated node:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:4 --exclusive \
  --partition=very-long --time=06:00:00 --no-requeue --export=ALL \
  "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm"
```

Adapt account, partition, mounts, image and time limit to the cluster. The example
requests six hours; a partition name does not imply its maximum wall time.

Alternatively, on a login host with an already held allocation:

```bash
export SLURM_JOB_ID=YOUR_HELD_JOB_ID
bash "$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm"
```

With `HOLD_AFTER_RUN=1`, the server remains available after client success or
failure, until `touch "$RUN_ROOT/release-requested"`, server exit or allocation
expiry. SIGINT/SIGTERM stops the invocation's server step. With `HOLD_AFTER_RUN=0`,
only that step is stopped after the run; an independently held allocation survives.
A directly submitted batch allocation ends when its batch script exits.
A controller can also create `RUN_ROOT/hold-after-client` before a successful
client audit to retain that service for more client runs; the same
`release-requested` mechanism applies. This marker does not retain failed runs.

## Benchmark and compare engines

For a formal run, set scenario `mode` to `benchmark`, `DURATION=1800`, and select
a new `RUN_ROOT`. Benchmark mode requires at least 900 seconds. `CLIENT_TIMEOUT`
must also cover dataset reconstruction, warmup and draining; check remaining job
time before starting. Do not compare a warmed smoke result with a cold baseline.
Each fresh harness invocation starts a new server, while the protocol's warmup
still runs. Verify the first warmup's cache-read count when asserting cold KV state.

`CONCURRENCY` is the root session-tree limit; child agents may run concurrently.
Keep trace, tokenizer, checkpoint, draft, seed, context capacity, max sequences,
GPU count, KV precision, memory fraction, idle cap and drain rules aligned.
Record differences in attention/MoE backends and engine-specific features rather
than treating them as matched. Change one engine at a time on the same nodes.

For vLLM, provide a `SERVER_SCRIPT` that uses its verified native Python and launches
one task per node. A system-Python image does not need venv activation; do not
activate the TokenSpeed venv in a vLLM image. Follow the K3 vLLM reference's native multi-node arguments:
`--nnodes "$SLURM_NNODES" --node-rank "$SLURM_NODEID" --master-addr "$HEAD"`,
adding `--headless` only on followers. Use `/health` as `READINESS_PATH`, set the
scenario engine/version accordingly, and use the same served model name.
The reference's 80000 context must be raised to 262144 for this dataset.
Validate K3 modelopt checkpoint support and Eagle3 support in the exact vLLM
image before a long benchmark; image tags alone are not evidence of compatibility.

## Evidence and acceptance

- `manifest.json`: client module hash, package versions, trace hash, launcher hash,
  harness revision and explicit run settings. The harness commit identifies the
  launcher checkout, not necessarily the code installed in the server image.
- `allocation.txt`, `gpu-preflight.log`, `server.log`: allocation and server evidence.
- `client/`: EvalScope summaries, AIPerf raw summary, JSONL and phase logs.
- `audit.json`: successful, cancelled and errored profiling counts.
- `client-exit-code.txt`: command/audit outcome; inspect `audit.json` as well.

AIPerf may omit `error_request_count` when zero and omit cancelled requests from
JSONL. The audit therefore requires the final profiling phase log. Smoke rejects
cancellations. Benchmark records `completed_with_cancellation` for a window/drain
cancellation; report it explicitly and do not count its partial output as success.
`submission_valid` and mirror revalidation are tool fields, not a leaderboard
submission. Report warmup separately and distinguish aggregate output throughput,
per-user decode throughput and request E2E throughput.

## Local tests

```bash
python test/agentic_benchmark/kimi_k3/tokenspeed/test_agentx.py
bash -n test/agentic_benchmark/kimi_k3/tokenspeed/agentx.slurm
```

Tests use fake Slurm/HTTP commands to exercise port conflicts, startup failure,
client failure, hold/release and cleanup isolation; no GPU allocation is created.


## Drain time

AIPerf 0.12.0 waits 30 seconds for outstanding responses after the measurement duration. A long generation can exceed that time even when the engine is healthy. Keep the zero-cancellation smoke audit. With an EvalScope version exposing `benchmark_grace_period`, set it explicitly in the scenario JSON, for example `"benchmark_grace_period":600`, and use the same value for both engines. This option is separate from `request_timeout_seconds`. Ensure `CLIENT_TIMEOUT` covers initialization, measurement and drain. The pinned EvalScope version above does not expose this option; an adapter change is required to use it. Record that change with the client version and preserve both the original failed run and the rerun.

The audit rejects a profiling grace-period timeout even when all exported requests succeeded. Pending replay branches can otherwise leave the client idle until the timeout and distort aggregate throughput. Preserve that run for diagnosis and rerun after resolving the drain.
