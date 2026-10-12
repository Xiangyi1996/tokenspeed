#!/usr/bin/env bash
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

# Run via sbatch with a native client, or from the login host in a held job.
# All paths must be absolute and visible where they are used. See agentx.md.
# Arguments after this script are passed unchanged to the server configuration.
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch or an existing allocation}"

validate_config() {
    : "${SOURCE_ROOT:?absolute checkout path; independent of sbatch submit directory}"
    : "${CONFIG_NAMES:?space-separated server configuration names}"
    read -r -a CONFIGS <<< "$CONFIG_NAMES"
    (( ${#CONFIGS[@]} > 0 ))
    for config in "${CONFIGS[@]}"; do
        case "$config" in attn_tp8_moe_tp8|attn_tp8_moe_ep8) ;; *) echo "Unknown config: $config" >&2; exit 2 ;; esac
    done
    declare -A seen=()
    for config in "${CONFIGS[@]}"; do
        [[ ! -v "seen[$config]" ]] || { echo "Duplicate config: $config" >&2; exit 2; }
        seen[$config]=1
    done
    : "${SERVER_VENV?server venv path; set empty to use the image environment}"
    : "${CONTAINER_IMAGE:?pinned server image}"
    : "${CONTAINER_MOUNTS:?explicit shared mounts}"
    : "${CLIENT_PYTHON:?absolute Python from the native EvalScope agentx venv}"
    : "${SCENARIO_FILE:?absolute EvalScope AgentX scenario JSON}"
    : "${MODEL_NAME:?served model name}"
    : "${TOKENIZER_PATH:?pinned tokenizer path or ID}"
    : "${DATASET_PATH:?local directory containing traces.jsonl}"
    : "${CONCURRENCY:?root session concurrency}"
    : "${DURATION:?measurement seconds}"
    : "${SEED:?client random seed}"
    : "${API_PORT:?server API port}"
    : "${READINESS_PATH:?readiness endpoint, including leading slash}"
    : "${READINESS_TIMEOUT:?maximum server startup seconds}"
    : "${CLIENT_TIMEOUT:?maximum total client wall time seconds}"
    : "${RUN_ROOT:?new absolute output directory}"
    for variable in SOURCE_ROOT CLIENT_PYTHON SCENARIO_FILE DATASET_PATH RUN_ROOT; do
        [[ "${!variable}" = /* ]] || { echo "$variable must be absolute" >&2; exit 2; }
    done
    for variable in CONCURRENCY DURATION API_PORT READINESS_TIMEOUT CLIENT_TIMEOUT; do
        [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
    done
    SCRIPT_DIR="$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed"
    [[ -f "$SCENARIO_FILE" && -x "$CLIENT_PYTHON" ]]
    for config in "${CONFIGS[@]}"; do
        [[ -f "$SCRIPT_DIR/configs/$config.sh" ]]
    done
    [[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%u')" = "$(id -un)" ]] || { echo 'Allocation not owned by current user' >&2; exit 1; }
    [[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%T')" = RUNNING ]] || { echo 'Allocation must be RUNNING' >&2; exit 1; }
}

STEP_NAME="agentx_server_${SLURM_JOB_ID}_$$"
SERVER_PID=
PREFLIGHT_PID=
CLIENT_PID=
HELPER_PID=
CLIENT_SIGNAL=TERM

finish() {
    local result=$?
    # Stop and reap the client before releasing its server. GNU timeout forwards
    # the signal to the client's process group and retains its kill-after bound.
    trap '' INT TERM
    trap - EXIT
    if [[ -n "$CLIENT_PID" ]]; then
        kill -s "$CLIENT_SIGNAL" "$CLIENT_PID" 2>/dev/null || true
        wait "$CLIENT_PID" 2>/dev/null || true
        CLIENT_PID=
    fi
    if [[ -n "$PREFLIGHT_PID" ]]; then
        kill -TERM "$PREFLIGHT_PID" 2>/dev/null || true
        wait "$PREFLIGHT_PID" 2>/dev/null || true
        PREFLIGHT_PID=
    fi
    if [[ -n "$HELPER_PID" ]]; then
        kill -TERM "$HELPER_PID" 2>/dev/null || true
        wait "$HELPER_PID" 2>/dev/null || true
        HELPER_PID=
    fi
    # Finish cleanup even if another signal arrives.
    printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"
    # Finish the owned-step cleanup even if a second signal arrives.
    trap '' INT TERM
    stop_server
    printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"
    exit "$result"
}

stop_server() {
    # Restrict cleanup to the unique step this invocation started.
    while read -r step name; do
        if [[ "$name" = "$STEP_NAME" ]]; then scancel "$step" || true; fi
    done < <(squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j')
    if [[ -n "$SERVER_PID" ]]; then
        # The tracked srun can still be creating its step and absent from squeue.
        # TERM also aborts that startup path; do not wait on an unsignalled srun.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

run_tracked() {
    local helper_result=0
    "$@" &
    HELPER_PID=$!
    wait "$HELPER_PID" || helper_result=$?
    HELPER_PID=
    return "$helper_result"
}

prepare_run() {
    # Atomic reservation: never overwrite evidence or run the same output twice.
    mkdir "$RUN_ROOT"
    export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
    # sbatch executes a spool copy that can differ from the current checkout.
    run_tracked cp -- "${BASH_SOURCE[0]}" "$RUN_ROOT/harness.slurm"
    printf '%s\n' "$AGENTX_SCENARIO_JSON" > "$RUN_ROOT/scenario.json"
    mkdir "$RUN_ROOT/dataset"
    run_tracked cp -- "$DATASET_PATH/traces.jsonl" "$RUN_ROOT/dataset/traces.jsonl"
    run_tracked cp -- "$SCRIPT_DIR/agentx_result.py" "$RUN_ROOT/agentx_result.py"
    run_tracked cp -- "$SERVER_SCRIPT" "$RUN_ROOT/server.sh"
    run_tracked "$CLIENT_PYTHON" "$RUN_ROOT/agentx_result.py" prepare "$RUN_ROOT" "$@"
}

setup_slurm() {
    scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
    mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o '%N')")
    [[ "${#NODES[@]}" -gt 0 ]]
    export HEAD="${NODES[0]}"
    BASE_URL="http://${HEAD}:${API_PORT}"
    SRUN=(srun --jobid="$SLURM_JOB_ID" --overlap --nodes="${#NODES[@]}" --ntasks-per-node=1
        --container-image="$CONTAINER_IMAGE" --container-name="agentx_${SLURM_JOB_ID}_$$"
        --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$SOURCE_ROOT"
        --container-remap-root --no-container-mount-home --container-writable)
}

check_resources() {
    # Any HTTP response on this port means another service owns it.
    if curl --silent --max-time 3 --output /dev/null "$BASE_URL/v1/models"; then
        echo 'API port already occupied; refusing to use or stop that service' >&2
        exit 1
    fi
    "${SRUN[@]}" --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        occupied=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        if [[ -n "$occupied" ]]; then echo "GPUs already occupied: $occupied" >&2; exit 1; fi
        nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
    ' > "$RUN_ROOT/gpu-preflight.log" 2>&1 &
    PREFLIGHT_PID=$!
    PREFLIGHT_RESULT=0
    wait "$PREFLIGHT_PID" || PREFLIGHT_RESULT=$?
    PREFLIGHT_PID=
    if (( PREFLIGHT_RESULT != 0 )); then exit "$PREFLIGHT_RESULT"; fi
}

launch_server() {
    "${SRUN[@]}" --job-name="$STEP_NAME" --mpi=pmix --kill-on-bad-exit=1 \
        bash -c '
            set -euo pipefail
            if [[ -n "$SERVER_VENV" ]]; then source "$SERVER_VENV/bin/activate"; fi
            exec bash "$RUN_ROOT/server.sh" "$@"
        ' agentx-server "$@" > "$RUN_ROOT/server.log" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    START=$SECONDS
    until curl --fail --silent --max-time 5 --output /dev/null "$BASE_URL$READINESS_PATH"; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server step exited before readiness' >&2; exit 1; }
        (( SECONDS - START < READINESS_TIMEOUT )) || { echo 'Server readiness timeout' >&2; exit 1; }
        sleep 5
    done
    printf '%s\n' "$BASE_URL" > "$RUN_ROOT/endpoint.txt"
    curl --fail --silent --max-time 10 "$BASE_URL/v1/models" > "$RUN_ROOT/models.json"
}

run_benchmark() {
    # The client runs on the orchestrating host, not implicitly inside the GPU image.
    # AIPerf 0.12 treats even offline=0 as true and mishandles local tokenizer paths.
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    timeout --signal=INT --kill-after=45 "$CLIENT_TIMEOUT" "$CLIENT_PYTHON" -c \
        'from evalscope.cli.cli import run_cmd; run_cmd()' perf \
        --scenario "$AGENTX_SCENARIO_JSON" --model "$MODEL_NAME" \
        --tokenizer-path "$TOKENIZER_PATH" --url "$BASE_URL/v1/chat/completions" \
        --dataset-path "$RUN_ROOT/dataset" --parallel "$CONCURRENCY" --duration "$DURATION" \
        --seed "$SEED" --outputs-dir "$RUN_ROOT/client" > "$RUN_ROOT/client.log" 2>&1 &
    CLIENT_PID=$!
    # Bash runs signal traps promptly while waiting for a tracked background child.
    CLIENT_RESULT=0
    wait "$CLIENT_PID" || CLIENT_RESULT=$?
    CLIENT_PID=
    if (( CLIENT_RESULT != 0 )); then exit "$CLIENT_RESULT"; fi
}

audit_results() {
    run_tracked "$CLIENT_PYTHON" "$RUN_ROOT/agentx_result.py" audit "$RUN_ROOT"
}

# Keep the configuration sweep beside the launch lifecycle, like agentic_bench.slurm.
validate_config
SWEEP_ROOT=$RUN_ROOT
mkdir "$SWEEP_ROOT"
# Every run and its manifest consume this same immutable scenario value.
AGENTX_SCENARIO_JSON=$(cat "$SCENARIO_FILE")
readonly AGENTX_SCENARIO_JSON
export AGENTX_SCENARIO_JSON
for CONFIG in "${CONFIGS[@]}"; do
    export SERVER_SCRIPT="$SCRIPT_DIR/configs/$CONFIG.sh"
    export RUN_ROOT="$SWEEP_ROOT/$CONFIG"
    STEP_NAME="agentx_server_${SLURM_JOB_ID}_$$_${CONFIG}"
    trap finish EXIT
    trap 'CLIENT_SIGNAL=INT; exit 130' INT
    trap 'CLIENT_SIGNAL=TERM; exit 143' TERM
    prepare_run "$@"
    setup_slurm
    check_resources
    launch_server "$@"
    wait_for_ready
    run_benchmark
    audit_results
    stop_server
    printf '0\n' > "$RUN_ROOT/client-exit-code.txt"
    trap - EXIT INT TERM
done
