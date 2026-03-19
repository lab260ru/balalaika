#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

if [[ ! -f ".dev_venv/bin/activate" ]]; then
    echo "Missing virtual environment: ${REPO_ROOT}/.dev_venv" >&2
    exit 1
fi

source ".dev_venv/bin/activate"

TARGET="${TARGET:-${1:-}}"
if [[ -z "${TARGET}" ]]; then
    echo "Usage: TARGET=<target> DATASET=/path/to/data ${SCRIPT_DIR}/run_benchmark.sh" >&2
    echo "Or: ${SCRIPT_DIR}/run_benchmark.sh <target>" >&2
    echo "Use TARGET=list to print available benchmark targets." >&2
    exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/config.yaml}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-3}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmarking/reports}"

ARGS=(
    "--config-path" "${CONFIG_PATH}"
    "--sample-mode" "${SAMPLE_MODE}"
    "--seed" "${SEED}"
    "--repeats" "${REPEATS}"
    "--warmup-repeats" "${WARMUP_REPEATS}"
    "--sample-interval-sec" "${SAMPLE_INTERVAL_SEC}"
    "--output-root" "${OUTPUT_ROOT}"
)

if [[ "${TARGET}" == "list" ]]; then
    exec python3 "${SCRIPT_DIR}/bench.py" --list-targets
fi

ARGS+=("--target" "${TARGET}")

if [[ -n "${DATASET:-}" ]]; then
    ARGS+=("--dataset" "${DATASET}")
fi

if [[ -n "${NUM_SAMPLES:-}" ]]; then
    ARGS+=("--num-examples" "${NUM_SAMPLES}")
fi

if [[ -n "${GPU_IDS:-}" ]]; then
    ARGS+=("--gpu-ids" "${GPU_IDS}")
elif [[ -n "${NUM_GPUS:-}" ]]; then
    ARGS+=("--num-gpus" "${NUM_GPUS}")
fi

if [[ -n "${CPU_WORKERS_PER_GPU:-}" ]]; then
    ARGS+=("--cpu-workers-per-gpu" "${CPU_WORKERS_PER_GPU}")
fi

if [[ -n "${CPU_WORKERS_TOTAL:-}" ]]; then
    ARGS+=("--cpu-workers-total" "${CPU_WORKERS_TOTAL}")
fi

if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
    ARGS+=("--batch-size-override" "${BATCH_SIZE_OVERRIDE}")
fi

if [[ -n "${MODEL_NAME_OVERRIDE:-}" ]]; then
    ARGS+=("--model-name-override" "${MODEL_NAME_OVERRIDE}")
fi

if [[ "${DISABLE_DIARIZATION:-0}" == "1" ]]; then
    ARGS+=("--disable-diarization")
fi

if [[ "${KEEP_WORKDIRS:-0}" == "1" ]]; then
    ARGS+=("--keep-workdirs")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ARGS+=("--dry-run")
fi

exec python3 "${SCRIPT_DIR}/bench.py" "${ARGS[@]}"
