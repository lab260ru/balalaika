#!/bin/bash
# Common runtime bootstrap for individual stage wrappers under src/*.
#
# Usage from a wrapper script:
#   . "$(dirname "$(realpath "$0")")/../stage_runner.sh"
#   stage_init "$@"
#   stage_run python_module [extra_args...]
#
# Sources runtime env from configs/config.yaml, activates the venv, and exposes
# `stage_run` which adds CPU pinning + --log_dir consistently.
set -euo pipefail

BALALAIKA_CONFIG_PATH=""

stage_init() {
    if [ -z "${1:-}" ]; then
        echo "Usage: $0 <config_path>" >&2
        exit 1
    fi
    BALALAIKA_CONFIG_PATH="$(realpath "$1")"

    if ! eval "$(python3 -m src.utils.runtime_env --config_path "$BALALAIKA_CONFIG_PATH")"; then
        echo "Failed to read runtime env from $BALALAIKA_CONFIG_PATH" >&2
        exit 1
    fi

    local venv="${BALALAIKA_VENV:-.dev_venv}"
    if [ ! -f "$venv/bin/activate" ]; then
        echo "Error: virtualenv not found at $venv" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$venv/bin/activate"

    # Mirror base.sh: prepend the venv's bundled NVIDIA / TensorRT shared
    # libraries to LD_LIBRARY_PATH so ONNX Runtime's CUDA/TensorRT execution
    # providers can dlopen them when stages launch via the *_yaml.sh wrappers.
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local nvidia_base="$venv/lib/python$python_version/site-packages/nvidia"
    if [ -d "$nvidia_base" ]; then
        export LD_LIBRARY_PATH="${nvidia_base}/cublas/lib:${nvidia_base}/cudnn/lib:${nvidia_base}/cuda_runtime/lib:${nvidia_base}/cuda_nvrtc/lib:${nvidia_base}/cufft/lib:${nvidia_base}/nvjitlink/lib:${nvidia_base}/cusolver/lib:${nvidia_base}/cusparse/lib:${LD_LIBRARY_PATH:-}"
    fi
    local trt_libs="$venv/lib/python$python_version/site-packages/tensorrt_libs"
    if [ -d "$trt_libs" ]; then
        export LD_LIBRARY_PATH="${trt_libs}:${LD_LIBRARY_PATH:-}"
    fi

    mkdir -p "${BALALAIKA_LOG_DIR:-./logs}"
}

stage_run() {
    local module="$1"; shift
    local cmd=(python3 -m "$module" --config_path "$BALALAIKA_CONFIG_PATH" --log_dir "$BALALAIKA_LOG_DIR" "$@")
    if [[ -n "${BALALAIKA_CPU_AFFINITY:-}" ]] && command -v taskset >/dev/null 2>&1; then
        cmd=(taskset -c "$BALALAIKA_CPU_AFFINITY" "${cmd[@]}")
    fi
    "${cmd[@]}"
}
