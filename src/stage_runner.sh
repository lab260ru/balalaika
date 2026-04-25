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
