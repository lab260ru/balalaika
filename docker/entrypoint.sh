#!/usr/bin/env bash
set -euo pipefail

readonly APP_ROOT="${BALALAIKA_PIPELINE_ROOT:-/opt/balalaika/app}"
readonly GENERATED_CONFIG="/tmp/balalaika/config.yaml"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p /cache/home /cache/ruaccent

render_config() {
    local source_config=$1

    if [[ ! -f "$source_config" ]]; then
        echo "Config file not found: $source_config" >&2
        exit 2
    fi

    python3 "$APP_ROOT/docker/prepare_config.py" \
        --input "$source_config" \
        --output "$GENERATED_CONFIG"
}

run_pipeline() {
    local source_config="${BALALAIKA_CONFIG_PATH:-/config/config.yaml}"
    local args=()

    while (($#)); do
        case "$1" in
            --config_path|--config)
                if (($# < 2)); then
                    echo "$1 requires a path" >&2
                    exit 2
                fi
                source_config=$2
                shift 2
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done

    if [[ ! -f "$source_config" && "$source_config" == "/config/config.yaml" ]]; then
        source_config="$APP_ROOT/configs/config.yaml"
    fi

    render_config "$source_config"
    exec bash "$APP_ROOT/base.sh" \
        --config_path "$GENERATED_CONFIG" \
        "${args[@]}"
}

case "${1:-}" in
    pipeline)
        shift
        run_pipeline "$@"
        ;;
    warmup)
        shift
        source_config="${BALALAIKA_CONFIG_PATH:-/config/config.yaml}"
        if [[ ! -f "$source_config" && "$source_config" == "/config/config.yaml" ]]; then
            source_config="$APP_ROOT/configs/config.yaml"
        fi
        render_config "$source_config"
        exec python3 -m benchmarking.warmup \
            --config_path "$GENERATED_CONFIG" \
            "$@"
        ;;
    smoke)
        shift
        exec python3 "$APP_ROOT/docker/smoke_test.py" "$@"
        ;;
    "")
        exec python3 "$APP_ROOT/docker/smoke_test.py"
        ;;
    *)
        exec "$@"
        ;;
esac
