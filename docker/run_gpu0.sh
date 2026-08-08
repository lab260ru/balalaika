#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${BALALAIKA_IMAGE:-balalaika:cuda12.8}
config_path=${BALALAIKA_HOST_CONFIG:-$repo_root/configs/config.yaml}
models_root=${BALALAIKA_HOST_MODELS:-$repo_root/models}
cache_root=${BALALAIKA_HOST_CACHE:-$HOME/.cache/balalaika}
output_root=${BALALAIKA_HOST_OUTPUT:-$repo_root/.docker-output}
logs_root=${BALALAIKA_HOST_LOGS:-$output_root/logs}
model_mount_mode=${BALALAIKA_MODEL_MOUNT_MODE:-ro}
run_uid=${BALALAIKA_RUN_UID:-$(id -u)}
run_gid=${BALALAIKA_RUN_GID:-$(id -g)}

if [[ "$model_mount_mode" != "ro" && "$model_mount_mode" != "rw" ]]; then
    echo "BALALAIKA_MODEL_MOUNT_MODE must be 'ro' or 'rw'" >&2
    exit 2
fi
if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 2
fi
if [[ ! -d "$models_root" ]]; then
    echo "Models directory not found: $models_root" >&2
    exit 2
fi

if (($# > 0)) && [[ "$1" == "pipeline" || "$1" == "warmup" ]] \
    && [[ -z "${BALALAIKA_HOST_DATA:-}" ]]; then
    echo "BALALAIKA_HOST_DATA is required for '$1'" >&2
    exit 2
fi

mkdir -p \
    "$cache_root/home" \
    "$cache_root/ruaccent" \
    "$output_root" \
    "$logs_root"

docker_args=(
    run
    --rm
    --gpus device=0
    --user "$run_uid:$run_gid"
    --shm-size "${BALALAIKA_SHM_SIZE:-8g}"
    --env CUDA_VISIBLE_DEVICES=0
    --env BALALAIKA_CONFIG_PATH=/config/config.yaml
    --env BALALAIKA_MODELS_ROOT=/models
    --env BALALAIKA_OUTPUT_ROOT=/output
    --mount "type=bind,source=$config_path,target=/config/config.yaml,readonly"
    --mount "type=bind,source=$cache_root,target=/cache"
    --mount "type=bind,source=$logs_root,target=/logs"
    --mount "type=bind,source=$output_root,target=/output"
)

model_mount="type=bind,source=$models_root,target=/models"
if [[ "$model_mount_mode" == "ro" ]]; then
    model_mount+=",readonly"
fi
docker_args+=(--mount "$model_mount")

if [[ -n "${BALALAIKA_HOST_DATA:-}" ]]; then
    if [[ ! -d "$BALALAIKA_HOST_DATA" ]]; then
        echo "BALALAIKA_HOST_DATA is not a directory: $BALALAIKA_HOST_DATA" >&2
        exit 2
    fi
    docker_args+=(
        --env BALALAIKA_DATA_ROOT=/data
        --mount "type=bind,source=$BALALAIKA_HOST_DATA,target=/data"
    )
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=(--env HF_TOKEN)
fi
if [[ -n "${YANDEX_KEY:-}" ]]; then
    docker_args+=(--env YANDEX_KEY)
fi

if (($# == 0)); then
    set -- smoke
fi

exec docker "${docker_args[@]}" "$image" "$@"
