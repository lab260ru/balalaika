#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${BALALAIKA_IMAGE:-balalaika:cuda13.0}

exec docker build \
    --file "$repo_root/Dockerfile" \
    --tag "$image" \
    --build-arg "CUDA_BASE=docker.io/nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1" \
    --build-arg "REQUIREMENTS_FILE=requirements_dev.cuda130.txt" \
    --build-arg "ONNXRUNTIME_GPU_VERSION=1.28.0" \
    --build-arg "BALALAIKA_UID=${BALALAIKA_UID:-10001}" \
    --build-arg "BALALAIKA_GID=${BALALAIKA_GID:-10001}" \
    "$repo_root"
