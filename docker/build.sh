#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${BALALAIKA_IMAGE:-balalaika:cuda12.8}

exec docker build \
    --file "$repo_root/Dockerfile" \
    --tag "$image" \
    --build-arg "BALALAIKA_UID=${BALALAIKA_UID:-10001}" \
    --build-arg "BALALAIKA_GID=${BALALAIKA_GID:-10001}" \
    "$repo_root"
