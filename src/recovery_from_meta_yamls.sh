#!/bin/bash

activate_venv() {
    local venv_path=$1
    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "Error: Virtual environment not found at $venv_path"
        exit 1
    fi
    source "$venv_path/bin/activate"
    echo "Activated: $(which python)"
}

PODCASTS_PATH=$(realpath "$1")
PARQUET_PATH=$(realpath "$2")
NUM_WORKERS="$3"

VENV_PATH=".user_venv"
activate_venv "$VENV_PATH"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

python3 -m src.recovery_from_meta --podcasts_path $PODCASTS_PATH --parquet_path $PARQUET_PATH --num_workers $NUM_WORKERS