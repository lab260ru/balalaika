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
PICKLE_PATH=$(realpath "$2")
NUM_WORKERS="$3"

activate_venv ".user_venv"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

taskset -c 0-24  python3 -m src.download.download_prepared --pickle_path "$PICKLE_PATH" --podcasts_path "$PODCASTS_PATH" --num_workers "$NUM_WORKERS"