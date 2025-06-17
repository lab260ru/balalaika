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


activate_venv ".main_venv"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PODCASTS_PATH=""
NUM_WORKERS=2

python3 -m src.yofication.yofication \
    --podcasts_path "$PODCASTS_PATH" \
    --num_workers "$NUM_WORKERS"
