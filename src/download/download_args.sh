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

activate_venv ".dev_venv"

SCRIPT_DIR=$(dirname "$(realpath "$0")")


PODCASTS_PATH="../../../podcasts"  
EPISODES_LIMIT=2
NUM_WORKERS=2

python3 -m src.download.download \
    --podcasts_path "$PODCASTS_PATH" \
    --episodes_limit "$EPISODES_LIMIT" \
    --num_workers "$NUM_WORKERS"