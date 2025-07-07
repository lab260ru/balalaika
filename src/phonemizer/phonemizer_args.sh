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
CONFIG_PATH="$SCRIPT_DIR/../../configs/config.yaml"

PODCASTS_PATH="../../../balalaika"
NUM_WORKERS=8
DEVICE="cuda" 

python3 -m src.phonemizer.phonemizer \
    --podcasts_path "$PODCASTS_PATH" \
    --num_workers "$NUM_WORKERS"