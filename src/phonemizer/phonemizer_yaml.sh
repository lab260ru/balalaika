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

activate_venv ".support_venv"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
CONFIG_PATH="$SCRIPT_DIR/../../configs/config.yaml"

python3 -m  src.phonemizer.phonemizer \
    --config_path "$CONFIG_PATH"