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

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH=$(realpath "$1")

activate_venv ".dev_venv"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

taskset -c 0-24 python3 -m src.phonemizer.phonemizer --config_path "$CONFIG_PATH"