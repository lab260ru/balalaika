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

VENV_PATH=".main_venv"
activate_venv "$VENV_PATH"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

python3 -m src.collate --config_path "$CONFIG_PATH"