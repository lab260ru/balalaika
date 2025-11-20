#!/bin/bash
set -euo pipefail

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

SCRIPTS=(
    # "./src/download/download_yaml.sh"
    # "./src/preprocess/preprocess_yaml.sh"
    "./src/separation/separation_yaml.sh"
    "./src/transcription/transcription_yaml.sh"
    "./src/punctuation/punctuation_yaml.sh"
    "./src/accents/accents_yaml.sh"
    "./src/phonemizer/phonemizer_yaml.sh"
    "./src/collate_yamls.sh"
)

activate_venv ".dev_venv"

for script in "${SCRIPTS[@]}"; do
    echo -e "\n\033[1;34m=== Executing $script ===\033[0m"

    if [ ! -f "$script" ]; then
        echo -e "\033[1;31mError: Script $script not found\033[0m"
        exit 1
    fi

    bash "$script" "$CONFIG_PATH" || {
        echo -e "\033[1;31mError in $script\033[0m"
        exit 1
    }
done

echo -e "\n\033[1;32mAll scripts executed successfully!\033[0m"
