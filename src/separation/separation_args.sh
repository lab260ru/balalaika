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
NISQA_CONFIG_PATH="$SCRIPT_DIR/../../configs/nisqa_config.yaml"
PODCASTS_PATH="../../../balalaika"
USE_NISQA="True"
USE_MONO="True"
ONE_SPEAKER="False"
NUM_WORKERS="4"

python3 -m src.separation.separation \
    --config_path "$CONFIG_PATH" \
    --nisqa_config "$NISQA_CONFIG_PATH" \
    --podcasts_path "$PODCASTS_PATH" \
    --use_nisqa "$USE_NISQA" \
    --use_mono "$USE_MONO" \
    --one_speaker "$ONE_SPEAKER" \
    --num_workers "$NUM_WORKERS"