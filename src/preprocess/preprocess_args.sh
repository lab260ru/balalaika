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

PODCASTS_PATH="../../../balalaika"
DURATION=15
DEVICE="cuda"
NUM_WORKERS=2
WHISPER_MODEL="large-v3"
COMPUTE_TYPE="float16"
BEAM_SIZE=5

python3 -m src.preprocess.preprocess \
    --podcasts_path "$PODCASTS_PATH" \
    --duration "$DURATION" \
    --device "$DEVICE" \
    --num_workers "$NUM_WORKERS" \
    --whisper_model "$WHISPER_MODEL" \
    --compute_type "$COMPUTE_TYPE" \
    --beam_size "$BEAM_SIZE"


