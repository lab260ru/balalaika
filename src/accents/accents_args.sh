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

PODCASTS_PATH="../../../balalaika"
MODEL_NAME="turbo3.1"
NUM_WORKERS=4

python3 -m src.accents.accents \
    --podcasts_path "$PODCASTS_PATH" \
    --model_name "$MODEL_NAME" \
    --num_workers "$NUM_WORKERS"
