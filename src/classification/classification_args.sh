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

MODEL_PATH="$SCRIPT_DIR/voxblink2_samresnet100_ft"
PODCASTS_PATH=""../../../podcasts""
THRESHOLD=0.8

python -m src.classificatoin.classificatoin \
    --podcasts_path "$PODCASTS_PATH" \
    --model_path "$MODEL_PATH" \
    --threshold "$THRESHOLD" \
