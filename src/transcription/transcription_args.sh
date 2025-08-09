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

PODCASTS_PATH="/home/nikita/Balalaika100H"
NUM_WORKERS=1
MODEL_NAMES=('giga_ctc' 'giga_rnnt' 'vosk' 'ton')
LM_PATH="/home/nikita/yapoddataset/ru.lm.bin"
WITH_TIMESTAMPS=True

python -m src.transcription.transcription \
    --podcasts_path "$PODCASTS_PATH" \
    --num_workers "$NUM_WORKERS" \
    --model_names "$MODEL_NAMES" \
    --lm_path "$LM_PATH" \
    --with_timestamps "$WITH_TIMESTAMPS" 
