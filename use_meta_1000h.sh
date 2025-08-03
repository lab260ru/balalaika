#!/bin/bash

activate_venv() {
    local venv_path=$1
    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "Error: Virtual environment not found at $venv_path"
        exit 1
    fi
    source "$venv_path/bin/activate"
    echo "Activated virtual environment: $(which python)"
}

download_if_not_exists() {
    local url=$1
    local filename=$2
    
    if [ ! -f "$filename" ]; then
        echo "Downloading $filename..."
        wget "$url" -O "$filename" || {
            echo "Error: Failed to download $filename"
            exit 1
        }
    else
        echo "$filename already exists, skipping download."
    fi
}

PODCASTS_PATH="Balalaika1000H"
PICKLE_PATH="Balalaika1000H.pkl"
PARQUET_PATH="Balalaika1000H.parquet"
NUM_WORKERS=4

PICKLE_URL="https://huggingface.co/datasets/MTUCI/Balalaika1000H/resolve/main/Balalaika1000H.pkl"
PARQUET_URL="https://huggingface.co/datasets/MTUCI/Balalaika1000H/resolve/main/Balalaika1000H.parquet"

download_if_not_exists "$PICKLE_URL" "$PICKLE_PATH"
download_if_not_exists "$PARQUET_URL" "$PARQUET_PATH"

activate_venv ".user_venv"

bash src/download/download_prepared.sh "$PODCASTS_PATH" "$PICKLE_PATH" "$NUM_WORKERS" 
bash src/recovery_from_meta_yamls.sh "$PODCASTS_PATH" "$PARQUET_PATH" "$NUM_WORKERS"
