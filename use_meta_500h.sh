# bin/bash


activate_venv() {
    local venv_path=$1
    if [ ! -f "$venv_path/bin/activate" ]; then
        echo "Error: Virtual environment not found at $venv_path"
        exit 1
    fi
    source "$venv_path/bin/activate"
    echo "Activated: $(which python)"
}

wget https://huggingface.co/datasets/MTUCI/Balalaika500H/resolve/main/Balalaika500H.parquet
wget https://huggingface.co/datasets/MTUCI/Balalaika500H/resolve/main/Balalaika500H.pkl

activate_venv ".user_venv"

PODCASTS_PATH="../Balalaika500H"
PICKLE_PATH="Balalaika500H.pkl"
PARQUET_PATH="Balalaika500H.parquet"
NUM_WORKERS=4

bash src/download/download_prepared.sh $PODCASTS_PATH $PICKLE_PATH $NUM_WORKERS
bash src/recovery_from_meta_yamls.sh $PODCASTS_PATH $PARQUET_PATH $NUM_WORKERS