# bin/bash

wget https://huggingface.co/datasets/MTUCI/Balalaika100H/resolve/main/Balalaika100H.parquet
wget https://huggingface.co/datasets/MTUCI/Balalaika100H/resolve/main/Balalaika100H.pkl

PODCASTS_PATH="../Balalaika100H"
PICKLE_PATH="/home/nikita/yapoddataset/Balalaika100H.pkl"
PARQUET_PATH="/home/nikita/yapoddataset/Balalaika100H.parquet"
NUM_WORKERS=4

bash src/download/download_prepared.sh $PODCASTS_PATH $PICKLE_PATH $NUM_WORKERS
bash src/recovery_from_meta_yamls.sh $PODCASTS_PATH $PARQUET_PATH $NUM_WORKERS