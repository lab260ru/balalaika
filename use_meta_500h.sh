# bin/bash

# wget ... (download meta .parquet )
# wget ... (download meta .pickle )

PODCASTS_PATH="../Balalaika500H"
PICKLE_PATH="500hBalalaika.pkl"
PARQUET_PATH="/home/nikita/balalaika/balalaika.parquet"

bash src/download/download_prepared.sh $PODCASTS_PATH $PICKLE_PATH
bash src/recovery_from_meta_yamls.sh $PODCASTS_PATH $PARQUET_PATH