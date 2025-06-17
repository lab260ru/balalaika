# bin/bash

# wget ... (download meta .parquet )
CONFIG_PATH="configs/config.yaml"

bash src/download/download_yaml.sh
bash src/recovery_from_meta_yamls.sh