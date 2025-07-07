from dataset import BalalaikaDataset
import time
if __name__ == "__main__":
    dataset = BalalaikaDataset(
        podcasts_path='/home/nikita/Balalaika100H',
        parquet_path='/home/nikita/yapoddataset/Balalaika100H.parquet'
    )