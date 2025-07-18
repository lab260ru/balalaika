from dataset import BalalaikaDataset
import time
if __name__ == "__main__":
    dataset = BalalaikaDataset(
        podcasts_path='/home/nikita/balalaika',
        parquet_path='/home/nikita/yapoddataset/Balalaika100H.parquet'
    )
    for item in dataset:
        print(item)
        time.sleep(1)