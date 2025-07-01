from dataset import BalalaikaDataset

if __name__ == "__main__":
    dataset = BalalaikaDataset(
        podcasts_path='../Balalaika100H',
        parquet_path='../balalaika/balalaika.parquet'
    )
