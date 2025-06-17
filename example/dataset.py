import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset


class BalalaikaDataset(Dataset):
    def __init__(
        self,
        podcasts_path: str,
        parquet_path: str,
        audio_key_column: str = "audio_path"
    ):
        self.podcasts_path = podcasts_path
        self.audio_key_column = audio_key_column
    
        self.metadata = pd.read_parquet(parquet_path)

        available_files = {
            str(path) for path in Path(self.podcasts_path).rglob('*.mp3')
        }

        self.valid_items = []
        for idx, row in self.metadata.iterrows():
            audio_path_in_meta = row[self.audio_key_column]
            if not os.path.isabs(audio_path_in_meta):
                full_audio_path = os.path.join(self.podcasts_path, audio_path_in_meta)
            else:
                full_audio_path = audio_path_in_meta

            if full_audio_path in available_files:
                self.valid_items.append((idx, full_audio_path))
            else:
                continue

        print(f"Найдено {len(self.valid_items)} совпадений между Parquet и аудиофайлами")

    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, idx):
        meta_idx, audio_path = self.valid_items[idx]
        row = self.metadata.iloc[meta_idx]
        return audio_path, row.to_dict()