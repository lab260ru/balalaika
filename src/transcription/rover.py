from pathlib import Path
from typing import List

import pandas as pd
from crowdkit.aggregation import ROVER
from loguru import logger
from tqdm import tqdm 

from src.utils.utils import read_file_content, get_audio_paths

class ROVERWrapper:
    def __init__(self, podcasts_path: str, model_names: List[str]):
        self.podcasts_path = Path(podcasts_path)
        self.model_names = model_names
        self.tokenizer = lambda s: s.lower().split()
        self.detokenizer = lambda tokens: ' '.join(tokens)
        self.rover_aggregator = ROVER(self.tokenizer, self.detokenizer)

    def aggregate_and_save(self):
        logger.info("Starting transcription aggregation based on audio files.")
        
        all_audio_paths = get_audio_paths(str(self.podcasts_path))
        
        if not all_audio_paths:
            logger.warning("Audio files not found. Aggregation finished.")
            return

        records = []
        excluded_patterns = ['_rover', '_phonemes', '_accent']

        for audio_path in tqdm(all_audio_paths, desc="Aggregating transcriptions"):
            if any(pattern in audio_path.stem for pattern in excluded_patterns):
                continue
            
            for model_name in self.model_names:
                suffix = 'vosk' if 'vosk' in model_name else model_name
                transcript_path = audio_path.with_name(f"{audio_path.stem}_{suffix}.txt")

                if not transcript_path.exists():
                    continue
                
                try:
                    text = read_file_content(transcript_path)
                    if not text:
                        continue
                    
                    records.append({
                        'task': str(audio_path),
                        'worker': model_name,
                        'text': text
                    })
                except Exception as e:
                    logger.error(f"Error reading file {transcript_path}: {e}")
        
        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No transcriptions found for aggregation. Check file paths and names.")
            return

        df['text'] = df['text'].str.lower()
        logger.info(f"Running ROVER on {len(df['task'].unique())} unique audio files...")
        result = self.rover_aggregator.fit_predict(df)
        
        logger.info("Saving aggregated results...")
        for task_path, agg_text in result.items():
            audio_path = Path(task_path)
            output_path = audio_path.with_name(f"{audio_path.stem}_rover.txt")
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(agg_text)
            except IOError as e:
                logger.error(f"Failed to write result to {output_path}: {e}")
        
        logger.info("Aggregation complete.")