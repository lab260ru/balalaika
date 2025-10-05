import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from loguru import logger

from src.utils import get_audio_paths, load_config
from src.libs.NISQA.run_predict import run_nisqa_with_config


def _create_input_csv(audio_files: List[Path], temp_dir: Path) -> Path:
    csv_file_path = temp_dir / 'input_audio_list.csv'
    df = pd.DataFrame({'filepath': [p.resolve() for p in audio_files]})
    df.to_csv(csv_file_path, index=False)
    logger.info(f"A temporary CSV file has been created: {csv_file_path}")
    return csv_file_path


def _save_results(output_dir: Path, final_output_path: Path):
    source_file = output_dir / 'NISQA_results.csv'
    if not source_file.exists():
        logger.warning(f"The NISQA output file was not found: {source_file}")
        return

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        new_df = pd.read_csv(source_file)

        if final_output_path.exists():
            existing_df = pd.read_csv(final_output_path)
            new_files_df = new_df[~new_df['filepath'].isin(existing_df['filepath'])]
            
            if not new_files_df.empty:
                result_df = pd.concat([existing_df, new_files_df], ignore_index=True)
                logger.success(f"Added {len(new_files_df)} new NISQA results.")
            else:
                result_df = existing_df
                logger.info("There are no new files to add from NISQA.")
        else:
            result_df = new_df

        if 'Unnamed: 0' in result_df.columns:
            result_df = result_df.drop('Unnamed: 0', axis=1)

        result_df.to_csv(final_output_path, index=False)
        logger.success(f"Results are saved in: {final_output_path}")

    except Exception as e:
        logger.error(f"Could not process and save the results: {e}")


def get_unprocessed_audio_paths(podcasts_path: Path, result_csv_path: Path) -> List[Path]:
    all_audio_paths = get_audio_paths(str(podcasts_path))
    
    if not result_csv_path.exists():
        logger.info("The results file was not found. All found audio files are processed.")
        return all_audio_paths

    logger.info(f"Checking an existing results file: {result_csv_path}")
    df = pd.read_csv(result_csv_path)
    processed_audio_paths = set(df['filepath'].astype(str).to_list())
    
    unprocessed_paths = [
        path for path in all_audio_paths
        if str(path.resolve()) not in processed_audio_paths
    ]
    
    return unprocessed_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path, 'separation')

    podcasts_path = Path(config.get('podcasts_path', '.'))
    bs = config.get('bs', 16)
    num_workers = config.get('num_workers_nisqa', 2)

    nisqa_dir = Path('./src/libs/NISQA').resolve() 
    pretrained_model = nisqa_dir / 'weights' / 'nisqa.tar'

    final_output_path = podcasts_path / 'balalaika.csv'
    temp_dir = final_output_path.parent / 'nisqa_temp'
    output_dir = temp_dir / 'nisqa_results'
    
    output_dir.mkdir(parents=True, exist_ok=True)

    unprocessed_paths = get_unprocessed_audio_paths(
        podcasts_path=podcasts_path,
        result_csv_path=final_output_path
    )

    if not unprocessed_paths:
        logger.warning("Не найдено аудиофайлов для обработки. Выход.")
        return

    logger.info(f' Found {len(unprocessed_paths)} files to process.')
    csv_file_path = _create_input_csv(unprocessed_paths, temp_dir)

    nisqa_config = {
        'mode': 'predict_csv',
        'pretrained_model': str(pretrained_model),
        'csv_file': str(csv_file_path),
        'csv_deg': 'filepath',
        'output_dir': str(output_dir),
        'bs': bs,
        'num_workers': num_workers,
        'ms_channel': None
    }

    try:
        logger.info(f"Launching NISQA with configuration: {nisqa_config}")
        
        run_nisqa_with_config(nisqa_config)
        
        logger.info("NISQA processing is complete. Saving the results...")
        _save_results(output_dir=output_dir, final_output_path=final_output_path)
        
    except Exception as e:
        logger.error(f"An error occurred during the execution of NISQA: {e}", exc_info=True)


if __name__ == "__main__":
    main()