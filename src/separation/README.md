## Overview
This module handles the post-processing and quality filtering of the segmented audio chunks. It ensures that only high-quality speech segments are kept for training by applying two main filtering mechanisms: **Music Detection** and **Speech Quality Assessment (DistillMOS)**.

### What it does:
1. **Music Detection**: 
   - Uses a fine-tuned `WavLM` model to scan all generated audio chunks.
   - Calculates the probability of music presence in each segment.
   - **Action**: If the probability exceeds a predefined threshold (e.g., `0.5`), the audio file is considered music/noise and is **permanently deleted** from the disk to clean up the dataset.
   
2. **Quality Scoring (DistillMOS)**:
   - Evaluates the remaining (speech-only) audio chunks using the `DistillMOS` model.
   - Predicts a Mean Opinion Score (MOS) representing the acoustic quality of the speech.
   - **Action**: Appends the calculated `DistillMOS` score to the centralized `balalaika.csv` metadata file.

---

## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`separation_args.sh`) and then run it:
~~~bash
bash separation/separation_args.sh
~~~  

### Running the Code via Config File  
The python scripts are executed by passing a YAML configuration file. Example:
~~~bash
bash separation/separation_yaml.sh config_path
~~~  

---

## Explanation of Parameters

The scripts primarily rely on the YAML configuration file passed via `--config_path`. Key parameters inside the config include:

- `podcasts_path`: Root directory containing the segmented audio files and `balalaika.csv`.
- `music_detect`: Dictionary containing music detection settings:
  - `music_detect_model`: Path to the WavLM music detection model weights.
  - `base_model`: Base HuggingFace model (e.g., `microsoft/wavlm-base-plus`).
  - `threshold`: Probability threshold (0.0 to 1.0) above which a file is classified as music and deleted (default: `0.5`).
  - `bs`: Batch size for inference.
  - `num_workers`: Number of data loading workers.

Both scripts automatically scale across **all available GPUs** using `torch.multiprocessing`.

---

## Output Structure

Unlike the previous step, this module does not create new audio folders. Instead, it **cleans and annotates** the existing dataset:

1. **Deleted Files**: Any chunk identified as music is physically removed from the `{episode_id}` folders.
2. **Updated Metadata**: The `balalaika.csv` file is safely updated (using temporary partial CSVs to prevent data corruption during multiprocessing).

The final `balalaika.csv` will look like this, now including the `DistillMOS` column:

| filepath | speaker_id | start | end | total_duration | playlist_id | podcast_id | silence_percent | max_silence_duration | DistillMOS |
|----------|------------|-------|-----|----------------|-------------|------------|-----------------|----------------------|------------|
| .../file1.mp3 | 1 | 12.50 | 26.30 | 13.80 | album_1 | ep_1 | 5.2 | 1.1 | 4.12 |
| .../file2.mp3 | 0 | 27.15 | 39.80 | 12.65 | album_1 | ep_1 | 2.0 | 0.5 | 3.85 |
