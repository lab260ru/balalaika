## Overview
This script is an advanced audio preprocessing pipeline designed to prepare datasets for TTS (Text-to-Speech) or ASR (Automatic Speech Recognition). It processes long podcast audio files by isolating clean, **single-speaker segments** and removing silences or overlapping speech.

### What it does:
1. **Speaker Diarization (`Sortformer`)**: Analyzes the audio in 15-minute chunks to identify who is speaking and when.
2. **Filtering**: Isolates segments where only a **single speaker** is talking (ignoring overlapping voices).
3. **End-of-Speech Detection (`SmartVAD`)**: Refines segment boundaries by cutting the audio at natural pauses.
4. **Metrics Calculation**: Computes the percentage of silence, maximum silence gap, and sets a boolean flag if the segment contains exactly one speaker.
5. **Segmentation**: Slices the original audio into short chunks (e.g., up to 15 seconds) and saves them as `.mp3`.
6. **Metadata Generation**: Saves all chunk metadata (paths, speaker IDs, durations, silence metrics, and single-speaker flags) into a centralized `balalaika.csv` file.
7. **Cleanup**: Automatically **deletes the original large audio file** after successful processing to save disk space.

## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`preprocess_args.sh`) and then run it:
~~~bash
sh preprocess/preprocess_args.sh
~~~  

### Running the Code via Config File  
The python script is executed by passing a YAML configuration file. Example:
~~~bash
sh preprocess/preprocess_yaml.sh config_path
~~~  

## Explanation of Parameters (in YAML config)

- `podcasts_path`: Root directory containing the raw audio files.
- `duration`: Maximum duration in seconds for each final audio segment (default: `15.0`).
- `chunk_duration`: Duration in seconds for processing large files in RAM (default: `900` / 15 minutes).
- `num_workers`: Number of parallel processes **per GPU**. The total number of workers is `num_gpus * num_workers`.
- `sortformer_model`: Path to the ONNX Sortformer model used for diarization.
- `vad_args`: Dictionary containing VAD settings (e.g., `smart_vad_model` path and `smart_vad_threshold`).

## Output Structure

After processing, the original large audio files are deleted and replaced with short, normalized, single-speaker clips. The resulting directory structure will look like this:

~~~text
podcasts/
├── balalaika.csv  <-- Generated metadata file
└── {album_id}/
    ├── {episode_id}/
    │   ├── 12.50_26.30_{album_id}_{episode_id}.mp3
    │   ├── 27.15_39.80_{album_id}_{episode_id}.mp3
    │   └── ... (other segments)
~~~

### Filename Convention
The `{start_time}` and `{end_time}` in the filenames (e.g., `12.50_26.30`) represent the timestamp positions (in seconds) of the segment from the original audio file.

### CSV Metadata (`balalaika.csv`)
The script automatically updates a CSV file with the following columns for each generated audio chunk:
- `filepath`: Absolute path to the generated chunk.
- `speaker_id`: ID of the speaker identified by the diarization model.
- `is_single_speaker`: Boolean flag (`True` or `False`) indicating if only one person speaks in the chunk.
- `start` / `end`: Original timestamps.
- `total_duration`: Length of the chunk in seconds.
- `playlist_id` / `podcast_id`: Source album and episode names.
- `silence_percent`: Percentage of silence within the chunk.
- `max_silence_duration`: Longest continuous silence gap in the chunk.