## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`predprocess_args.sh`) and then run it:
~~~ 
sh predprocess/predprocess_args.sh
~~~  

### Running the Code via Config File  
Example:
~~~ 
sh predprocess/predprocess_yaml.sh config_path
~~~  

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file (default: None).
- `--podcasts_path`: Root directory containing podcast audio files (default: '../podcasts').
- `--whisper_model`: Name of the Whisper model to use (default: 'large-v3').
- `--compute_type`: Compute type for the model (default: 'float16').
- `--beam_size`: Beam size for beam search decoding (default: 5).
- `--duration`: Target duration in seconds for each audio segment (default: 15).
- `--device`: Hardware accelerator for the Whisper model (default: 'cpu').
- `--num_workers`: Number of parallel processes for audio processing (default: 1).

## Output Structure

After processing, the original audio file is segmented into shorter clips. The resulting structure will be:

~~~ 
podcasts/
└── {album_id}/
    ├── {episode_id}/
    │   ├── start_time_end_time_{album_id}_{episode_id}.mp3
    │   ├── start_time_end_time_{album_id}_{episode_id}_whisper.txt
    │   └── ... (other segments)
~~~

Each podcast episode (originally an `.mp3` file) is moved to its own folder named after the episode ID (within its album folder) and then segmented. For each segment, two files are created:
- An audio file (`{start_time}_{end_time}_{album_id}_{episode_id}.mp3`) containing the audio segment
- A text file (`{start_time}_{end_time}_{album_id}_{episode_id}_whisper.txt`) containing the transcription for that segment

The `start_time` and `end_time` in the filenames represent the timestamp positions (in seconds) of the segment in the original audio file.

If the segmentation is successful, the original file is deleted.
