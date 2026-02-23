## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`preprocess_args.sh`) and then run it:
~~~ 
sh preprocess/preprocess_args.sh
~~~  

### Running the Code via Config File  
Example:
~~~ 
sh preprocess/preprocess_yaml.sh config_path
~~~  

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing audio files.
- `--duration`: Maximum duration in seconds for each audio segment (default: 15).
- `--num_workers`: Number of parallel processes for audio processing.

## Output Structure

After processing, the original audio files are normalized and segmented into shorter clips. The resulting structure will be:

~~~ 
podcasts/
└── {album_id}/
    ├── {episode_id}/
    │   ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
    │   └── ... (other segments)
~~~

Each audio file is moved to its own folder named after the episode ID (within its album folder) and then segmented.

The `{start_time}` and `{end_time}` in the filenames represent the timestamp positions (in seconds) of the segment in the original audio file.
