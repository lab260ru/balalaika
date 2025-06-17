## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`transcription/transcription_args.sh`) and then run it:
~~~ 
sh transcription/transcription_args.sh
~~~  

### Running the Code via Config File  
Example:
~~~ 
sh transcription/transcription_yaml.sh
~~~  

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.  
  *Note: When provided, the config file may include additional settings such as `model_name` and `device`.*
- `--podcasts_path`: Path to the directory containing audio files for transcription (default: "../../../podcasts").
- `--num_workers`: Number of worker processes per GPU for parallel processing (default: 4).
- `--model_name`: Name of the model to use for transcription (default: "rnnt").  

## Output Structure

For each `.mp3` audio file found within the specified `podcasts_path`, a corresponding `_giga.txt` file will be created in the same directory containing the transcription:

```
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        └── {start_time}_{end_time}_{album_id}_{episode_id}_giga.txt
```

The `_giga.txt` file contains the transcribed text from the audio file. The transcription is performed using the specified model (default: "rnnt") and is processed in parallel using multiple GPUs if available.
