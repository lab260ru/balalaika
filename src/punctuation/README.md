## Usage/Examples

### Running the Code via Command-Line Arguments
You can modify the parameters directly in the shell script (`punctuation/punctuation_args.sh`) and then run it:
```sh
sh punctuation/punctuation_args.sh
```

### Running the Code via Config File
Example:
```sh
bash punctuation/punctuation_yaml.sh config_path
```

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing text files for processing.
- `--model_name`: Name of the punctuation model (e.g., "RUPunct/RUPunct_big").
- `--num_workers`: Number of worker processes for parallel processing.

## Output Structure

For each consensus transcription, a new file with restored punctuation will be created:

```
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_rover.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
```

### File Descriptions
- `.mp3`: Audio segment
- `_rover.txt`: Consensus transcription without punctuation (input file)
- `_punct.txt`: Text with restored punctuation using the RUPunct model

The script processes all `_rover.txt` files and creates corresponding `_punct.txt` files.
