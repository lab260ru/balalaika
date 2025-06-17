## Usage/Examples

### Running the Code via Command-Line Arguments
You can modify the parameters directly in the shell script (`punctuation/punctuation_args.sh`) and then run it:
```sh
sh punctuation/punctuation_args.sh
```

### Running the Code via Config File
Example:
```sh
bash punctuation/punctuation_yaml.sh
```

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing text files for processing (default: "../../../podcasts").
- `--model_name`: Name of the punctuation model (default: "RUPunct/RUPunct_big").
- `--num_workers`: Number of worker processes per GPU for parallel processing (default: 4).

## Output Structure

For each transcribed audio file, a new file with restored punctuation will be created:

```
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_giga.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
```

### File Descriptions
- `.mp3`: Original audio file
- `_giga.txt`: Transcribed text without punctuation (input file)
- `_punct.txt`: Text with restored punctuation using the RUPunct model

The script processes all `_giga.txt` files found in the directory structure and creates corresponding `_punct.txt` files with restored punctuation. Processing is done in parallel using available GPUs for better performance.

## Important Notice
The punctuation and yofication scripts must be executed sequentially!

