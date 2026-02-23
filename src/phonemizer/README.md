## Usage/Examples

### Running the Code via Command-Line Arguments
You can modify the parameters directly in the shell script (`phonemizer/phonemizer_args.sh`) and then run it:
~~~sh
sh phonemizer/phonemizer_args.sh
~~~

### Running the Code via Config File
Example:
~~~sh
bash phonemizer/phonemizer_yaml.sh config_path
~~~

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing the text files for phoneme conversion.
- `--num_workers`: Number of worker processes for parallel processing.

## Output Structure

For each consensus transcription, a corresponding phoneme file will be created:

~~~
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_rover.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_accent.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_rover_phonemes.txt
~~~

### File Descriptions
- `.mp3`: Audio segment
- `_rover.txt`: Consensus transcription
- `_punct.txt`: Text with restored punctuation
- `_accent.txt`: Text with restored accents
- `_rover_phonemes.txt`: Text converted to phonemes
