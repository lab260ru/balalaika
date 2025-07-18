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
- `--podcasts_path`: Root directory containing the text files for phoneme conversion (default: "../../../podcasts").
- `--num_workers`: Number of worker processes per GPU for parallel processing (default: 8).

## Output Structure

For each text file ending with `_giga.txt`, a corresponding `_phonemes.txt` file will be created:

~~~
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_giga.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_accent.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_giga_phonemes.txt
~~~

### File Descriptions
- `.mp3`: Original audio file
- `_giga.txt`: Initial transcription without punctuation
- `_punct.txt`: Text with restored punctuation
- `_accent.txt`: Text with restored accents
- `_giga_phonemes.txt`: Text converted to phonemes
