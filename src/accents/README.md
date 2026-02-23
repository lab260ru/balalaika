## Usage/Examples

### Running the Code via Command-Line Arguments
You can modify the parameters directly in the shell script (`accents/accents_args.sh`) and then run it:
~~~sh
sh accents/accents_args.sh
~~~

### Running the Code via Config File
Example:
~~~sh
bash accents/accents_yaml.sh config_path
~~~

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing the text files for accent restoration.
- `--num_workers`: Number of worker processes for parallel processing.
- `--model_name`: Model version to use with ruAccent (e.g., "turbo3.1").
- `--device`: Device to run the model on.

## Output Structure

For each punctuated text file, a corresponding accented file will be created:

~~~
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_rover.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_accent.txt
~~~

### File Descriptions
- `.mp3`: Audio segment
- `_rover.txt`: Initial consensus transcription
- `_punct.txt`: Text with restored punctuation (input file for accent restoration)
- `_accent.txt`: Final text with restored accents, punctuation, and capitalization
