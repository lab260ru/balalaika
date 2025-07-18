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
- `--podcasts_path`: Root directory containing the text files for accent restoration (default: "../../../podcasts").
- `--num_workers`: Number of worker processes per GPU for parallel processing (default: 4).
- `--model_name`: Model version to use with RUAccent (default: "turbo3.1").
- `--device`: Device to run the model on (default: "cuda").

## Output Structure

For each text file ending with `_punct.txt`, a corresponding `_accent.txt` file will be created:

~~~
podcasts/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_giga.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_accent.txt
~~~

### File Descriptions
- `.mp3`: Original audio file
- `_giga.txt`: Initial transcription without punctuation
- `_punct.txt`: Text with restored punctuation (input file for accent restoration)
- `_accent.txt`: Final text with restored accents, punctuation, and capitalization

The script processes all `_punct.txt` files found in the directory structure and creates corresponding `_accent.txt` files. Processing is done in parallel using available GPUs for better performance.


