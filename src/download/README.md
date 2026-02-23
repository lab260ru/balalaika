## Usage/Examples  

### Running the Code via Command-Line Arguments
You can modify the parameters directly in the shell script (`download_args.sh`) and then run it.
~~~ 
bash download/download_args.sh
~~~  

### Running the Code via Config File
Example:
~~~ 
bash download/download_yaml.sh config_path
~~~  

## Explanation of Parameters
- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Directory to save downloaded podcasts. 
- `--episodes_limit`: Maximum number of episodes to download per podcast.
- `--num_workers`: Number of parallel threads for downloading.

## Output Structure
~~~ 
podcasts/
└── {podcast_id}/
    ├── {episode_id}/
    │   └── episode_audio.mp3
    └──...
~~~ 
