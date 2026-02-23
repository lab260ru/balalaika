## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`separation_args.sh`) and then run it:
~~~ 
bash separation/separation_args.sh
~~~  

### Running the Code via Config File  
Example:
~~~ 
bash separation/separation_yaml.sh config_path
~~~  

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing audio files for processing.
- `--one_speaker`: Boolean flag to indicate if only one speaker is expected per audio file.
- `--num_workers`: Number of parallel processes for audio processing.

## Output Structure

After running the script, a `balalaika.csv` file will be created in the specified `podcasts_path` directory:

~~~ 
podcasts/
└── {album_id}/
    └── {episode_id}/
    ....
└── balalaika.csv
~~~

The `balalaika.csv` file contains metadata for each processed audio segment, including speaker diarization flags, NISQA quality metrics, and silence analysis.
