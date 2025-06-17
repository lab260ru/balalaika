## Usage/Examples

### Running the Code via Command-Line Arguments  
You can modify the parameters directly in the shell script (`separation_args.sh`) and then run it:
~~~ 
bash separation/separation_args.sh
~~~  

### Running the Code via Config File  
Example:
~~~ 
bash separation/separation_yaml.sh
~~~  

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Root directory containing podcast audio files for processing.
- `--one_speaker`: Boolean flag to indicate if only one speaker is expected per audio file (default: True).
- `--num_workers`: Number of parallel processes for audio processing (default: 4).

## Output Structure

After running the script, a results.csv file will be created in the specified `podcasts_path` directory:

~~~ 
podcasts/
└── results.csv
~~~

The `results.csv` file contains the following information for each processed audio file:
- `audio_path`: Path to the audio file relative to the podcasts directory
- `is_mono`: Boolean indicating if the file contains a single speaker
- `NOI`: Noise metric score
- `COL`: Coloration metric score
- `DISC`: Discontinuity metric score
- `LOUD`: Loudness metric score
- `MOS`: Mean Opinion Score
- `playlist_id`: ID of the playlist
- `podcast_id`: ID of the podcast
- `start`: Start time of the segment
- `end`: End time of the segment
