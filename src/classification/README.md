## Usage/Examples

### Running the Code via Command-Line Arguments
You can modify the parameters directly in the shell script (e.g., `classification/classification_args.sh`) and then run it:
~~~sh
sh classification/classification_args.sh
~~~

### Running the Code via Config File
Example:
~~~sh
bash classification/classification_yaml.sh
~~~

## Explanation of Parameters

- `--config_path`: Path to the YAML configuration file.
- `--podcasts_path`: Path to the podcast folder. This folder must contain the `results.csv` file with metadata about the podcast segments.
- `--threshold`: Similarity threshold for clustering speaker embeddings (default: 0.8). Higher values result in stricter clustering.
- `--model_path`: Embedder model path or identifier (default: `"voxblink2_samresnet100_ft"`). This model is used to generate speaker embeddings.
- `--device`: Device to run the embedder model (e.g., `cuda` or `cpu`).

## Output Structure

After execution, a CSV file named `clustering_result.csv` will be generated in the specified podcasts folder. This file includes the original metadata along with an additional column `speaker` that indicates the assigned speaker cluster ID for each audio segment.


### File Descriptions
- **`results.csv`**: The input CSV file located in the podcasts folder containing metadata for each podcast segment. This file should include a column `IsMono` used to filter segments for clustering.
- **`clustering_result.csv`**: The output CSV file containing the original metadata along with an additional `speaker` column. Each entry in this column represents the speaker cluster ID assigned to that segment.

## Important Notice

- Ensure that the `results.csv` file exists in the specified podcasts folder and contains the required data (e.g., the `IsMono` column) for proper clustering.
- The clustering process is applied only to segments marked as mono (i.e., `IsMono == True`).

