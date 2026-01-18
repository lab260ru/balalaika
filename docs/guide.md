# Usage Guide

This guide explains how to use the YapodDataset pipeline, what files are created at each stage, and how to run individual processing stages.

---

## Table of Contents

1. [Pipeline Stages](#pipeline-stages)
2. [Running the Pipeline](#running-the-pipeline)
3. [Output Files](#output-files)
4. [Configuration](#configuration)
5. [Running Individual Stages](#running-individual-stages)

---

## Pipeline Stages

### 1. Download (`src/download/`)
Downloads podcast episodes from Yandex Music based on provided URLs or playlists.

**Input**: Podcast URLs or playlist IDs  
**Output**: Raw audio files (`.mp3`) organized by `{album_id}/{episode_id}/`

**Configuration**: `config.yaml` → `download` section

---

### 2. Preprocess (`src/preprocess/`)
The preprocessing stage consists of three sequential steps:

#### 2.1. Crest Factor Removal (`crest_factor_remover.py`)
Removes audio files that have excessive crest factor (peak/RMS ratio). Files with crest factor exceeding the threshold are deleted to filter out problematic audio with extreme dynamic range.

**Input**: Raw audio files  
**Output**: Filtered audio files (files with high crest factor are deleted)

**Configuration**: `config.yaml` → `preprocess` section
- `crest_treshold`: Maximum allowed crest factor (peak/RMS). Files exceeding this value are deleted. Default: 10.0

#### 2.2. Loudness Normalization (`preprocess_audio.py`)
Normalizes audio loudness using ITU-R BS.1770-4 standard. All audio files are normalized to a consistent loudness level, overwriting the original files.

**Input**: Filtered audio files  
**Output**: Loudness-normalized audio files (original files overwritten)

**Configuration**: `config.yaml` → `preprocess` section
- `peak`: Peak normalization level in dB. Default: -1.0
- `loudness`: Target loudness level in LUFS. Default: -23.0
- `block_size`: Block size for loudness measurement in seconds. Default: 0.400

#### 2.3. Audio Segmentation (`preprocess.py`)
Splits long audio files into shorter segments (default: 15 seconds) using Voice Activity Detection (VAD). Removes segments that are too short (< 1 second) or too long (> duration limit).

**Input**: Normalized audio files  
**Output**: Segmented audio files named `{start_time}_{end_time}_{album_id}_{episode_id}.mp3`

**Configuration**: `config.yaml` → `preprocess` section
- `duration`: Maximum segment length in seconds
- `vad_args`: VAD thresholds and model path
  - `smart_vad_model`: Path to Smart VAD model
  - `silero_vad_threshold`: Threshold for Silero VAD (0.0-1.0)
  - `smart_vad_threshold`: Threshold for Smart VAD (0.0-1.0)

**Note**: The preprocessing stage runs all three steps sequentially. After preprocessing, the separation stage will create `balalaika.csv` with metadata including single speaker flags and audio quality metrics. Files with detected music will be automatically deleted during the separation stage.

---

### 3. Separation (`src/separation/`)
Performs four types of analysis:
- **Diarization**: Identifies and separates different speakers, creates `.rttm` files
- **NISQA**: Assesses audio quality metrics
- **Music Detection**: Detects music segments in audio
- **Silence Detection**: Analyzes silence patterns in audio

**Input**: Segmented audio files  
**Output**: 
- `.rttm` files (speaker diarization data)
- **`balalaika.csv`**: Metadata file containing:
  - Single speaker flags (indicating whether each audio segment contains only one speaker)
  - Audio quality metrics (from NISQA assessment)
  - **Silence percent**: Percentage of silence in each audio segment
  - **Max silence duration**: Maximum continuous silence duration in seconds
  - File paths and processing status
- Files with detected music are **automatically deleted** during music detection stage
- Can filter out multi-speaker files if `one_speaker: True` is set (files are deleted)

**Configuration**: `config.yaml` → `separation` section
- `diarization`: Speaker diarization settings
  - `num_workers`: Number of workers per GPU
  - `one_speaker`: Filter for single-speaker audio only
- `nisqa`: Audio quality assessment settings
  - `bs`: Batch size
  - `num_workers`: Number of workers
  - `nisqa_config_path`: Path to NISQA config
- `music_detect`: Music detection settings
  - `bs`: Batch size
  - `num_workers`: Number of workers per GPU
  - `music_detect_model`: Path to model
  - `threshold`: Detection threshold
- `silence_detect`: Silence detection settings
  - `num_workers`: Number of workers per GPU

**Note**: The `balalaika.csv` file is created/updated during the separation stage and contains important metadata about each audio segment, including speaker information, quality metrics, and silence analysis. Files detected as containing music are removed from the dataset.

---

### 4. Transcription (`src/transcription/`)
Transcribes audio using multiple ASR models in parallel. Each model creates its own transcription file, then all transcriptions are aggregated using ROVER (Recognizer Output Voting Error Reduction) consensus algorithm.

**Optimization**: If `consensus_num` is set, the pipeline will automatically skip processing remaining models for files where the specified number of models have already produced identical transcriptions. This significantly speeds up processing when models agree.

**Input**: Audio files (`.mp3`)  
**Output**: 
- Individual model transcriptions: `{filename}_{model_name}.txt`
  - `{filename}_giga_ctc.txt`
  - `{filename}_giga_rnnt.txt`
  - `{filename}_giga_ctc_lm.txt`
  - `{filename}_vosk.txt`
  - `{filename}_tone.txt`
- Timestamp files (if enabled): `{filename}_{model_name}.tst`
- **Consensus transcription**: `{filename}_rover.txt` (aggregated from all models)

**Configuration**: `config.yaml` → `transcription` section
- `model_names`: List of models to use
- `consensus_num`: Number of models that need to agree before skipping remaining models (e.g., `3` means if 3 models produce the same transcription, remaining models won't process that file). Set to `0` to process all models for all files.
- `with_timestamps`: Enable timestamp generation (works with `giga_ctc_lm` and `tone`)

**Example**: With `consensus_num: 3` and 5 models, if the first 3 models produce identical transcriptions for a file, the remaining 2 models will skip that file, saving processing time.

---

### 5. Punctuation (`src/punctuation/`)
Restores punctuation marks in transcribed text using RUPunct model.

**Input**: `{filename}_rover.txt`  
**Output**: `{filename}_punct.txt`

**Configuration**: `config.yaml` → `punctuation` section
- `model_name`: RUPunct model name (e.g., `"RUPunct/RUPunct_big"`)

---

### 6. Accents (`src/accents/`)
Restores stress marks (accents) in Russian text using ruAccent model.

**Input**: `{filename}_punct.txt`  
**Output**: `{filename}_accent.txt`

**Configuration**: `config.yaml` → `accent` section
- `model_name`: ruAccent model name (e.g., `"turbo3.1"`)

---

### 7. Phonemizer (`src/phonemizer/`)
Converts text to phonetic representation (phonemes) using TryIPaG2P.

**Input**: `{filename}_rover.txt`  
**Output**: `{filename}_rover_phonemes.txt`

**Configuration**: `config.yaml` → `phonemizer` section

---

### 8. Collate (`src/collate.py`)
Collects all generated metadata files and aggregates them into a single Parquet file for easy access and analysis.

**Input**: All generated text files (`_rover.txt`, `_punct.txt`, `_accent.txt`, `_rover_phonemes.txt`)  
**Output**: `balalaika.parquet` (contains columns: filepath, rover, punct, accent, phonemes)

**Usage**:
```bash
bash src/collate_yamls.sh configs/config.yaml
```

---

## Running the Pipeline

### Complete Pipeline

To run the complete annotation pipeline:

```bash
bash base.sh configs/config.yaml
```

This executes all enabled scripts in sequence:
1. Separation (diarization, quality assessment, music detection)
2. Transcription (multi-model ASR with ROVER consensus)
3. Punctuation restoration
4. Accent restoration
5. Phonemization

### Collecting Metadata

After processing, collect all metadata:

```bash
bash src/collate_yamls.sh configs/config.yaml
```

This creates `balalaika.parquet` in your `podcasts_path` directory.

---

## Output Files

For each audio segment, the pipeline generates:

```
{start_time}_{end_time}_{album_id}_{episode_id}.mp3          # Audio file
{start_time}_{end_time}_{album_id}_{episode_id}.rttm         # Speaker diarization

# Individual model transcriptions (if enabled)
{start_time}_{end_time}_{album_id}_{episode_id}_giga_ctc.txt
{start_time}_{end_time}_{album_id}_{episode_id}_giga_rnnt.txt
{start_time}_{end_time}_{album_id}_{episode_id}_giga_ctc_lm.txt
{start_time}_{end_time}_{album_id}_{episode_id}_vosk.txt
{start_time}_{end_time}_{album_id}_{episode_id}_tone.txt
{start_time}_{end_time}_{album_id}_{episode_id}_giga_ctc_lm.tst  # Timestamps (if enabled)

# Consensus and processed text files
{start_time}_{end_time}_{album_id}_{episode_id}_rover.txt         # Consensus transcription
{start_time}_{end_time}_{album_id}_{episode_id}_punct.txt         # With punctuation
{start_time}_{end_time}_{album_id}_{episode_id}_accent.txt        # With accents
{start_time}_{end_time}_{album_id}_{episode_id}_rover_phonemes.txt # Phonetic representation
```

**Intermediate metadata:**
- `balalaika.csv`: Created during separation stage, contains:
  - Single speaker flags (indicating one-speaker vs multi-speaker segments)
  - Audio quality metrics (NISQA scores)
  - **Silence metrics**: 
    - `silence_percent`: Percentage of silence in each segment
    - `max_silence_duration`: Maximum continuous silence duration (seconds)
  - File paths and processing status
  - Files with music are automatically deleted and not included

**Final aggregated metadata:**
- `balalaika.parquet`: All metadata in structured format (created by collate stage)
- Contains: filepath, rover, punct, accent, phonemes columns

---

## Configuration

The main configuration file is `configs/config.yaml`. Key sections:

### Global Parameters
- `cache_path`: Path for caching temporary files
- `podcasts_path`: **Absolute path** to your data directory

### Stage-Specific Configuration

Each stage has its own configuration section. See `config.yaml` for all available parameters.

**Important**: All paths must be **absolute paths**.

---

## Running Individual Stages

### Modify `base.sh`

Edit the `SCRIPTS` array to run only specific stages:

```bash
SCRIPTS=(
    # "./src/download/download_yaml.sh"
    # "./src/preprocess/preprocess_yaml.sh"
    # "./src/separation/separation_yaml.sh"
    "./src/transcription/transcription_yaml.sh"
    # "./src/punctuation/punctuation_yaml.sh"
    # "./src/accents/accents_yaml.sh"
    # "./src/phonemizer/phonemizer_yaml.sh"
    # "./src/collate_yamls.sh"
)
```

### Run Scripts Directly

```bash
# Activate virtual environment
source .dev_venv/bin/activate

# Run specific stages
bash src/download/download_yaml.sh configs/config.yaml
bash src/preprocess/preprocess_yaml.sh configs/config.yaml
bash src/separation/separation_yaml.sh configs/config.yaml
bash src/transcription/transcription_yaml.sh configs/config.yaml
bash src/punctuation/punctuation_yaml.sh configs/config.yaml
bash src/accents/accents_yaml.sh configs/config.yaml
bash src/phonemizer/phonemizer_yaml.sh configs/config.yaml
bash src/collate_yamls.sh configs/config.yaml
```

### Processing Order

The stages must be run in this order:
1. **Download** → Downloads raw audio files
2. **Preprocess** → Three sequential steps:
   - **Crest Factor Removal** → Removes files with excessive peak/RMS ratio
   - **Loudness Normalization** → Normalizes audio loudness (overwrites files)
   - **Audio Segmentation** → Segments audio into chunks using VAD
3. **Separation** → Diarization, quality assessment, music detection, silence detection
4. **Transcription** → Creates individual model transcriptions + `_rover.txt` (consensus)
5. **Punctuation** → Processes `_rover.txt` → `_punct.txt`
6. **Accents** → Processes `_punct.txt` → `_accent.txt`
7. **Phonemizer** → Processes `_rover.txt` → `_rover_phonemes.txt`
8. **Collate** → Aggregates all metadata into Parquet

**Important Notes:**
- All scripts must be executed from the **project root directory**
- Processing scripts (punctuation, accents, phonemizer) should be run **sequentially** after transcription
- The pipeline processes files in place, so ensure you have backups if needed
- Transcription stage creates individual model files first, then aggregates them into `_rover.txt`

---

## Troubleshooting

### Common Issues

1. **Path errors**: Ensure all paths in `config.yaml` are **absolute paths**
2. **Missing `_rover.txt` files**: Ensure transcription stage completed successfully. ROVER aggregation runs automatically after all model transcriptions finish
3. **File naming**: The pipeline expects specific file naming patterns. Ensure audio files follow the expected structure

For more troubleshooting tips, see individual module READMEs in `src/*/README.md`.
