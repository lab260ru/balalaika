# Preparing Your Dataset

This guide explains how to set up the pipeline and prepare your own audio datasets for processing. **The pipeline can process not only podcasts from Yandex Music, but also your own audio datasets** by organizing them in the expected format.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Models Setup](#models-setup)
5. [Configuration](#configuration)
6. [Preparing Your Own Dataset](#preparing-your-own-dataset)
7. [Additional Requirements for Vosk](#additional-requirements-for-vosk)

---

## Prerequisites

Ensure you have the following tools installed:

```bash
sudo apt update && sudo apt install -y ffmpeg
wget -qO- https://astral.sh/uv/install.sh | sh
```

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/mtuciru/balalaika
cd balalaika
```

Choose the appropriate environment:

- **For development/annotation/modification**: 
  ```bash
  bash create_dev_env.sh
  ```

- **For using pre-annotated dataset only**:
  ```bash
  bash create_user_env.sh
  ```

---

## Environment Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```ini
HF_TOKEN=<your_huggingface_token>
YANDEX_KEY=<your_yandex_music_token>
```

- `HF_TOKEN`: Required for speaker diarization and accessing Hugging Face models
- `YANDEX_KEY`: Required only if downloading podcasts from Yandex Music (optional if using your own data)

**How to get tokens:**
- **Hugging Face Token**: Create an account at [huggingface.co](https://huggingface.co) and generate a token in your account settings
- **Yandex Music Token**: Follow the guide at [yandex-music.readthedocs.io](https://yandex-music.readthedocs.io/en/main/token.html)

---

## Models Setup

Place all required models under the `models/` directory:

- **Vosk Model**: `./models/vosk-model-ru` (directory with Vosk model files)
- **Smart VAD Model**: `./models/smart-turn-v3.0.onnx`
- **Music Detection Model**: `./models/music_detection.safetensors`
- **KenLM Language Model**: `./models/kenlm.bin` (for Giga CTC model)
- **NISQA Config**: `./configs/nisqa_b.yaml`

**Note**: Some models are downloaded automatically from Hugging Face (RUPunct, ruAccent). Make sure your `HF_TOKEN` is set correctly.

---

## Configuration

Edit `configs/config.yaml` and set the `podcasts_path` to an **absolute path** where your data is located:

```yaml
preprocess:
  podcasts_path: /absolute/path/to/your/dataset
  crest_treshold: 10  # Maximum allowed crest factor (peak/RMS). Files exceeding this are deleted
  peak: -1.0  # Peak normalization level in dB (for loudness normalization)
  loudness: -23.0  # Target loudness level in LUFS (for loudness normalization)
  block_size: 0.400  # Block size for loudness measurement in seconds
  duration: 15  # Maximum segment length in seconds
  num_workers: 4  # Number of parallel workers
```

**Important**: 
- All paths in the config file must be **absolute paths**
- The `podcasts_path` should point to the root directory containing your audio files
- Preprocessing runs three sequential steps: crest factor removal → loudness normalization → segmentation

---

## Preparing Your Own Dataset

**The pipeline is not limited to podcasts!** You can process any Russian audio dataset by organizing it in the expected format.

### Supported Input Formats

The pipeline accepts audio files in one of the following structures:

#### Format 1: Pre-segmented Audio Files

If your audio is already segmented into short clips:

```
your_dataset/
└── {album_id}/          # Can be any identifier (e.g., speaker_id, session_id)
    └── {episode_id}/     # Can be any identifier (e.g., recording_id, clip_id)
        ├── audio_1.mp3
        ├── audio_2.wav
        ├── audio_3.opus
        ...
```

**Supported audio formats**: `.mp3`, `.wav`, `.opus`, `.flac`, `.m4a`

#### Format 2: Long Audio Files

If you have long audio files that need to be segmented:

```
your_dataset/
└── {album_id}/
    └── {episode_id}/
        └── long_audio.mp3    # Will be automatically segmented
        ...
```

The preprocessing stage will automatically perform three sequential steps:
1. **Crest Factor Removal**: Remove files with excessive peak/RMS ratio (crest factor > `crest_treshold`)
2. **Loudness Normalization**: Normalize all audio to consistent loudness (ITU-R BS.1770-4 standard)
3. **Audio Segmentation**: 
   - Segment files into chunks (default: 15 seconds)
   - Remove segments that are too short (< 1 second)
   - Use Voice Activity Detection to find speech segments

#### Format 3: Already Processed Data

If you already have processed data with the expected naming:

```
your_dataset/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ...
```

### Dataset Organization Tips

1. **Use meaningful identifiers**: Replace `{album_id}` and `{episode_id}` with identifiers that make sense for your dataset:
   - Speaker IDs, session IDs, recording dates, etc.
   - Example: `speaker_001/session_20240101/`

2. **Keep directory structure shallow**: Avoid deeply nested directories for better performance

3. **Audio quality**: The pipeline works best with:
   - Clear speech audio
   - Sample rate: 16kHz or higher (will be resampled automatically)
   - Mono or stereo (will be converted to mono automatically)
   - Reasonable dynamic range (excessive crest factor files will be filtered out)

4. **Preprocessing stages**: The preprocessing stage includes:
   - **Crest factor filtering**: Files with peak/RMS ratio > `crest_treshold` are removed
   - **Loudness normalization**: All audio is normalized to consistent loudness (ITU-R BS.1770-4)
   - **Segmentation**: Long files are split into shorter segments using VAD

### Processing Your Dataset

Once your dataset is organized, you can skip the download stage and start from preprocessing:

1. **Set `podcasts_path`** in `config.yaml` to point to your dataset

2. **Modify `base.sh`** to skip download:
   ```bash
   SCRIPTS=(
       # "./src/download/download_yaml.sh"  # Skip if using your own data
       "./src/preprocess/preprocess_yaml.sh"
       "./src/separation/separation_yaml.sh"
       "./src/transcription/transcription_yaml.sh"
       "./src/punctuation/punctuation_yaml.sh"
       "./src/accents/accents_yaml.sh"
       "./src/phonemizer/phonemizer_yaml.sh"
       "./src/collate_yamls.sh"
   )
   ```

3. **Run the pipeline**:
   ```bash
   bash base.sh configs/config.yaml
   ```

**Important**: After the preprocessing stage (`./src/preprocess/preprocess_yaml.sh`), the separation stage creates a `balalaika.csv` file in your dataset directory. This file contains:
- **Single speaker flags**: Indicates whether each audio segment contains only one speaker
- **Audio quality metrics**: NISQA quality assessment scores for each segment
- **Silence metrics**: 
  - `silence_percent`: Percentage of silence in each audio segment
  - `max_silence_duration`: Maximum continuous silence duration in seconds
- **File paths**: References to all processed audio files

Additionally, files detected as containing music are **automatically deleted** during the music detection step in the separation stage.

### Example: Processing a Custom Dataset

Let's say you have a dataset of Russian speech recordings organized like this:

```
/my_data/
└── speaker_001/
    ├── recording_001.wav
    ├── recording_002.wav
    └── recording_003.wav
└── speaker_002/
    ├── recording_001.wav
    └── recording_002.wav
```

1. Set in `config.yaml`:
   ```yaml
   podcasts_path: /my_data
   ```

2. The pipeline will process all audio files and create:
   - Transcriptions
   - Punctuation
   - Accents
   - Phonemes
   - Metadata in `balalaika.parquet`

**Note**: The `{album_id}` and `{episode_id}` in file names are just organizational identifiers. You can use any naming scheme that fits your data structure.

---

## Additional Requirements for Vosk

If you plan to use Vosk models for transcription, install additional dependencies:

```bash
python -m pip install git+https://github.com/lhotse-speech/lhotse
python -m pip install https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
python -m pip install https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cuda/1.25.5.dev20241029/linux/kaldifeat-1.25.5.dev20250807+cuda12.8.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
python3 -m pip install git+https://github.com/k2-fsa/icefall
```

**Note**: These dependencies are only needed if you enable Vosk in the transcription configuration.

---

## Next Steps

After setting up your environment and preparing your dataset:

1. Review the [Usage Guide](guide.md) to understand how to run the pipeline
2. Check `configs/config.yaml` for all available configuration options
3. Start with a small subset of your data to test the pipeline
4. Monitor the output files to ensure everything is working correctly

---

## Troubleshooting

### Common Setup Issues

1. **Model not found**: Verify all models are placed in the `models/` directory with correct paths in config
2. **Path errors**: Ensure all paths in `config.yaml` are **absolute paths**
3. **Permission errors**: Ensure the virtual environment is activated before running scripts
4. **CUDA errors**: Check GPU availability and CUDA installation if using GPU-accelerated models

For more help, see individual module READMEs in `src/*/README.md`.
