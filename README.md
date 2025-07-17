# Balalaika Pipeline

A complete production-ready pipeline for processing podcast audio data, from download to feature extraction.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
   - [Quick Setup (Default Parameters)](#quick-setup)
   - [Custom Metadata Download](#custom-metadata-download)
4. [Running the Pipeline](#running-the-pipeline)
   - [Basic Scenario (Local Processing)](#basic-scenario-local-processing)
5. [Configuration](#configuration)
6. [Environment Variables](#environment-variables)
7. [Models](#models)
8. [Citation](#citation)
9. [Acknowledgments](#acknowledgments)

---

## Prerequisites

Ensure you have the following tools installed on your system:

```bash
sudo apt update && sudo apt install -y ffmpeg
wget -qO- https://astral.sh/uv/install.sh | sh

````

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/mtuciru/balalaika
cd balalaika
# Use this if you want to annotate/modify the dataset
bash create_dev_env.sh
# Use this if you only want to use the pre-annotated dataset
bash create_user_env.sh 
```

---

## Data Preparation

### Quick Setup (Default Parameters)

To download and prepare the dataset with default settings, choose one of the preconfigured dataset sizes:

* **100-hour dataset**
  ```bash
  bash use_meta_100h.sh
  ```

* **500-hour dataset**
  ```bash
  bash use_meta_500h.sh
  ```

* **1000-hour dataset**
  ```bash
  bash use_meta_1000h.sh
  ```

* **2000-hour dataset**
  ```bash
  bash use_meta_2000h.sh
  ```

All metadata can also be downloaded from [Hugging Face – MTUCI](https://huggingface.co/MTUCI).

### Custom Metadata Download

If you already have generated metadata files (`balalaika.parquet` and `balalaika.pkl`), place them in the project root and run:

```bash
bash use_meta.sh
```

---

## Running the Pipeline


### Basic Scenario (Local Processing)


This scenario will:

1. Download datasets
2. Split audio into semantic chunks
3. Transcribe all segments
4. Perform speaker segmentation
5. Apply phonemization

To execute locally, run:

```bash
bash base.sh configs/config.yaml
```

All output metadata will be saved in `podcasts/result.csv`.

---

## Configuration

The main configuration file is located at `yapoddataset/configs/config.yaml`. This file is organized into several sections, each corresponding to a specific stage of the podcast processing pipeline. Below is a detailed explanation of the key parameters within each section.

---

### Global Parameters

* `podcasts_path`:  It specifies the **absolute path** to the directory where all downloaded podcast files will be stored and where subsequent processing (preprocessing, separation, transcription, etc.) will look for and save its output.
---

### `download` Section

This section controls how podcast episodes are downloaded.

* `podcasts_path`: (As explained above) The directory where downloaded podcasts will be saved.
* `episodes_limit`: This sets a **limit on the number of episodes** to download from a single podcast playlist.
* `num_workers`: Specifies the **number of parallel processes** to use for downloading. A higher number can speed up downloads but will consume more system resources.
* `podcasts_urls_file`: This parameter points to the **path of a `.pkl` file** that contains a list of podcast URLs to be downloaded.

---

### `preprocess` Section

This section handles the initial processing of downloaded audio files, such as chopping them into smaller segments.

* `podcasts_path`: (As explained above) The directory containing the raw downloaded podcasts that need to be preprocessed.
* `duration`: Defines the **maximum length in seconds** for each audio sample (segment).
* `num_workers`: Specifies the **number of parallel processes** to use during preprocessing.
* `whisper_model`: Specifies the **name or path of the Faster-Whisper compatible model** to be used for initial audio processing.
* `compute_type`: Determines the **computation type** for the Whisper model, affecting performance and memory usage.
* `beam_size`: This parameter is related to the **beam search algorithm** used in the Whisper model's decoding process.

---

### `separation` Section

This section calculates metrics for each audio

* `podcasts_path`: (As explained above) The directory where the chopped podcasts (from the `preprocess` stage) are located.
* `num_workers`: The **number of parallel processes** to use for audio separation.
* `nisqa_config`: Specifies the **path to the configuration file for NISQA** 
* `one_speaker`: A **boolean flag** (`True`/`False`) that, when enabled (`True`), instructs the system to download and process only those audio recordings that should contain a single speaker.

---

### `transcription` Section

This section is responsible for converting audio into text.

* `podcasts_path`: (As explained above) The directory containing the processed audio files ready for transcription.
* `model_name`: Specifies the **type of automatic speech recognition (ASR) model** to use. Options typically include `"ctc" or "rnnt"`.
* `num_workers`: The **number of parallel processes per GPU** to use for transcription.
* `with_timestamps`: A **boolean flag** (`True`/`False`) that, when enabled, allows the transcription process to generate timestamps for each word or segment. **it only works with ctc**
* `lm_path`: Specifies the **path to a language model file (`.bin`)**. A language model can improve transcription accuracy by providing contextual information. 

---

### `punctuation` Section

This section focuses on adding proper punctuation to the transcribed text.

* `podcasts_path`: (As explained above) The directory where the transcribed text files are located.
* `model_name`: Specifies the **name of the RUPunct model** to be used for punctuation restoration. 
* `num_workers`: The **number of parallel processes per GPU** to use for punctuation.
---

### `accent` Section

In the transcribed text this part is restored with accents.

* `podcasts_path`: (As explained above) The directory containing the relevant podcast files.
* `num_workers`: The **number of parallel processes per GPU** to use for accent processing.
* `model_name`: Specifies the **name of the ruAccent model** to be used.

---

### `phonemizer` Section

This section is responsible for converting text into phonetic representations (phonemes).

* `podcasts_path`: (As explained above) The directory where the text files (from transcription and punctuation stages) are located.
* `num_workers`: The **number of parallel processes per GPU** to use for phonemization.
---

### `classification` Section

This section relates to global speaker clustering.

* `podcasts_path`: (As explained above) The directory containing the podcast files relevant for classification.
* `num_workers`: The **number of parallel processes per GPU** to use for classification.
* `threshold`: This is the **speaker classification confidence threshold**. Values typically range from `0.6` to `0.9`. A higher threshold means the model needs to be more confident in its classification to assign a label. 
* `model_path`: Specifies the **path to the pretrained speaker classification model** in `.pt` format.
---

### Execution Scripts

Each processing script (`*_yaml.sh` and `*_args.sh`) offers flexibility in how parameters are provided:

* `*_yaml.sh`: These scripts read all necessary parameters directly from the main `config.yaml` file, ensuring consistency across different stages.
* `*_args.sh`: These scripts allow for hardcoded arguments directly within the shell script itself, which can be useful for quick tests or specific overrides without modifying the main configuration file.

## Environment Variables

Create a `.env` file in the project root with the following:

```ini
HF_TOKEN=<your_huggingface_token>
YANDEX_KEY=<your_yandex_music_token>
```

* `HF_TOKEN`: Required for speaker count estimation.
* `YANDEX_KEY`: Required for dataset downloads.

---

## Important Notes

- All scripts must be executed from the **project root directory**.
- Paths in the config file must be **absolute**.
- The processing scripts (punctuation, accents, yofication) should be run **sequentially**.
- You’ll need:
  - Yandex Music API key ([How to get one](https://yandex-music.readthedocs.io/en/main/token.html)) 
  - Hugging Face token

## Models

Place all required models under the `models/` directory with the following structure:

```
models/
├── vosblink_resnet/        # Speaker classification model
│   └── ...
└── nisqa_s.tar             # Audio quality assessment model
```

Supported models:

- [NISQA](https://github.com/deepvk/NISQA-s)  – Audio quality assessment.
- [GigaAM](https://github.com/salute-developers/GigaAM)  – ASR.
- [ruAccent](https://github.com/Den4ikAI/ruaccent)  – Accent restoration.
- [RUPunct](https://huggingface.co/RUPunct/RUPunct_big)  – Punctuation restoration.
- [VoxBlink ResNet](https://github.com/wenet-e2e/wespeaker)  – Speaker classification.
- [TryIPaG2P](https://github.com/NikiPshg/TryIPaG2P)  – Phonemization.
- [Speaker Diarization](https://github.com/pyannote/pyannote-audio)  – Speaker diarization.
- [Whisper](https://github.com/SYSTRAN/faster-whisper)  – ASR + segmentation

---

## Citation

If you use this pipeline in your research or production, please cite:
```
```

---

## References and Acknowledgements

Thanks to all the developers and contributors who made this project possible.

<a href="https://github.com/mtuciru/balalaika/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yeongpin/balalaikap&preview=true&max=&columns=" />
</a>


