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

* **100-hour dataset (Balalaika100H)**

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

The main configuration file is located at `yapoddataset/configs/config.yaml`. Key parameters:

* `device`: Compute device (e.g., `cpu`, `cuda`) calculations are automatically parallelized to available resources..
* `podcasts_path`: Absolute path where downloaded and processed files are stored.
* `duration`: Maximum length of audio segments in seconds (e.g., `15`).
* `num_workers`: Number of parallel processes per gpu  (in preprocess 1 process - 9 GB VRAM).
* `threshold`: Speaker classification confidence threshold (0.6–0.9, default 0.8).
* `model_path`: (In classification) Path to the pretrained speaker classification model.

Each processing script supports both:

* `*_yaml.sh`: Reads parameters from the YAML config.
* `*_args.sh`: Uses hardcoded arguments in the shell script.

---

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

- [NISQA](https://github.com/gabrielmittag/NISQA)  – Audio quality assessment.
- [GigaAM](https://github.com/salute-developers/GigaAM)  – ASR.
- [ruAccent](https://github.com/Den4ikAI/ruaccent)  – Accent restoration.
- [RUPynct](https://huggingface.co/RUPunct/RUPunct_big)  – Punctuation restoration.
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


