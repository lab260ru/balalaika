# YaPodDataset Pipeline

A full pipeline for podcast processing.

## Prerequisites

```bash
sudo apt update && sudo apt install ffmpeg

cd yapoddataset
bash create_envs.sh
```

### Configuration

The configuration file is located at `yapoddataset/configs/config.yaml`. You can modify all key parameters there.

**Key Parameters:**

- `podcasts_path` – Path where podcasts will be downloaded and processed.
  - Each script has its own `podcasts_path` to prevent data loss in case of a crash.
  - Each script comes with two `.sh` files:
    - `_yaml.sh` – Runs the script using parameters from the config.
    - `_args.sh` – Runs the script with parameters passed directly in the `.sh` file.

- `duration` – In `preprocess`, defines max length of audio segments (e.g., `15` means all segments will be strictly under 15 seconds).

- `num_workers` – In `preprocess`, sets number of parallel processes.  
  (For an RTX 4060 Ti with 6GB VRAM, 1 process per 6GB is recommended.)

- `threshold` – In speaker classification, recommended value is `0.8` (optimal value, found experimentally).
  - Range: from `0.6` to `0.9`.
  - Lower values result in more speakers detected; higher values in fewer.

- `model_path` – Path to the pre-trained VoxBlink ResNet model for speaker classification.
  - [Available models list](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) 
  - Do not use ONNX versions.
  - You can use your own embedder by modifying `yapoddataset/classification/emb/embeder.py`.

## Models Location

All required models should be placed in the following directory:

```
models/
```

Expected structure:

```
models/
├── voxblink.../     # Speaker classification model
│   └── ...
└── nisqa_s.tar      # Audio quality assessment model
```

Make sure these directories and files are present before running any processing scripts.
  
## Environment Variables

Create a `.env` file in the project root:

```ini
HF_TOKEN=your_huggingface_token
YANDEX_KEY=your_yandex_speechkit_key
```

- `YANDEX_KEY` – Required for downloading datasets.
- `HF_TOKEN` – Used for speaker count estimation.

## Running the Pipeline

### Basic Scenario (BASE)

```bash
bash base.sh
```

This scenario:
- Downloads datasets
- Splits audio into semantic chunks
- Performs transcription of all segments
- Segments by speaker
- Applies phonemization

All metadata is saved in `result.csv` inside the podcasts folder.

## Download Dataset Using Existing Metadata

Before running, specify `parquet_path` and `podcasts_path`, as well as the desired podcasts in `configs/config.yaml` like this:

```yaml 
download:
  podcasts_path: /home/nikita/podcasts_1
  episodes_limit: 4
  num_workers: 10
  parquet_path: /home/nikita/podcasts_4/balalaika.parquet
  podcasts_urls:  
    - 'https://music.yandex.ru/album/21851634'      
    - 'https://music.yandex.ru/album/32863444'      
    - 'https://music.yandex.ru/album/30859840'      
    - 'https://music.yandex.ru/album/18332051'      
```

Then run:

```bash
bash use_meta.sh
```

## Important Notes

- All scripts must be executed from the **project root directory**.
- Paths in the config file must be **absolute**.
- The processing scripts (`punctuation`, `accents`, `yofication`) should be run **sequentially**.
- You’ll need:
  - Yandex Music API key ([How to get one](https://yandex-music.readthedocs.io/en/main/token.html)) 
  - Hugging Face token

## Models Used

- [NISQA](https://github.com/gabrielmittag/NISQA)  – Audio quality assessment.
- [GigaAM](https://github.com/salute-developers/GigaAM)  – Acoustic model.
- [ruAccent](https://github.com/Den4ikAI/ruaccent)  – Accent restoration.
- [RUPynct](https://huggingface.co/RUPunct/RUPunct_big)  – Punctuation restoration.
- [VoxBlink ResNet](https://github.com/wenet-e2e/wespeaker)  – Speaker classification.
- [TryIPaG2P](https://github.com/NikiPshg/TryIPaG2P)  – Phonemization.
- [Speaker Diarization](https://github.com/pyannote/pyannote-audio)  – Speaker diarization.
- [Whisper](https://github.com/SYSTRAN/faster-whisper)  – ASR + segmentation.