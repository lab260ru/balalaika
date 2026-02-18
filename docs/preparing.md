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
7. [ASR Inference (onnx-asr)](#asr-inference-onnx-asr)

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

### Preprocessing & Separation Models

Place the following models under the `models/` directory:

- **Smart VAD Model**: `./models/smart-turn-v3.0.onnx`
- **Music Detection Model**: `./models/music_detection.safetensors`
- **NISQA Config**: `./configs/nisqa_b.yaml`

### ASR Models (onnx-asr)

ASR-модели загружаются **автоматически** из Hugging Face при первом запуске через библиотеку [onnx-asr](https://github.com/istupakov/onnx-asr). Ручная установка не требуется.

Поддерживаемые модели:

| Имя в конфиге    | onnx-asr модель                          | Язык         |
|------------------|------------------------------------------|--------------|
| `giga_ctc`       | `gigaam-v3-ctc`                          | Русский      |
| `giga_rnnt`      | `gigaam-v3-rnnt`                         | Русский      |
| `vosk`           | `alphacep/vosk-model-ru`                 | Русский      |
| `tone`           | `t-tech/t-one`                           | Русский      |
| `parakeet_v3`    | `nemo-parakeet-tdt-0.6b-v3`             | Multilingual |
| `canary`         | `nemo-canary-1b-v2`                      | Multilingual |
| `whisper_turbo`  | `onnx-community/whisper-large-v3-turbo`  | Multilingual |

Инференс выполняется на GPU через ONNX Runtime с поддержкой CUDA и TensorRT execution providers. Батчевая обработка включена по умолчанию для максимальной утилизации GPU.

**Конфигурация GPU-инференса** (`configs/config.yaml`):

```yaml
transcription:
  use_tensorrt: True   # TensorRT EP — fp16, максимальная скорость на NVIDIA GPU
  model_names: ['giga_ctc', 'giga_rnnt', 'vosk', 'tone']

  giga:
    batch_size: 16       # размер батча (подбирается под объем VRAM)
    quantization: int8   # опционально: int8 квантизация
```

При включённом `use_tensorrt: True` используется TensorRT execution provider с fp16 для максимальной производительности. При нескольких GPU файлы автоматически распределяются между картами.

**Note**: Some models (RUPunct, ruAccent) are downloaded automatically from Hugging Face. Make sure your `HF_TOKEN` is set correctly.

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

## ASR Inference (onnx-asr)

Транскрипция реализована через [onnx-asr](https://github.com/istupakov/onnx-asr) — лёгкую Python-библиотеку для ASR на базе ONNX Runtime. Основные преимущества:

- **Единый интерфейс** для всех моделей (GigaAM, Vosk, Parakeet, Whisper, T-one)
- **GPU-инференс** через CUDA / TensorRT execution providers
- **Батчевая обработка** для максимальной утилизации GPU
- **Мульти-GPU** — файлы автоматически распределяются между всеми доступными GPU
- **Автоматическая загрузка** моделей из Hugging Face
- **Минимум зависимостей** — не требуется PyTorch, Transformers или FFmpeg для ASR

### Установка

```bash
pip install onnx-asr[gpu,hub]
```

Для TensorRT (опционально, максимальная скорость):

```bash
pip install onnxruntime-gpu[cuda,cudnn] tensorrt-cu12-libs
```

### Пример использования

```python
import onnx_asr

# Загрузка модели с TensorRT на конкретной GPU
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_max_workspace_size": 6 * 1024**3,
    }),
    ("CUDAExecutionProvider", {"device_id": 0}),
]

model = onnx_asr.load_model("gigaam-v3-rnnt", providers=providers)

# Батчевое распознавание — максимальная утилизация GPU
results = model.recognize(["audio1.wav", "audio2.wav", "audio3.wav"])

# С таймстемпами
model_ts = onnx_asr.load_model("gigaam-v3-ctc", providers=providers).with_timestamps()
result = model_ts.recognize("audio.wav")
```

### Конфигурация

В `configs/config.yaml` секция `transcription`:

```yaml
transcription:
  podcasts_path: /path/to/dataset
  consensus_num: 3        # пропуск файлов при совпадении N моделей
  with_timestamps: True
  use_tensorrt: True      # TensorRT EP (fp16, максимальная скорость)
  use_vad: False          # Silero VAD для длинных аудио (>30с)
  model_names: ['giga_ctc', 'giga_rnnt', 'vosk', 'tone']

  giga:
    batch_size: 16        # подбирается под VRAM (16 для 24GB, 8 для 12GB)

  vosk:
    batch_size: 16

  tone:
    batch_size: 16
```

**Примечание**: дополнительные зависимости для Vosk (k2, kaldifeat, icefall) **больше не требуются** — onnx-asr включает всё необходимое.

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
