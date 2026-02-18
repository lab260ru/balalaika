## Transcription (onnx-asr)

ASR-транскрипция реализована через [onnx-asr](https://github.com/istupakov/onnx-asr) — единый интерфейс для множества ASR-моделей на базе ONNX Runtime.

### Поддерживаемые модели

| Имя в конфиге    | onnx-asr модель                          | Язык         |
|------------------|------------------------------------------|--------------|
| `giga_ctc`       | `gigaam-v3-ctc`                          | Русский      |
| `giga_rnnt`      | `gigaam-v3-rnnt`                         | Русский      |
| `vosk`           | `alphacep/vosk-model-ru`                 | Русский      |
| `vosk_small`     | `alphacep/vosk-model-small-ru`           | Русский      |
| `tone`           | `t-tech/t-one`                           | Русский      |
| `parakeet_v2`    | `nemo-parakeet-tdt-0.6b-v2`             | English      |
| `parakeet_v3`    | `nemo-parakeet-tdt-0.6b-v3`             | Multilingual |
| `canary`         | `nemo-canary-1b-v2`                      | Multilingual |
| `whisper_base`   | `whisper-base`                           | Multilingual |
| `whisper_turbo`  | `onnx-community/whisper-large-v3-turbo`  | Multilingual |

Модели загружаются автоматически из Hugging Face при первом запуске.

### Запуск

```bash
sh src/transcription/transcription_yaml.sh configs/config.yaml
```

### Конфигурация (`configs/config.yaml`)

```yaml
transcription:
  podcasts_path: /path/to/dataset
  consensus_num: 3        # пропуск файлов при совпадении N моделей
  with_timestamps: True
  use_tensorrt: False     # TensorRT EP (fp16, максимальная скорость)
  use_vad: False          # Silero VAD для длинных аудио
  model_names: ['giga_ctc', 'giga_rnnt', 'vosk', 'tone']

  giga:
    batch_size: 16        # размер батча (подбирается под VRAM)
    # quantization: int8  # опциональная квантизация

  vosk:
    batch_size: 16
    # vosk_path: ./models/vosk-model-ru  # локальная модель (иначе скачивается из HF)

  tone:
    batch_size: 16
```

### GPU-инференс

- **CUDA**: используется по умолчанию (`CUDAExecutionProvider`)
- **TensorRT**: `use_tensorrt: True` — fp16, максимальная скорость на NVIDIA GPU
- **Мульти-GPU**: файлы автоматически распределяются между всеми доступными GPU
- **Батчи**: `batch_size` контролирует количество файлов в одном батче

### Выходная структура

Для каждого аудиофайла создаётся файл с транскрипцией:

```
dataset/
└── {album_id}/
    └── {episode_id}/
        ├── audio.wav
        ├── audio_giga_ctc.txt      # транскрипция GigaAM CTC
        ├── audio_giga_ctc.tst      # таймстемпы (если with_timestamps: True)
        ├── audio_giga_rnnt.txt     # транскрипция GigaAM RNN-T
        ├── audio_vosk.txt          # транскрипция Vosk
        └── audio_tone.txt          # транскрипция T-one
```

### Зависимости

```bash
pip install onnx-asr[gpu,hub] soundfile
```

Для TensorRT (опционально):

```bash
pip install onnxruntime-gpu[cuda,cudnn] tensorrt-cu12-libs
```
