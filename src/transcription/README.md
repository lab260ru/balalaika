## Transcription (onnx-asr)

ASR via **[onnx-asr](https://github.com/istupakov/onnx-asr)** on **ONNX Runtime**, optionally **TensorRT** — no custom PyTorch dataloaders in this repo.

### Features

- Run multiple models sequentially with **early skip** when `consensus_num` earlier models agree on normalized text.
- **ROVER** → `{stem}_rover.txt` when `use_rover: True`.
- **Word-level timestamps** → `{stem}_{model}.tst` (TSV) when `with_timestamps: True` and the model is in the supported set.
- **Multi-GPU** via multiprocessing.

### Typical `model_names` (Russian)

| Config name | Backend (onnx-asr / HF id) |
|-------------|----------------------------|
| `giga_ctc` | GigaAM v3 CTC |
| `giga_rnnt` | GigaAM v3 RNN-T |
| `vosk` | Vosk Russian |
| `tone` | T-one |

Others: `parakeet_v2`, `parakeet_v3`, `canary`, `whisper_base`, `whisper_turbo`, … — see `MODEL_MAP` in `transcription.py` and comments in `configs/config.yaml`.

## Run

```bash
bash src/transcription/transcription_yaml.sh configs/config.yaml
```

## Config snippet

All keys are documented under **`transcription`** in `configs/config.yaml` (`podcasts_path`, `consensus_num`, `with_timestamps`, `use_tensorrt`, `use_vad`, `use_rover`, `model_names`, `batch_size`, plus optional `model_path`, `vosk_path`, `quantization`, `vad_params`).

## On-disk artifacts

For chunk `{stem}.mp3`:

- `{stem}_{model}.txt` — hypothesis.
- `{stem}_{model}.tst` — timestamps when enabled.
- `{stem}_rover.txt` — ROVER consensus.

## Dependencies

`create_dev_env.sh` typically installs nightly **onnxruntime-gpu** for your CUDA, **`tensorrt-cu13`** (or matching wheel), and **`onnx-asr[gpu,hub]`** — pin versions there.
