## Transcription (onnx-asr)

ASR transcription is implemented using [onnx-asr](https://github.com/istupakov/onnx-asr) — a library providing a unified interface for multiple ASR models (GigaAM, Vosk, Parakeet, Canary, Whisper, T-one) based on **ONNX Runtime (GPU)** and **TensorRT 10**.

### Benefits
- **No PyTorch/Dataloaders**: All batching and audio preprocessing occur inside `onnx-asr`, reducing overhead.
- **TensorRT 10**: Direct acceleration via TensorRT with automatic fp16 conversion.
- **Multi-GPU**: Tasks are automatically distributed across all available GPUs using `multiprocessing`.
- **Auto-download**: Models are downloaded directly from Hugging Face on the first call.

### Supported Models

| Config Name      | onnx-asr model                           | Language     |
|------------------|------------------------------------------|--------------|
| `giga_ctc`       | `gigaam-v3-ctc`                          | Russian      |
| `giga_rnnt`      | `gigaam-v3-rnnt`                         | Russian      |
| `vosk`           | `alphacep/vosk-model-ru`                 | Russian      |
| `tone`           | `t-tech/t-one`                           | Russian      |
| `parakeet_v3`    | `nemo-parakeet-tdt-0.6b-v3`             | Multilingual |
| `canary`         | `nemo-canary-1b-v2`                      | Multilingual |
| `whisper_turbo`  | `onnx-community/whisper-large-v3-turbo`  | Multilingual |

### Running

```bash
bash src/transcription/transcription_yaml.sh configs/config.yaml
```

### Configuration (`configs/config.yaml`)

```yaml
transcription:
  podcasts_path: /path/to/dataset
  consensus_num: 3        # skip files if N models agree
  with_timestamps: True   # generate word-level timestamps (.tst)
  use_tensorrt: True      # TensorRT EP (fp16, maximum speed)
  use_vad: False          # Silero VAD for processing long audio in chunks
  model_names: ['giga_ctc', 'giga_rnnt', 'vosk', 'tone']

  giga:
    batch_size: 16        # adjust based on VRAM (16 for 24GB)
    # quantization: int8  # optional quantization
```

### Output Structure

For each audio file, the following are created:
- `{filename}_{model_name}.txt` — transcription text.
- `{filename}_{model_name}.tst` — timestamps in TSV format (if `with_timestamps: True`).
- `{filename}_rover.txt` — final transcription obtained by voting (ROVER) of all selected models.

### Installation (Automated in create_dev_env.sh)

The pipeline uses nightly builds of **ONNX Runtime** to support **CUDA 13.1** and **TensorRT 10**.

```bash
# Nightly ORT for CUDA 13
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu

# TensorRT 10 libraries
pip install tensorrt-cu13

# ASR library itself
pip install onnx-asr[gpu,hub]
```
