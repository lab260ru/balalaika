## Overview

Stage 10 performs in-place speech enhancement / denoising on chunked clips.
The current implementation runs a dynamic ONNX export of
**[MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)**,
a 48 kHz speech-enhancement model from **ClearerVoice-Studio**. The original
ClearerVoice API exposes this model as
`ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])`;
Balalaika uses an ONNX Runtime path instead so the same stage can run through
CUDA or TensorRT Execution Provider.

The stage rewrites the existing audio files in place and records progress in
`balalaika.csv` via the `denoised` column. Re-running after a successful pass
is a no-op unless files are missing the `denoised` value.

## Model

- Source family: ClearerVoice-Studio / ClearVoice speech enhancement.
- Model: `MossFormer2_SE_48K` for 48 kHz speech enhancement.
- Runtime artifact: dynamic ONNX, configured by `denoising.onnx_path`.
- Optional download: if `onnx_path` is missing, the stage downloads
  `denoising.hf_repo_id` + `denoising.hf_filename` into the local model
  directory.

## Run

```bash
# As stage 10 of the main runner:
bash base.sh --config_path configs/config.yaml --stage 10 --stop_stage 10

# Or the per-stage wrapper:
bash src/denoising/denoising_yaml.sh configs/config.yaml
```

## Parameters

Documented under **`denoising`** in `configs/config.yaml`:

| Key | Purpose |
|-----|---------|
| `podcasts_path` | Dataset root containing audio files and `balalaika.csv`. |
| `onnx_path` | Local dynamic ONNX model path. |
| `hf_repo_id` | Hugging Face repo used only when `onnx_path` is missing. |
| `hf_filename` | ONNX filename inside the HF repo. |
| `batch_size` | Batch size per GPU worker. |
| `num_workers` | DataLoader workers inside each GPU process. |
| `prefetch_factor` | DataLoader prefetch factor when workers are enabled. |
| `processes` | Worker process count. `0` means one process per visible GPU. |
| `use_tensorrt` | Use TensorRT EP through `src.utils.gpu.get_onnx_providers`. |

Model-specific constants such as 48 kHz sample rate, padding multiple, and TRT
profile shapes live in `src/denoising/denoising.py`; they are not runtime YAML
knobs unless the model itself changes.

## Data Flow

1. `DenoisingDataset` decodes each file with `torchaudio.load_with_torchcodec`,
   mixes to mono, resamples to 48 kHz, and prepares model input batches in
   `src/utils/datasets/denoising.py`.
2. Each GPU worker builds an ONNX Runtime session. With `use_tensorrt: True`,
   provider setup and engine cache location come from the shared `runtime:`
   block via `get_onnx_providers`.
3. The ONNX output is trimmed to the original decoded length.
4. The enhanced waveform is written back to the same path with `torchaudio.save`.
5. The worker streams `denoised=True` rows to `denoising_part_<rank>.csv`; the
   main process periodically merges partials into `balalaika.csv`.

## `balalaika.csv` columns added here

| Column | Description |
|--------|-------------|
| `denoised` | Boolean marker showing the file was enhanced and written back. |

## Resume / Interrupt Safety

The stage uses `src.utils.csv_manager` just like other long-running stages:

- leftover `denoising_part_*.csv` files are absorbed at startup;
- pending files are selected with `unprocessed_paths(..., "denoised", ...)`;
- worker progress is flushed row-by-row to partial CSVs;
- pending files are split into `.balalaika_work/denoising/shard_*.pending`
  files and claimed by workers, avoiding huge multiprocessing pickle payloads;
- `PeriodicCsvMerger` keeps `balalaika.csv` fresh during long runs;
- final merge happens before the stage exits.

## Notes

- Audio is overwritten in place. Keep a source copy if you need the noisy
  version later.
- TensorRT engine build can be slow for the first dynamic profile and is cached
  under `runtime.trt_cache_path`.
- If TensorRT cannot build the model/profile, set `use_tensorrt: False` to run
  the CUDA Execution Provider path.
