## Overview

Quality filtering and quality annotation on chunked clips. Speaker diarization
is handled in **preprocess** (Sortformer), not here. The separation package
currently contains four model/filter stages:

1. **Music detection** — WavLM backbone + fine-tuned head at
   `separation.music_detect.music_detect_model`. For every processed clip the
   model's music probability is written to `balalaika.csv` as `music_prob`.
   Clips above the threshold are deleted from disk, their rows removed from the
   CSV, and the kept/dropped totals are appended to `filter_summary.csv`.
2. **DistillMOS scoring** — speech quality score written to `balalaika.csv` as
   `DistillMOS`. Annotation only; no deletion.
3. **DistillMOS filter** — deletes clips below
   `separation.distillmos_filter.threshold` after scoring and records an audit
   row in `filter_summary.csv`.
4. **Anti-spoofing** — [Spectra-0](https://huggingface.co/lab260/spectra_0) ONNX classifier.
   It estimates generated/spoofed speech probability,
   writes `antispoof_score` and `antispoof_generated_prob`, deletes clips above
   `separation.antispoofing.threshold`, and records an `antispoofing` audit row.

Every script writes a rotating, timestamped log file under `BALALAIKA_LOG_DIR`
(default `./logs`).

## Models

| Stage | Model | Runtime | Notes |
|-------|-------|---------|-------|
| Music detection | WavLM + fine-tuned head | PyTorch | Local safetensors head configured by `music_detect_model`. |
| DistillMOS | DistillMOS | PyTorch | Adds MOS estimate only. |
| Anti-spoofing | [Spectra-0](https://huggingface.co/lab260/spectra_0) | ONNX Runtime | Binary bonafide/spoof classifier on raw waveforms; downloads `model.onnx` if missing. |

## Run

```bash
# Stages 4..5.6 of the main runner:
bash base.sh --config_path configs/config.yaml --stage 4 --stop_stage 5.6

# Or the legacy wrapper for music + DistillMOS scoring only:
bash src/separation/separation_yaml.sh configs/config.yaml

# Individual stages:
python -m src.separation.music_detect       --config_path configs/config.yaml
python -m src.separation.distillmos_process --config_path configs/config.yaml
python -m src.separation.distillmos_filter  --config_path configs/config.yaml
python -m src.separation.antispoofing       --config_path configs/config.yaml
```

## Parameters

Documented under **`separation`** in `configs/config.yaml`:

| Config key | Purpose |
|------------|---------|
| `podcasts_path` | Dataset root containing audio files and `balalaika.csv`. |
| `music_detect.*` | WavLM classifier batch/DataLoader settings, model path, threshold, optional cache/base model. |
| `distillmos.*` | DistillMOS batch/DataLoader settings. |
| `distillmos_filter.*` | Threshold and deletion worker count. |
| `antispoofing.onnx_path` | Local Spectra-0 ONNX path. Missing file is downloaded from HF. |
| `antispoofing.batch_size`, `num_workers`, `prefetch_factor` | Anti-spoofing batching and DataLoader settings. |
| `antispoofing.threshold` | Generated/spoofed probability above which a clip is deleted. |
| `antispoofing.use_tensorrt` | Enable TensorRT EP when the ONNX/profile supports it. |

## `balalaika.csv` columns added here

| Column | Description |
|--------|-------------|
| `music_prob` | Music classifier probability (0-1). Row removed if file deleted. |
| `DistillMOS` | Predicted MOS score. |
| `antispoof_score` | Anti-spoofing score written by the pipeline. |
| `antispoof_generated_prob` | Deletion score compared with `antispoofing.threshold`. |

## Filter summary rows emitted here

| `stage` | Notes |
|---------|-------|
| `music_detect` | Files / hours kept vs. dropped at `music_detect.threshold`. |
| `distillmos_filter` | Files / hours kept vs. dropped at `distillmos_filter.threshold`. |
| `antispoofing` | Files / hours kept vs. dropped at `antispoofing.threshold`. |

## Resume / Interrupt Safety

All long-running sub-stages use `src.utils.csv_manager`:

- **Atomic writes** of `balalaika.csv` through tmp file + rename.
- **Auto-bootstrap** from the audio tree if the CSV is missing.
- **Incremental partial CSVs** such as `music_part_<rank>.csv`,
  `distillmos_part_<rank>.csv`, and `antispoof_part_<rank>.csv`.
- **Disk-backed work shards** under `.balalaika_work/<stage>/`, so workers
  claim bounded shard files instead of unpickling millions of paths at start.
- **Resume on next run** by absorbing leftovers and scheduling only files still
  missing the relevant column.
- **Deletion-aware merges** prune rows whose files were removed by filtering
  stages.

Re-running a completed stage is a no-op unless new files are added or a result
column is missing.

## Result

- Music-heavy chunks removed.
- Low-DistillMOS chunks removed when the filter threshold is enabled.
- Generated/spoofed speech removed by Spectra-0 anti-spoofing.
- `balalaika.csv` and `filter_summary.csv` updated for downstream report and
  export stages.

For merged fields in exported WebDataset `json`, see
[`example/README.md`](../../example/README.md).
