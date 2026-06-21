## Overview

Quality filtering and quality annotation on chunked clips. Speaker diarization
is handled in **preprocess** (Sortformer), not here. The separation package
pairs each **scoring** stage with a **filter** stage. A scoring stage only
writes its column to `balalaika.parquet`; the paired filter stage shows the
distribution, picks a threshold, deletes the matching clips, and appends the
kept/dropped totals to `filter_summary.csv`. Each scoring stage also accepts
`inline_filter: true` (config), which applies the same delete predicate **in the
scoring pass** (reading the threshold from the paired `*_filter` subsection) so
the separate filter stage can be skipped — see `inline_filter.py`. The package
contains these model/filter stages:

1. **Music detection (scoring)** — WavLM backbone + fine-tuned head. For every
   processed clip the model's music probability is written to
   `balalaika.parquet` as `music_prob`. Annotation only; no deletion (unless
   `music_detect.inline_filter` is set).
1.5. **Music filter** — deletes clips with `music_prob` **above**
   `separation.music_detect_filter.threshold` and records a `music_detect_filter`
   audit row in `filter_summary.csv`.
2. **DistillMOS scoring** — speech quality score written to `balalaika.parquet`
   as `DistillMOS`. Annotation only; no deletion.
3. **DistillMOS filter** — deletes clips below
   `separation.distillmos_filter.threshold` after scoring and records an audit
   row in `filter_summary.csv`.
4. **Anti-spoofing scoring** — [Spectra-0](https://huggingface.co/lab260/spectra_0)
   ONNX classifier. It writes the untouched model outputs as `score_spoof`
   (output index 0) and `score_bonafide` (output index 1). No deletion occurs.
5. **Anti-spoofing filter** — computes
   `score_spoof - score_bonafide`, deletes clips above
   `separation.antispoofing_filter.threshold`, and records an
   `antispoofing_filter` audit row.
6. **TTS-suitability scoring** —
   [TTS-Suitability-Classifier](https://huggingface.co/lab260/TTS-Suitability-Classifier)
   ONNX classifier (wav2vec2-300M head). Each clip is layer-normalized, split
   into 10 s chunks, and the per-chunk logits are averaged; a simple softmax is
   applied and the probabilities are written to `balalaika.csv` as `p_not_tts`
   and `p_tts`. No deletion occurs.
7. **TTS-suitability filter** — computes
   `p_tts`, deletes clips below
   `separation.tts_suitability_filter.threshold`, and records a `tts_suitability_filter`
   audit row.

Every script writes a rotating, timestamped log file under `BALALAIKA_LOG_DIR`
(default `./logs`).

## Models

| Stage | Model | Runtime | Notes |
|-------|-------|---------|-------|
| Music detection | WavLM + fine-tuned head | PyTorch | Local safetensors head configured by `music_detect_model`. |
| DistillMOS | DistillMOS | PyTorch | Adds MOS estimate only. |
| Anti-spoofing | [Spectra-0](https://huggingface.co/lab260/spectra_0) | ONNX Runtime | Binary bonafide/spoof classifier on raw waveforms; downloads `model.onnx` if missing. |
| TTS-suitability | [TTS-Suitability-Classifier](https://huggingface.co/lab260/TTS-Suitability-Classifier) | ONNX Runtime | wav2vec2-300M binary not_tts/tts classifier; per-file 10 s chunking + mean logits + simple softmax; downloads `model.onnx` if missing. |

## Run

```bash
# Stages 4..7.5 of the main runner:
bash base.sh --config_path configs/config.yaml --stage 4 --stop_stage 7.5

# Or the legacy wrapper for music + DistillMOS scoring only:
bash src/separation/separation_yaml.sh configs/config.yaml

# Individual stages:
python -m src.separation.music_detect          --config_path configs/config.yaml
python -m src.separation.music_detect_filter   --config_path configs/config.yaml
python -m src.separation.distillmos_process    --config_path configs/config.yaml
python -m src.separation.distillmos_filter     --config_path configs/config.yaml
python -m src.separation.antispoofing          --config_path configs/config.yaml
python -m src.separation.antispoofing_filter   --config_path configs/config.yaml
python -m src.separation.tts_suitability        --config_path configs/config.yaml
python -m src.separation.tts_suitability_filter --config_path configs/config.yaml
```

## Parameters

Documented under **`separation`** in `configs/config.yaml`:

| Config key | Purpose |
|------------|---------|
| `podcasts_path` | Dataset root containing audio files and `balalaika.csv`. |
| `music_detect.*` | WavLM classifier batch/DataLoader settings, model path, `inline_filter`. |
| `music_detect_filter.*` | Music-prob threshold (delete above) and deletion worker count. |
| `distillmos.*` | DistillMOS batch/DataLoader settings, `inline_filter`. |
| `distillmos_filter.*` | Threshold and deletion worker count. |
| `antispoofing.onnx_path` | Local Spectra-0 ONNX path. Missing file is downloaded from HF. |
| `antispoofing.batch_size`, `num_workers`, `prefetch_factor` | Spectra-0 batching and DataLoader settings. |
| `antispoofing.use_tensorrt` | Enable TensorRT EP when the ONNX/profile supports it. |
| `antispoofing_filter.threshold` | Delete when `score_spoof - score_bonafide` exceeds this raw-score margin. |
| `antispoofing_filter.num_workers` | Parallel CPU deletion workers. |
| `tts_suitability.onnx_path` | Local TTS-suitability ONNX path. Missing file is downloaded from HF. |
| `tts_suitability.batch_size`, `num_workers`, `prefetch_factor` | DataLoader grouping/prefetch (inference is per file). |
| `tts_suitability.use_tensorrt` | Enable TensorRT EP (off by default; inputs are variable-length). |
| `tts_suitability_filter.threshold` | Delete when `p_tts` is below this threshold. |
| `tts_suitability_filter.num_workers` | Parallel CPU deletion workers. |

## `balalaika.csv` columns added here

| Column | Description |
|--------|-------------|
| `music_prob` | Music classifier probability (0-1). Row removed if file deleted. |
| `DistillMOS` | Predicted MOS score. |
| `score_bonafide` | Raw Spectra-0 output at class index 1. |
| `score_spoof` | Raw Spectra-0 output at class index 0. |
| `p_not_tts` | TTS-suitability probability for class `not_tts`. |
| `p_tts` | TTS-suitability probability for class `tts`. |

## Filter summary rows emitted here

| `stage` | Notes |
|---------|-------|
| `music_detect` | Files / hours kept vs. dropped at `music_detect.threshold`. |
| `distillmos_filter` | Files / hours kept vs. dropped at `distillmos_filter.threshold`. |
| `antispoofing_filter` | Files / hours kept vs. dropped at the configured raw-score margin. |
| `tts_suitability_filter` | Files / hours kept vs. dropped at the configured `p_tts` threshold. |

## Resume / Interrupt Safety

All long-running sub-stages use `src.utils.csv_manager`:

- **Atomic writes** of `balalaika.csv` through tmp file + rename.
- **Auto-bootstrap** from the audio tree if the CSV is missing.
- **Incremental partial CSVs** such as `music_part_<rank>.csv`,
  `distillmos_part_<rank>.csv`, `antispoof_part_<rank>.csv`,
  `antispoof_filter_part_<rank>.csv`, `tts_suit_part_<rank>.csv`, and
  `tts_suit_filter_part_<rank>.csv`.
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
- TTS-unsuitable speech removed by the TTS-suitability classifier.
- `balalaika.csv` and `filter_summary.csv` updated for downstream report and
  export stages.

For merged fields in exported WebDataset `json`, see
[`example/README.md`](../../example/README.md).
