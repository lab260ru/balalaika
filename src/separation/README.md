## Overview

Quality filtering on chunked clips:

1. **Music detection** — WavLM backbone + fine-tuned head at
   `music_detect.music_detect_model`. For every processed clip the model's
   music probability is written to `balalaika.csv` as `music_prob`. Clips
   above the threshold are deleted from disk, their rows removed from the
   CSV, and the kept/dropped totals (files + hours) appended to
   `filter_summary.csv` for the final report.
2. **DistillMOS** — speech quality score written to `balalaika.csv` as
   `DistillMOS`. Annotation only — no deletion, no audit row.

Speaker diarization is handled in **preprocess** (Sortformer), not here.

Every script in this folder writes a rotating, timestamped log file under
`BALALAIKA_LOG_DIR` (default `./logs`).

## Run

```bash
# As stages 4..5 of the main runner:
bash base.sh --config_path configs/config.yaml --stage 4 --stop_stage 5

# Or the legacy per-folder wrapper:
bash src/separation/separation_yaml.sh configs/config.yaml
```

## Parameters

Documented under **`separation`** and **`separation.music_detect`** in
`configs/config.yaml` (`podcasts_path`, `bs`, `num_workers`,
`music_detect_model`, `threshold`, optional `base_model` / `cache_path`).

## `balalaika.csv` columns added here

| Column | Description |
|--------|-------------|
| `music_prob` | Music classifier probability (0–1). Row removed if file deleted. |
| `DistillMOS` | Predicted MOS score. |

## Filter summary rows emitted by this stage

| `stage` | Notes |
|---------|-------|
| `music_detect` | Files / hours kept vs. dropped at `music_detect.threshold`. |

## Resume / interrupt safety

Both sub-stages funnel CSV state through `src.utils.csv_manager`:

* **Atomic writes** of `balalaika.csv` (tmp file + rename).
* **Auto-bootstrap** of `balalaika.csv` from the audio tree if missing — so
  this folder can run as the *first* CSV-touching stage if needed.
* **Incremental partial CSVs.** Workers stream rows to
  `music_part_<rank>.csv` and `distillmos_part_<rank>.csv` (one flush per
  row), so a forced stop preserves whatever was already scored.
* **Resume on next run.** At startup each sub-stage absorbs any leftover
  partials into `balalaika.csv` and schedules only the files that are still
  missing the relevant column (`music_prob` / `DistillMOS`). Files deleted
  by `music_detect` are pruned from `balalaika.csv` during the merge.

Re-running these scripts after a successful run is a no-op.

## Result

- Music-heavy chunks removed; CSV rows for deleted files are also removed.
- `balalaika.csv` updated with `music_prob` and `DistillMOS`; parallel runs
  use partial CSVs for safety.

For merged fields in exported WebDataset `json`, see
[`example/README.md`](../../example/README.md).
