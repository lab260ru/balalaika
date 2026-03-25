## Overview

Quality filtering on chunked clips:

1. **Music detection** — WavLM backbone + fine-tuned head at `music_detect.music_detect_model`. For every processed clip the model's music probability is written to `balalaika.csv` as `music_prob`. Clips above the threshold are deleted from disk and their rows removed from the CSV.
2. **DistillMOS** — speech quality score written to `balalaika.csv` as `DistillMOS`. Runs only on files not yet scored; skips if all entries are already present.

Speaker diarization is handled in **preprocess** (Sortformer), not here.

## Run

```bash
bash src/separation/separation_yaml.sh configs/config.yaml
```

## Parameters

Documented under **`separation`** and **`separation.music_detect`** in `configs/config.yaml` (`podcasts_path`, `bs`, `num_workers`, `music_detect_model`, `threshold`, optional `base_model` / `cache_path`).

## `balalaika.csv` columns added here

| Column | Description |
|--------|-------------|
| `music_prob` | Music classifier probability (0–1). Row removed if file deleted. |
| `DistillMOS` | Predicted MOS score. |

## Result

- Music-heavy chunks removed; CSV rows for deleted files are also removed.
- `balalaika.csv` updated with `music_prob` and `DistillMOS`; parallel runs use partial CSVs for safety.

For merged fields in exported WebDataset `json`, see [`example/README.md`](../../example/README.md).
