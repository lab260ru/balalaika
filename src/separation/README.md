## Overview

Quality filtering on chunked clips:

1. **Music detection** — WavLM backbone from Hugging Face + fine-tuned head at `music_detect.music_detect_model`. Clips above the probability threshold are **deleted**.
2. **DistillMOS** — speech quality score written to **`balalaika.csv`** (`DistillMOS` column).

Speaker diarization is handled in **preprocess** (Sortformer), not here.

## Run

```bash
bash src/separation/separation_yaml.sh configs/config.yaml
```

## Parameters

Documented under **`separation`** and **`separation.music_detect`** in `configs/config.yaml` (`podcasts_path`, `bs`, `num_workers`, `music_detect_model`, `threshold`, optional `base_model` / `cache_path`).

## Result

- Music-heavy chunks removed under `{playlist_id}/{podcast_id}/`.
- `balalaika.csv` updated (including `DistillMOS`); parallel runs use partial CSVs for safety.

For merged fields in exported WebDataset `json`, see [`example/README.md`](../../example/README.md).
