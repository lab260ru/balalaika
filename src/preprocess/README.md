## Overview

Prepares long recordings for ASR/TTS: **Sortformer (ONNX)** diarization, **single-speaker** selection, **Smart Turn** end-of-segment refinement, chunk export and `balalaika.csv`, then **crest-factor** filtering and **loudness** normalization (ITU-R BS.1770-4).

### Stage order (`preprocess_yaml.sh`)

1. **`src.preprocess.preprocess`** — Sortformer in windows up to `chunk_duration`, overlap filtering, Smart VAD, writes `{start}_{end}_{playlist}_{podcast}.mp3`, appends `balalaika.csv`; **deletes the original long file** after successful chunking.
2. **`crest_factor_remover`** — removes files with excessive peak/RMS ratio.
3. **`preprocess_audio`** — peak + target LUFS, overwrites audio.

### Parameters (`configs/config.yaml` → `preprocess`)

See the **preprocess** block in `configs/config.yaml` for a line-by-line description (`podcasts_path`, `duration`, `chunk_duration`, `num_workers`, `crest_treshold`, `peak`, `loudness`, `block_size`, `sortformer_model`, `use_tensorrt`, `vad_args.*`).

## Run

```bash
bash src/preprocess/preprocess_yaml.sh configs/config.yaml
```

(Optional: `src/preprocess/preprocess_args.sh` for ad-hoc CLI args.)

## Output layout

```text
{podcasts_path}/
├── balalaika.csv
└── {playlist_id}/
    └── {podcast_id}/
        ├── 12.50_26.30_{playlist_id}_{podcast_id}.mp3
        └── ...
```

Filename times are seconds in the **source** episode.

## `balalaika.csv` columns after preprocess

Includes `filepath`, `speaker_id`, `start`, `end`, `total_duration`, `playlist_id`, `podcast_id`, `silence_percent`, `max_silence_duration`, `is_single_speaker`. **DistillMOS** and later fields are added in **separation** / downstream stages.
