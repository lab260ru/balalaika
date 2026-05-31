## Overview

Prepares long recordings for ASR/TTS: **Sortformer (ONNX)** diarization,
**single-speaker** selection, **Smart Turn** end-of-segment refinement, chunk
export and `balalaika.csv`, then **crest-factor** filtering and **loudness**
normalization (ITU-R BS.1770-4).

### Stage order (`preprocess_yaml.sh`)

1. **`src.preprocess.preprocess`** — Sortformer in windows up to
   `chunk_duration`, overlap filtering, Smart VAD; writes
   `{start}_{end}_{playlist}_{podcast}.{ext}` (extension follows `chunk_format`,
   default `auto`), upserts rows into `balalaika.csv`; **deletes the original
   long file** after successful chunking.
2. **`crest_factor_remover`** — computes crest factor (peak/RMS) for every
   chunk, writes it to `balalaika.csv` as `crest_factor`, deletes files that
   exceed the threshold, and records the kept/dropped totals (files + hours)
   to `filter_summary.csv`.
3. **`preprocess_audio`** — peak + target LUFS. Lossless containers
   (FLAC / WAV) are written through `soundfile` and stay lossless; lossy
   containers (MP3 / OGG / OPUS) round-trip through `torchaudio.save`. Marks
   each successfully normalized file with `loudness_normalized=True`.

Every script writes a rotating, timestamped log under `BALALAIKA_LOG_DIR`
(default `./logs`); pass `--log_dir <path>` on the CLI to override per
invocation.

### Audio quality

* `chunk_format: auto` (default) preserves the source extension. FLAC input →
  FLAC chunks, WAV → WAV, MP3 → MP3 (no extra encode pass at chunking).
* Override with `chunk_format: flac | wav | mp3 | ogg | opus` to force a
  specific container.
* Loudness normalization keeps lossless inputs lossless. Lossy inputs
  re-encode once during normalization (unavoidable, since libsndfile cannot
  write MP3/OGG/OPUS).

### Parameters (`configs/config.yaml` → `preprocess`)

See the **preprocess** block in `configs/config.yaml` for a line-by-line
description (`podcasts_path`, `duration`, `chunk_duration`, `chunk_format`,
`min_segment_duration`, `min_save_duration`, `num_workers`, `crest_treshold`,
`peak`, `loudness`, `block_size`, `sortformer_model`, `use_tensorrt`,
`vad_args.*`). TensorRT cache / workspace come from the global `runtime:` block.

## Run

```bash
# All preprocess sub-stages via the orchestrator (stages 1..3):
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3

# Or the legacy per-folder wrapper:
bash src/preprocess/preprocess_yaml.sh configs/config.yaml
```

## Output layout

```text
{podcasts_path}/
├── balalaika.csv
├── filter_summary.csv
└── {playlist_id}/
    └── {podcast_id}/
        ├── 12.50_26.30_{playlist_id}_{podcast_id}.flac
        └── ...
```

Filename times are seconds in the **source** episode.

## `balalaika.csv` columns after preprocess

| Column | Added by |
|--------|----------|
| `filepath`, `speaker_id`, `start`, `end`, `total_duration`, `playlist_id`, `podcast_id`, `silence_percent`, `max_silence_duration`, `is_single_speaker` | `preprocess.py` |
| `crest_factor` | `crest_factor_remover.py` (files above threshold are deleted and their rows removed) |
| `loudness_normalized` | `preprocess_audio.py` (boolean flag set on success) |

`DistillMOS`, `music_prob`, and transcription fields are added in
**separation** / downstream stages.

## Filter summary rows emitted by this stage

| `stage` | Notes |
|---------|-------|
| `preprocess` | Counts long-form sources processed and total chunked hours kept. |
| `crest_factor` | Files / hours kept vs. dropped at the crest-factor threshold. |

The `loudness` step does not filter, so it does not append a row.

## Resume / interrupt safety

All three sub-stages funnel CSV state through `src.utils.csv_manager` so a
forced stop never breaks the dataset:

* **Atomic writes.** `balalaika.csv` is rewritten via tmp-file + rename, so a
  kill mid-write cannot corrupt it; a stale `.tmp` is recovered on the next
  run.
* **Auto-bootstrap.** If `balalaika.csv` is missing at the start of any
  sub-stage, it is created from the audio tree before work is scheduled.
* **Incremental partial CSVs.** Each worker streams rows to its own
  `<prefix>_part_<rank>.csv` (`crest_part_*`, `loudness_part_*`) row by row.
  A `Ctrl+C` keeps everything that already landed on disk.
* **Disk-backed work shards.** Crest-factor and loudness stages write pending
  paths to `.balalaika_work/<stage>/shard_*.pending`; workers claim those
  files instead of receiving giant Python lists through multiprocessing.
* **Resume on next run.** At startup each sub-stage absorbs any leftover
  partials into `balalaika.csv` (deleting rows for files that were removed
  by the same stage, e.g. crest-factor deletions), then schedules only the
  files still missing the relevant column.

The behaviour is idempotent: re-running `preprocess` / `crest_factor` /
`preprocess_audio` after a successful run is a no-op.
