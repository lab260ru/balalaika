# Balalaika — Pipeline Documentation

## 1. Overview

Balalaika is an end-to-end speech-data pipeline that transforms long-form Russian
audio recordings (podcasts, interviews, radio) into cleaned, segmented, annotated
datasets ready for TTS/ASR training.  The pipeline is described in the paper
*"A Data-Centric Framework for Addressing Phonetic and Prosodic Challenges in
Russian Speech Generative Models"* (arXiv:2507.13563, Borodin et al., 2025).

It is a **Kaldi-style stage-based pipeline**: you run `bash base.sh --stage N
--stop_stage M` and everything between (inclusive) is executed. Each stage is
idempotent — you can stop at any point and resume from where you left off.

---

## 2. Pipeline Orchestration (`base.sh`)

`base.sh` is the single entry point. It reads `configs/config.yaml`, activates
the Python venv, and calls each stage's Python module in order.

```bash
bash base.sh --config_path configs/config.yaml                   # Full pipeline (stages 1–9)
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3  # Preprocess only
bash base.sh --config_path configs/config.yaml --stage 6 --stop_stage 6  # Transcription only
```

### How `base.sh` works internally

1. Parses `--stage` (default 1) and `--stop_stage` (default 9).
2. Runs `python3 -m src.utils.runtime_env --config_path ...` which prints shell
   `export` statements (venv path, CPU affinity, log dir, TensorRT cache). The
   output is `eval`'d, setting environment variables like `BALALAIKA_VENV`,
   `BALALAIKA_CPU_AFFINITY`, `BALALAIKA_LOG_DIR`, `BALALAIKA_TRT_CACHE_PATH`,
   `BALALAIKA_TRT_WORKSPACE`, and `BALALAIKA_TRT_FP16`.
3. Activates the venv and sets up `LD_LIBRARY_PATH` for NVIDIA/CUDA/TensorRT libs.
4. For each of the 13 stages (0–12), checks whether `stage <= N <= stop_stage`.
   If yes, calls the stage's Python module with:
   ```bash
   taskset -c $BALALAIKA_CPU_AFFINITY python3 -m <module> --config_path ... --log_dir ...
   ```

---

## 3. Stage-by-Stage Breakdown

---

### Stage 0 — Download (`src/download/download.py`)

**Purpose:** Download podcast episodes from Yandex Music.

**Input:**
- A pickle file (`podcasts_urls_file` in config) containing podcast URLs, OR
  a direct playlist/podcast ID.
- `YANDEX_KEY` environment variable (Yandex Music API token).

**Process:**
1. Initializes `yandex_music.Client` with the token.
2. For each podcast URL: extracts the podcast (album) ID, fetches track metadata.
3. Downloads each episode as `.mp3` via `requests.get()`.
4. Tags each file with metadata (title, artist, track number) via `music_tag`.
5. Output organized as `{podcasts_path}/{podcast_id}/{episode_id}.mp3`.

**Output:**
- MP3 files on disk under `{podcast_id}/{episode_id}/`.

**Idempotency:** Checks whether the target file or folder already exists.
Skips if so. No CSV state.

**Config section:** `download`

---

### Stage 1 — Diarization + Chunking (`src/preprocess/preprocess.py`)

**Purpose:** Identify who speaks when in long audio, split into single-speaker
segments, refine boundaries with Smart Turn EOS detection, and export chunks.

**Input:**
- Long-form audio files (`.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`) under
  `<podcasts_path>/<playlist_id>/<podcast_id>/`.

**Process (4 steps):**

1. **Skip already-chunked files.** Files matching the chunk-name pattern
   `^\d+\.\d+_\d+\.\d+_` (e.g. `0.00_15.00_abc_def.flac`) are skipped.

2. **Sortformer diarization** (`src/preprocess/sortformer_onnx.py`):
   - Streams the ONNX Sortformer model in 900-second windows over the audio.
   - Runs on TensorRT (if enabled) or CUDA via `get_onnx_providers()`.
   - Returns speaker segments: `[(start_sec, end_sec, speaker_id), ...]`.
   - One `ProcessPoolExecutor` per GPU holds the model.

3. **Single-speaker timeline** (`build_single_speaker_timeline`):
   - Trims overlapping segments from different speakers at the midpoint.
   - Merges same-speaker overlaps.
   - Splits turns longer than `max_duration` (default 15 s) into windows.

4. **Smart Turn EOS refinement** (`apply_eos_classification`):
   - Uses `SmartVAD` (`src/libs/smart_turn/offline_svad.py`), an ONNX model
     predicting end-of-speech (EOS) probabilities.
   - Buffers low-EOS segments; saves final chunks when EOS is detected.
   - Enforces `min_segment_duration` (1.0 s), `max_merge_gap` (0.5 s),
     `smart_vad_threshold` (0.4).

5. **Chunk export** (`cut_audio`):
   - Saves audio segments as `{start}_{end}_{playlist_id}_{podcast_id}.{ext}`.
   - Default `chunk_format: auto` preserves source container (FLAC stays FLAC).
   - Computes per-chunk: `silence_percent`, `max_silence_duration`,
     `is_single_speaker`.
   - **Deletes the original long file** after successful chunking.
   - Short files (`total_duration <= max_duration`) are kept in place.

**Output:**
- Chunked audio files in `{playlist_id}/{podcast_id}/` subdirectories.
- `balalaika.csv` upserted with columns:
  `filepath`, `speaker_id`, `start`, `end`, `total_duration`, `playlist_id`,
  `podcast_id`, `silence_percent`, `max_silence_duration`, `is_single_speaker`.
- Audit row appended to `filter_summary.csv` (stage: `preprocess`).

**Idempotency:**
- Source files that match the chunk pattern are skipped.
- `upsert_columns()` merges on `filepath` — new chunks are added, existing ones
  are overwritten.
- Source files are deleted only after chunks are safely written to disk.

**Config section:** `preprocess`

---

### Stage 2 — Crest Factor Filter (`src/preprocess/crest_factor_remover.py`)

**Purpose:** Delete audio files whose crest factor (peak/RMS amplitude ratio)
exceeds a threshold, indicating heavily distorted or clipped audio.

**Input:** All audio files under `podcasts_path`.

**Process:**
1. Computes `crest_factor = peak / RMS` for each file using GPU-batched
   computation via a PyTorch DataLoader.
2. Writes `crest_factor` to `balalaika.csv`.
3. If `crest_factor > crest_threshold` (default 10.0), **deletes** the audio file.
4. Records the deletion status and duration in a per-worker partial CSV.

**Output:**
- `crest_factor` column in `balalaika.csv`.
- Audio files exceeding threshold are deleted from disk.
- Deleted files are pruned from `balalaika.csv` (`drop_missing_files=True`).
- Audit row in `filter_summary.csv` (stage: `crest_factor`).

**Idempotency (full resume support):**
1. `ensure_main_csv()` — creates `balalaika.csv` from the audio tree if missing.
2. `absorb_partial_csvs()` — merges any leftover `crest_part_*.csv` from a
   previously interrupted run into the main CSV.
3. `unprocessed_paths()` — finds only files missing a `crest_factor` value.
4. Workers stream rows to `crest_part_<rank>.csv` with `flush()` after every row.
5. On `KeyboardInterrupt`, whatever partial rows exist are merged on next run.

**Config section:** `preprocess` (keys: `crest_treshold`,
`num_workers_crest_factor`, `crest_factor_batch_size`)

---

### Stage 3 — Loudness Normalization (`src/preprocess/preprocess_audio.py`)

**Purpose:** Normalize audio loudness to ITU-R BS.1770-4 standard.

**Input:** All audio files under `podcasts_path`.

**Process:**
1. Applies peak normalization first (`pyloudnorm.normalize.peak`).
2. Measures integrated LUFS via `pyloudnorm.Meter`.
3. Normalizes to target loudness via `pyloudnorm.normalize.loudness`.
4. Target: -23.0 LUFS, peak ceiling: -1.0 dBFS, block size: 0.400 s.
5. Writes back in-place via `torchaudio.save`.

**Output:**
- Normalized audio files (overwritten in-place).
- `loudness_normalized` boolean column in `balalaika.csv`.

**Idempotency:** Same pattern as crest factor — `ensure_main_csv`,
`absorb_partial_csvs` for `loudness_part_*.csv`, `unprocessed_paths` on
`loudness_normalized` column. No audit row (this stage transforms, doesn't
delete).

**Config section:** `preprocess` (keys: `peak`, `loudness`, `block_size`,
`loudness_num_workers`, `loudness_batch_size`)

---

### Stage 4 — Music Detection (`src/separation/music_detect.py`)

**Purpose:** Detect and delete music-heavy audio clips.

**Input:** All audio files under `podcasts_path`.

**Process:**
1. Loads a fine-tuned WavLM model (`microsoft/wavlm-base-plus` backbone,
   custom classification head loaded from safetensors).
2. Uses `LengthBasedBatchSampler` from `musicdetection` for efficient batching.
3. Predicts `music_prob` (0–1) per file.
4. If `music_prob > threshold` (default 0.6), **deletes** the file from disk.

**Output:**
- `music_prob` column in `balalaika.csv`.
- Music-heavy files deleted from disk and pruned from CSV.
- Audit row in `filter_summary.csv` (stage: `music_detect`).

**GPU parallelism:** One `mp.Process` per GPU, each holding its own model copy
and processing a round-robin shard of files.

**Idempotency:** Same pattern as Stage 2 — `ensure_main_csv`,
`absorb_partial_csvs` for `music_part_*.csv`, `unprocessed_paths` on
`music_prob` column.

**Config section:** `separation.music_detect`

---

### Stage 5 — DistillMOS Scoring (`src/separation/distillmos_process.py`)

**Purpose:** Predict a Mean Opinion Score (MOS) for each audio clip.

**Input:** All audio files under `podcasts_path`.

**Process:**
1. Loads `distillmos.ConvTransformerSQAModel` per GPU.
2. Uses `DistillMOSDataset` with length-sorted batching for throughput.
3. Resamples all audio to 16 kHz, runs inference.

**Output:**
- `DistillMOS` column in `balalaika.csv` (float score).
- No deletion — annotation only. No audit row.

**GPU parallelism:** One `mp.Process` per GPU.

**Idempotency:** `ensure_main_csv`, `absorb_partial_csvs` for
`distillmos_part_*.csv`, `unprocessed_paths` on `DistillMOS` column.

**Config section:** `separation.distillmos`

---

### Stage 6 — Transcription (`src/transcription/transcription.py`)

**Purpose:** Multi-model ASR via onnx-asr, with optional consensus skip, word-level
timestamps, and ROVER aggregation.

**Input:** All audio files under `podcasts_path`.

**Process:**
1. Runs multiple ASR models sequentially from the `model_names` list.
2. **Consensus skip:** if `consensus_num` earlier models agree on normalized
   text for a file, later models are skipped for that file.
3. Supported model mappings:
   - `giga_rnnt` → `gigaam-v3-rnnt`
   - `giga_ctc` / `giga_ctc_lm` → `gigaam-v3-ctc`
   - `tone` → `t-tech/t-one`
   - `vosk` → `alphacep/vosk-model-ru`
   - `parakeet_v2/v3`, `canary`, `whisper_*`
4. Word-level timestamps in `.tst` TSV format for models in
   `SUPPORTED_TIMESTAMPS` set (giga_ctc, tone, parakeet, canary).
5. TensorRT via `get_onnx_providers()` when `use_tensorrt: True`.
6. **ROVER aggregation** (`src/transcription/rover.py`): uses crowd-kit's ROVER
   to produce consensus text from all model outputs.

**Output per chunk:**
- `{stem}_{model}.txt` — per-model ASR text.
- `{stem}_{model}.tst` — word-level timestamps (TSV: `start\tend\tword`, if supported).
- `{stem}_rover.txt` — ROVER consensus text (if `use_rover: True`).

**GPU parallelism:** `run_per_gpu_processes()` — one `mp.Process` per GPU, each
loading one model.

**Idempotency:** No CSV state. Checks for existing `{stem}_{model}.txt` files.
Each model skips chunks that already have output. ROVER also checks for
existing `_rover.txt`.

**Config section:** `transcription`

---

### Stage 7 — Punctuation (`src/punctuation/punctuation.py`)

**Purpose:** Restore punctuation and capitalization on ROVER consensus text using
RUPunct.

**Input:** `*_rover.txt` sidecar files (next to audio chunks).

**Process:**
1. Loads RUPunct via HuggingFace `transformers.pipeline("ner")` (token
   classification task).
2. Maps 22 entity labels to output characters via `process_token()`:
   - `LOWER_PERIOD` → lowercase + `.`, `UPPER_COMMA` → capitalized + `,`,
   - `UPPER_TOTAL_QUESTIONVOSKL` → ALL-CAPS + `?!`, etc.

**Output:** `{stem}_punct.txt` — text with punctuation and capitalization.

**GPU parallelism:** `run_per_gpu_pool()` — one `ProcessPoolExecutor` per GPU,
multiple workers per pool.

**Idempotency:** Uses `pending_audio_to_sidecar(in_suffix="_rover.txt",
out_suffix="_punct.txt")` to find only chunks missing `_punct.txt`. Output is
written atomically vial `Path.write_text()`.

**Config section:** `punctuation`

---

### Stage 8 — Accents / Stress Marks (`src/accents/accents.py`)

**Purpose:** Add lexical stress marks and normalize text using ruAccent.

**Input:** `*_punct.txt` sidecar files.

**Process:**
1. Loads `ruaccent.RUAccent()` per worker.
2. Supports TensorRT ONNX providers.
3. Calls `accentizer.process_all(text)` on each file content.

**Output:** `{stem}_accent.txt` — text with stress marks (e.g. `приве́т`).

**GPU parallelism:** `run_per_gpu_pool()`.

**Idempotency:** Uses `pending_sidecar_chain(in_suffix="_punct.txt",
out_derive=lambda p: replace_in_stem(p, "_punct", "_accent"))`.

**Config section:** `accent` (singular — note the key name)

---

### Stage 9 — Phonemizer (`src/phonemizer/phonemizer.py`)

**Purpose:** Convert grapheme text to IPA phonemes using TryIParu.

**Input:** `*_rover.txt` sidecar files.

**Process:**
1. Loads `tryiparu.G2PModel(load_dataset=True, device=...)` per GPU.
2. Calls `g2p_model(text)` to produce a list of IPA symbols.
3. Joins with space separator.

**Output:** `{stem}_rover_phonemes.txt` — space-separated IPA phonemes.

**GPU parallelism:** `run_per_gpu_pool()`.

**Idempotency:** Uses `pending_sidecar_chain(in_suffix="_rover.txt",
out_derive=lambda p: p.with_name(f"{p.stem}_phonemes.txt"))`.

**Config section:** `phonemizer`

---

### Stage 10 — Collate to Parquet (`src/collate.py`)

**Purpose:** Merge all metadata and sidecar text files into a single Parquet file.

**Input:** `balalaika.csv` + all audio chunks + sidecar `.txt` files.

**Process:**
1. Reads `balalaika.csv` (or creates from audio tree if absent).
2. For each `filepath`, reads four sidecar text files:
   - `_accent.txt`, `_rover.txt`, `_punct.txt`, `_rover_phonemes.txt`.
3. Uses `concurrent.futures.ThreadPoolExecutor` for parallel I/O.
4. Left-joins text data on `filepath` column.
5. Saves to `balalaika.parquet` using PyArrow engine.

**Output:** `balalaika.parquet` — aggregated metadata + text in columnar format.

**Idempotency:** Overwrites `balalaika.parquet` on each run. Fast, so no
resume logic needed.

**Config section:** `download` (reads `podcasts_path` and `num_workers` from this section)

---

### Stage 11 — Export to WebDataset (`src/to_webdataset.py`)

**Purpose:** Pack audio bytes + metadata + text sidecars into WebDataset tar
shards for efficient streaming training.

**Input:** Audio files + `balalaika.csv`.

**Process:**
1. Reads `balalaika.csv` metadata into a dictionary keyed by file stem.
2. For each audio file:
   - Reads raw audio bytes (no re-encoding).
   - Collects all sidecar `.txt` files for that stem.
   - Builds a JSON dict with CSV columns + sidecar text.
   - Replaces dots in keys with underscores for HuggingFace/WebDataset compatibility.
   - Writes `{safe_key}.{ext}` (audio) + `{safe_key}.json` (metadata) samples.
3. Sharded into `.tar` files: `shard_{worker_id:03d}_{shard_id:04d}.tar`.
4. Default max shard size: 512 MiB; max samples per shard: 10,000.

**Output:** `{podcasts_path}_webdataset/train/shard_*.tar` — WebDataset shards.

**GPU parallelism:** CPU-bound `ProcessPoolExecutor`. Each worker writes its own
shard series.

**Config section:** `export`

---

### Stage 12 — Filter Report (`src/report.py`)

**Purpose:** Generate a human-readable Markdown report showing hours kept/removed
at each filtering stage.

**Input:** `<podcasts_path>/filter_summary.csv`.

**Process:**
1. Reads all audit rows from `filter_summary.csv`.
2. Groups by stage, takes the latest run per stage.
3. Computes per-stage: files in/out, hours in/out/removed, % removed.
4. Computes pipeline net effect.
5. Includes full history table of all runs.

**Output:** `<podcasts_path>/filter_report.md` — Markdown report with:
- Per-stage summary table (latest run of each stage).
- Pipeline net effect (total hours in vs out).
- Full history table (every run ever recorded).

**Config section:** Resolves `podcasts_path` from `download`, `preprocess`,
`separation`, or `export` sections.

---

## 4. Input / Output Summary Table

| Stage | Primary Input | Primary Output |
|-------|---------------|----------------|
| 0 | Podcast URLs / pickle file | `{podcast_id}/{episode_id}.mp3` |
| 1 | Long audio files | Chunked audio + `balalaika.csv` (10 columns) + `filter_summary.csv` |
| 2 | All audio files | Deleted high-crest audio; `crest_factor` in CSV + audit |
| 3 | All audio files | Normalized audio (in-place); `loudness_normalized` in CSV |
| 4 | All audio files | Deleted music-heavy files; `music_prob` in CSV + audit |
| 5 | All audio files | `DistillMOS` in CSV (annotation only) |
| 6 | All audio files | `{stem}_{model}.txt` + `.tst` + `_rover.txt` |
| 7 | `*_rover.txt` | `{stem}_punct.txt` |
| 8 | `*_punct.txt` | `{stem}_accent.txt` |
| 9 | `*_rover.txt` | `{stem}_rover_phonemes.txt` |
| 10 | `balalaika.csv` + sidecars | `balalaika.parquet` |
| 11 | Audio + `balalaika.csv` | WebDataset `.tar` shards |
| 12 | `filter_summary.csv` | `filter_report.md` |

---

## 5. Idempotency & Resume System

The pipeline can be stopped at any point (`Ctrl+C`, process kill, power loss)
and resumed safely. The resume system has four layers.

### Layer 1: Stage-level control

`base.sh --stage N --stop_stage M` lets you run any contiguous subset of stages.
If the pipeline fails at stage 6, re-run `--stage 6` to continue from there.

### Layer 2: CSV state with atomic writes (`balalaika.csv`)

Used by stages 2–5 (crest factor, loudness, music detection, DistillMOS) via
`src/utils/csv_manager.py`.

**`ensure_main_csv(podcasts_path, audio_paths)`**
- If `balalaika.csv` doesn't exist, creates it from the audio tree with all
  discovered file paths. Any stage can be the first to run.

**`atomic_write_csv(df, path)`**
- Writes to a `.tmp` file, calls `fsync()`, then `os.replace()` to atomically
  swap the tmp file in place.
- If the process is killed mid-write, only the `.tmp` file is corrupted; the
  real CSV is intact.
- On next read, if the real CSV is missing or unreadable but a `.tmp` exists,
  the `.tmp` is recovered.

**Partial CSVs: `absorb_partial_csvs()` / `PartialCsvWriter`**
- Each worker writes to its own `<prefix>_part_<rank>.csv` (e.g.
  `crest_part_0.csv`, `music_part_1.csv`).
- `PartialCsvWriter.flush()` is called after every row, so `SIGKILL` preserves
  everything produced so far.
- On startup, `absorb_partial_csvs()` merges any leftover partial files into
  the main CSV, then deletes the partials. Re-running a stage **resumes**
  instead of re-computing everything.

**`unprocessed_paths(podcasts_path, column, audio_paths)`**
- Returns only files that don't yet have a value in the target column (e.g.
  `crest_factor`, `music_prob`, `DistillMOS`). Workers round-robin over the
  pending list.

**The resume flow for stages 2–5:**

```
Step 1: ensure_main_csv()           # Create CSV from audio tree if missing
Step 2: absorb_partial_csvs()       # Merge any leftover partials from prior crash
Step 3: unprocessed_paths()         # Find files still needing this column
Step 4: mp.spawn() / run_workers()  # Process only pending files
Step 5: absorb_partial_csvs()       # Merge new results (also on Ctrl+C)
```

### Layer 3: Sidecar file checks (stages 6–9)

Stages 6–9 (transcription, punctuation, accents, phonemizer) don't use
`balalaika.csv` at all. Instead, they check for the existence of output files:

- **Transcription (Stage 6):** Checks `{stem}_{model}.txt` per model. Files
  with existing output are skipped. Also uses `consensus_num`: if enough
  earlier models agree on normalized text, later models skip that file
  entirely.
- **Punctuation (Stage 7):** `pending_audio_to_sidecar(in_suffix="_rover.txt",
  out_suffix="_punct.txt")` — finds only audio chunks that have `_rover.txt`
  but lack `_punct.txt`.
- **Accents (Stage 8):** `pending_sidecar_chain(in_suffix="_punct.txt",
  out_derive=...)` — finds only `_punct.txt` files missing `_accent.txt`.
- **Phonemizer (Stage 9):** `pending_sidecar_chain(in_suffix="_rover.txt",
  out_derive=...)` — finds only `_rover.txt` files missing
  `_rover_phonemes.txt`.

Output files are written atomically via `Path.write_text()`, so a crash during
write leaves no partial file that could be mistaken as complete.

### Layer 4: Download skip (Stage 0)

The downloader checks whether the target file or folder already exists on disk.
If a file or episode folder is present, it's skipped. No CSV involved.

### What happens on SIGINT / SIGKILL?

- **CSV-based stages:** `KeyboardInterrupt` handler calls `absorb_partial_csvs()`
  before exit, merging whatever partial rows exist into the main CSV. Even if
  `SIGKILL` prevents the handler, the partial files remain on disk and are
  absorbed on the next run.
- **Sidecar stages:** Output files are written atomically. Partial writes
  during `SIGKILL` don't produce output files, so those chunks will be
  re-processed on the next run.
- **Atomic CSV writes:** `atomic_write_csv` uses tmp file + rename, so the
  main CSV is never corrupted mid-write. If the original file is somehow lost
  but `.tmp` survives, `_read_csv_safe()` recovers it.
- **Chunking (Stage 1):** Source files are deleted only after chunks are
  successfully written to disk by `cut_audio()`. If the process is killed
  between chunk export and file deletion, the source file may survive as a
  duplicate, but no data is lost.

---

## 6. Report & Audit System

### Filter Summary (`filter_summary.csv`)

Filtering stages (1-preprocess, 2-crest_factor, 4-music_detect) call
`record_stage_summary()` from `src/utils/audit.py`, which appends a row with:

| Column | Description |
|--------|-------------|
| `timestamp` | UTC ISO-8601 timestamp |
| `stage` | Stage name (`preprocess`, `crest_factor`, `music_detect`) |
| `files_in` | Number of files entering the stage |
| `files_out` | Number of files after filtering |
| `hours_in` | Total audio hours entering |
| `hours_out` | Total audio hours surviving |
| `hours_removed` | `hours_in - hours_out` (capped at 0) |
| `params` | JSON blob with stage-specific parameters (threshold, etc.) |

This file is stored at `<podcasts_path>/filter_summary.csv` and accumulates
rows over multiple pipeline runs.

### Filter Report (`filter_report.md`)

Stage 12 (`src/report.py`) reads `filter_summary.csv` and generates a Markdown
report at `<podcasts_path>/filter_report.md` with three sections:

1. **Per-stage summary table:** For the latest run of each stage, shows the
   files in → out, hours in → out → removed, percentage removed, and the
   parameters used.

2. **Pipeline net effect:** Total hours in (from the first filter stage) vs
   total hours out (from the last stage), showing total hours and percentage
   removed across the whole pipeline.

3. **Full history table:** Every run ever recorded, sorted by timestamp.

Example usage:
```bash
bash base.sh --config_path configs/config.yaml --stage 12 --stop_stage 12
```

### Per-Stage Log Files

Every stage writes a timestamped rotating log file via `src/utils/logging_setup.py`:
```
<log_dir>/<stage>_YYYYMMDD-HHMMSS.log
```
Log files rotate at 200 MB and keep the 10 most recent files per stage.

---

## 7. Configuration (`configs/config.yaml`)

Each stage reads only its own YAML section via `load_config(config_path, SECTION)`.

### Top-level keys
| Key | Purpose |
|-----|---------|
| `cache_path` | General cache directory (benchmarks) |
| `runtime` | Orchestration: venv path, CPU affinity, log dir, TensorRT cache |
| `download` | Yandex Music downloader |
| `preprocess` | Diarization, crest filter, loudness (stages 1–3) |
| `separation` | Music detection + DistillMOS (stages 4–5) |
| `transcription` | onnx-asr ASR + ROVER (stage 6) |
| `punctuation` | RUPunct (stage 7) |
| `accent` | ruAccent (stage 8) — note: singular `accent` |
| `phonemizer` | TryIParu G2P (stage 9) |
| `export` | WebDataset shard export (stage 11) |

### `runtime` block keys

| Key | Default | Purpose |
|-----|---------|---------|
| `venv_path` | `.dev_venv` | Python venv path |
| `cpu_affinity` | `"0-32"` | `taskset -c` range (empty = disable) |
| `log_dir` | `./logs` | Per-stage log directory |
| `trt_cache_path` | `./cache/trt` | TensorRT engine cache root |
| `trt_workspace_bytes` | 4 GiB | Per-session TensorRT workspace |
| `trt_fp16` | `True` | FP16 for TensorRT |

### Important note
`src/collate.py` (Stage 10) reads the **`download`** section for
`podcasts_path` and `num_workers`, so keep `download.podcasts_path` aligned
with the rest of the pipeline even if you don't use Stage 0.

---

## 8. Key Models and Tools

| Model | Purpose | Format |
|-------|---------|--------|
| Sortformer | Streaming speaker diarization | ONNX |
| Smart Turn v3.2 | End-of-speech detection | ONNX |
| WavLM (fine-tuned) | Music detection | Safetensors |
| DistillMOS | Speech quality prediction (MOS) | PyTorch |
| onnx-asr (GigaAM, Vosk, T-one, Whisper, etc.) | Speech-to-text | ONNX |
| ROVER (crowd-kit) | Multi-model consensus aggregation | Python |
| RUPunct | Punctuation restoration | HuggingFace |
| ruAccent | Lexical stress marks | ONNX/PyTorch |
| TryIParu | Grapheme → IPA phonemes | PyTorch |

---

## 9. Repository Structure

```
balalaika/
├── base.sh                          # Main orchestrator (--stage / --stop_stage)
├── src/
│   ├── __init__.py
│   ├── collate.py                   # Stage 10: Parquet collation
│   ├── to_webdataset.py             # Stage 11: WebDataset export
│   ├── report.py                    # Stage 12: Filter report
│   ├── recovery_from_meta.py        # Reconstruct chunks from parquet metadata
│   ├── stage_runner.sh              # Shared shell bootstrap
│   │
│   ├── utils/
│   │   ├── utils.py                 # load_config, get_audio_paths, get_txt_paths, etc.
│   │   ├── logging_setup.py         # loguru: stderr + rotating file
│   │   ├── csv_manager.py           # Core CSV resilience (bootstrap, atomic, partials, resume)
│   │   ├── audit.py                 # filter_summary.csv recording
│   │   ├── parallel.py              # Multi-GPU parallelism: run_per_gpu_pool / run_per_gpu_processes
│   │   ├── gpu.py                   # TF32/SDP defaults, ONNX providers (CUDA/TensorRT)
│   │   ├── sidecars.py              # Sidecar file discovery: pending_audio_to_sidecar, pending_sidecar_chain
│   │   ├── runtime_env.py           # Shell env export from runtime config block
│   │   └── datasets/
│   │       ├── preprocess.py        # CrestFactor/Loudness/Diarization datasets + DataLoaders
│   │       ├── separation.py        # DistillMOS dataset with length-sorted batching
│   │       └── transcription.py     # Transcription dataset + batch recognition
│   │
│   ├── download/
│   │   ├── download.py              # Stage 0: Yandex Music downloader
│   │   └── download_prepared.py     # Variant: specific episode downloader
│   │
│   ├── preprocess/
│   │   ├── preprocess.py            # Stage 1: Sortformer diarization + Smart Turn + chunk export
│   │   ├── sortformer_onnx.py       # ONNX Sortformer model wrapper
│   │   ├── crest_factor_remover.py  # Stage 2: Crest factor filter
│   │   └── preprocess_audio.py      # Stage 3: ITU-R BS.1770-4 loudness normalization
│   │
│   ├── separation/
│   │   ├── music_detect.py          # Stage 4: WavLM music detection filter
│   │   └── distillmos_process.py    # Stage 5: DistillMOS scoring
│   │
│   ├── transcription/
│   │   ├── transcription.py         # Stage 6: Multi-model ASR via onnx-asr
│   │   └── rover.py                 # ROVER aggregation (crowd-kit)
│   │
│   ├── punctuation/
│   │   └── punctuation.py           # Stage 7: RUPunct punctuation restoration
│   │
│   ├── accents/
│   │   └── accents.py               # Stage 8: ruAccent stress marks
│   │
│   ├── phonemizer/
│   │   └── phonemizer.py            # Stage 9: TryIParu G2P
│   │
│   └── libs/
│       └── smart_turn/
│           ├── offline_svad.py      # SmartVAD ONNX EOS classifier
│           └── inference.py         # Standalone Smart Turn inference (legacy)
│
├── configs/
│   └── config.yaml                  # Central configuration
├── logs/                            # Rotating stage logs (gitignored)
├── cache/                           # TensorRT / transient caches (gitignored)
├── models/                          # Model weights (gitignored)
├── docs/
│   ├── guide.md                     # Full usage guide
│   ├── preparing.md                 # Dataset preparation guide
│   └── dev.md                       # Developer guide for adding stages
├── example/
│   ├── README.md                    # WebDataset loading instructions
│   └── example.py                   # Python script for loading WebDataset shards
├── benchmarking/                    # Benchmark harness
├── create_dev_env.sh                # Creates venv + installs deps
├── requirements_dev.txt             # Python dependency list
├── .env                             # HF_TOKEN, YANDEX_KEY
└── README.md                        # Top-level project documentation
```
