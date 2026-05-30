# Usage Guide

End-to-end walkthrough of the Balalaika pipeline: per-stage outputs, the
configuration knobs that matter most, the logging layout, and how to read the
final filter report.

---

## Table of Contents

1. [Pipeline stages](#pipeline-stages)
2. [Running the pipeline](#running-the-pipeline)
3. [Audio quality policy](#audio-quality-policy)
4. [Logging](#logging)
5. [Filter audit & final report](#filter-audit--final-report)
6. [Output files](#output-files)
7. [Configuration](#configuration)
8. [Running individual stages](#running-individual-stages)
9. [Troubleshooting](#troubleshooting)

---

## Pipeline stages

### 1. Download (`src/download/`)
Downloads podcast episodes from Yandex Music based on provided URLs or
playlists. Skip this stage if you bring your own corpus.

**Input**: Podcast URLs or playlist IDs
**Output**: Raw audio files (`.mp3`) organized by `{album_id}/{episode_id}/`
**Configuration**: `config.yaml` → `download` section

---

### 2. Preprocess (`src/preprocess/`)

Three sequential steps that build chunked segments + initial metadata:

#### 2.1. Sortformer chunking (`preprocess.py`)
Streams **Sortformer (ONNX)** diarization in 15-minute windows, picks
**single-speaker** segments, refines turn boundaries with **Smart Turn**, and
exports chunks with names like `{start}_{end}_{album}_{episode}.{ext}`. Source
files are deleted **after** their chunks land on disk.

**Quality preserving by default**: chunk extension follows the source. FLAC
input → FLAC chunks; WAV → WAV; lossy formats keep their container so the
pipeline doesn't insert an extra encode pass. Override with
`preprocess.chunk_format` (`auto` / `flac` / `wav` / `mp3` / `ogg` / `opus`).

**Output**:
- Chunked audio files
- `balalaika.csv` rows with `filepath`, `speaker_id`, `start`, `end`,
  `total_duration`, `playlist_id`, `podcast_id`, `silence_percent`,
  `max_silence_duration`, `is_single_speaker`
- Audit row in `filter_summary.csv` (stage `preprocess`)

#### 2.2. Crest factor filter (`crest_factor_remover.py`)
Computes `crest_factor = peak / RMS` per file, writes the value to
`balalaika.csv`, deletes files above `preprocess.crest_treshold` and emits a
`crest_factor` row in `filter_summary.csv` with hours kept vs. removed.

#### 2.3. Loudness normalization (`preprocess_audio.py`)
ITU-R BS.1770-4 normalization (peak + integrated LUFS). Files are overwritten
in place with `torchaudio.save`, and `loudness_normalized` is written to
`balalaika.csv` so interrupted runs can resume.

**Configuration**: `config.yaml` → `preprocess` section
- `crest_treshold`, `peak`, `loudness`, `block_size`
- `duration`, `chunk_duration`
- `chunk_format` (default `auto`)
- `sortformer_model`, `use_tensorrt`, `vad_args.*`

---

### 3. Separation (`src/separation/`)

Quality filtering on chunked clips. Diarization itself is handled in the
preprocess stage; separation covers music detection, DistillMOS scoring/filtering,
and Spectra-0 anti-spoofing.

#### 3.1. Music detection (`music_detect.py`)
Fine-tuned **WavLM** classifier. Writes `music_prob` to `balalaika.csv`,
deletes clips above `separation.music_detect.threshold` and emits a
`music_detect` row in `filter_summary.csv`.

#### 3.2. DistillMOS scoring (`distillmos_process.py`)
Predicts MOS for every surviving clip and writes `DistillMOS` to
`balalaika.csv`. No deletion — purely an annotation stage.

#### 3.3. DistillMOS filter (`distillmos_filter.py`)
Reads the `DistillMOS` column, prints distribution statistics, and deletes
clips below `separation.distillmos_filter.threshold`. If the threshold is
`null` or the stage is run with `--manual`, it asks for a threshold
interactively. Deletions are streamed through partial CSV files and recorded in
`filter_summary.csv` as `distillmos_filter`.

#### 3.4. Anti-spoofing (`antispoofing.py`)
Runs the **[Spectra-0](https://huggingface.co/lab260/spectra_0)** ONNX classifier on fixed 16 kHz / 64,600-sample batches.
Audio preparation follows the official repo after decoding with
`torchaudio.load_with_torchcodec`: mono mixdown, 16 kHz resampling,
preemphasis, then random crop for long clips or repeat for short clips. The
stage writes `antispoof_score` and
`antispoof_generated_prob` to `balalaika.csv`, deletes clips above
`separation.antispoofing.threshold`, and records deletions in
`filter_summary.csv` as `antispoofing`.

**Configuration**: `config.yaml` → `separation` section
- `music_detect.bs`, `num_workers`, `music_detect_model`, `threshold`, optional
  `base_model` / `cache_path`
- `distillmos.batch_size`, `distillmos.num_workers`, `distillmos.prefetch_factor`
- `distillmos_filter.threshold`, `distillmos_filter.num_workers`
- `antispoofing.onnx_path`, `threshold`, `batch_size`, `num_workers`,
  `use_tensorrt`

---

### 4. Transcription (`src/transcription/`)
Multi-model ASR via **[onnx-asr](https://github.com/istupakov/onnx-asr)** with
optional **TensorRT**.

**Key features**:
- **Consensus skip**: if `consensus_num` earlier models agree on the
  normalized transcript, later models are skipped for that clip.
- **Multi-GPU** via `multiprocessing`.
- **Word-level timestamps** (`.tst` TSV) for models in `SUPPORTED_TIMESTAMPS`.
- **ROVER** consensus → `{stem}_rover.txt` when `use_rover: True`.

**Output**:
- `{stem}_{model}.txt` per model, `{stem}_{model}.tst` for timestamp-capable
  models, `{stem}_rover.txt` consensus.

**Configuration**: `config.yaml` → `transcription` section
- `model_names`, `consensus_num`, `with_timestamps`, `use_tensorrt`, `use_vad`,
  `use_rover`, `batch_size`

---

### 5. Punctuation (`src/punctuation/`)
**RUPunct** restores punctuation and casing from `{stem}_rover.txt`.

**Output**: `{stem}_punct.txt`
**Config**: `config.yaml` → `punctuation` (`model_name`, `num_workers`)

---

### 6. Accents (`src/accents/`)
**ruAccent** annotates lexical stress on `{stem}_punct.txt`.

**Output**: `{stem}_accent.txt`
**Config**: `config.yaml` → `accent` (note: section name is `accent`, not
`accents`). Keys: `model_name`, `num_workers`, `use_tensorrt`.

---

### 7. Phonemizer (`src/phonemizer/`)
**TryIParu** grapheme-to-IPA on `{stem}_rover.txt`.

**Output**: `{stem}_rover_phonemes.txt`
**Config**: `config.yaml` → `phonemizer` (`num_workers`).

---

### 8. Denoising / speech enhancement (`src/denoising/`)

The denoising stage runs a dynamic ONNX export of
**[MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)**,
the 48 kHz speech-enhancement model from ClearerVoice-Studio. The original
ClearVoice API exposes it as
`ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])`;
Balalaika uses ONNX Runtime so the same stage can run with CUDA or TensorRT EP.

The stage decodes audio with `torchaudio`, converts it to mono 48 kHz batches,
runs ONNX inference, trims the output to the source length, and overwrites the
audio file in place.

**Output**: same audio paths rewritten at 48 kHz, plus `denoised=True` in
`balalaika.csv`.

**Config**: `config.yaml` → `denoising` (`podcasts_path`, `onnx_path`,
`hf_repo_id`, `hf_filename`, `processes`, `batch_size`, `num_workers`,
`prefetch_factor`, `use_tensorrt`).

---

### 9. Collate / export (`src/collate.py` + `src/to_webdataset.py`)

`collate.py` aggregates the text sidecars into `balalaika.parquet`; it reads
`podcasts_path` and `num_workers` from the **`download`** section of the
config (keep it aligned with the dataset root).

`to_webdataset.py` writes WebDataset shards. Audio bytes are written as-is so
the chunked container is preserved end-to-end (no extra encode at export).

**Run**:

```bash
bash base.sh --config_path configs/config.yaml --stage 11 --stop_stage 11
bash base.sh --config_path configs/config.yaml --stage 12 --stop_stage 12
```

---

### 10. Filter report (`src/report.py`)

After filtering stages finish, `src.report` reads `filter_summary.csv` and
writes `<podcasts_path>/filter_report.md` summarising hours filtered at every
stage. It is stage 13 in `base.sh`; it runs only when your selected
`--stage`/`--stop_stage` range includes 13.

---

## Running the pipeline

`base.sh` is a Kaldi-style orchestrator with numbered stages
(`--stage` / `--stop_stage` like CosyVoice's `run.sh`). With no stage flags it
runs stages 1..9: chunking through phonemization, skipping download, parquet
collation, WebDataset export, and the final report.

```bash
bash base.sh --config_path configs/config.yaml
```

Run all local processing stages, including denoising, parquet, WebDataset export, and the
report:

```bash
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 13
```

Include Yandex Music download as well:

```bash
bash base.sh --config_path configs/config.yaml --stage 0 --stop_stage 13
```

Run a contiguous subrange (e.g. preprocess only):

```bash
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3
```

Run a single stage (e.g. transcription) after data is already chunked:

```bash
bash base.sh --config_path configs/config.yaml --stage 6 --stop_stage 6
```

Stage map:

| ID | Stage | Module |
|----|-------|--------|
| 0 | Download | `src.download.download` |
| 1 | Preprocess: chunking | `src.preprocess.preprocess` |
| 2 | Preprocess: crest factor | `src.preprocess.crest_factor_remover` |
| 3 | Preprocess: loudness | `src.preprocess.preprocess_audio` |
| 4 | Separation: music detection | `src.separation.music_detect` |
| 5 | Separation: DistillMOS | `src.separation.distillmos_process` |
| 5.5 | Separation: DistillMOS filter | `src.separation.distillmos_filter` |
| 5.6 | Separation: anti-spoofing | `src.separation.antispoofing` |
| 6 | Transcription | `src.transcription.transcription` |
| 7 | Punctuation | `src.punctuation.punctuation` |
| 8 | Accents | `src.accents.accents` |
| 9 | Phonemizer | `src.phonemizer.phonemizer` |
| 10 | Denoising / enhancement | `src.denoising.denoising` |
| 11 | Collate (parquet) | `src.collate` |
| 12 | Export (WebDataset) | `src.to_webdataset` |
| 13 | Filter report | `src.report` |

`base.sh` reads runtime parameters (venv path, CPU affinity, log dir, TRT cache
and workspace) from the **`runtime`** block in the YAML via
`src.utils.runtime_env`. Add `--strict` if you want `base.sh` to stop when a
stage writes a `stage_<id>_status.json` file with non-zero errors.

---

## Audio quality policy

- Chunked segments preserve the source container by default (FLAC stays FLAC).
  Use `preprocess.chunk_format` to pin a specific extension when you need it.
- Loudness normalization overwrites clips in place with `torchaudio.save` after
  peak and integrated-loudness normalization.
- Denoising intentionally rewrites audio in place using the ONNX export of
  ClearerVoice-Studio `MossFormer2_SE_48K`; the model sample rate is 48 kHz mono.
- Read-only stages (crest filter, music detection, DistillMOS, ASR, RUPunct,
  ruAccent, TryIParu, WebDataset export) never re-encode the audio. WebDataset
  export copies the current audio bytes verbatim.

---

## Logging

Every script in `src/` calls `setup_logging(stage_name)` at startup, which:

- Removes loguru's default sinks.
- Adds a colored stderr sink.
- Adds a rotating file sink (`200 MB`, last 10 retained).

The log directory resolution order is:

1. `--log_dir <path>` CLI flag (every stage accepts it).
2. `BALALAIKA_LOG_DIR` environment variable.
3. `./logs` relative to the working directory (the default).

A run produces files like `./logs/preprocess_20260425-150301.log`,
`./logs/music_detect_20260425-152114.log`, …

Quick recipes:

```bash
# Tail the active stage live
less +F logs/preprocess_*.log

# Find errors across the whole run
rg "ERROR|WARNING" logs/

# Send all logs to a custom directory by setting runtime.log_dir in configs/config.yaml
bash base.sh --config_path configs/config.yaml
```

---

## Filter audit & final report

Each filtering stage appends a single row to
`<podcasts_path>/filter_summary.csv` via
`src.utils.audit.record_stage_summary`. Schema:

| Column | Meaning |
|--------|---------|
| `timestamp` | UTC ISO-8601 (when the stage finished) |
| `stage` | `preprocess` / `crest_factor` / `music_detect` / `distillmos_filter` / `antispoofing` |
| `files_in` / `files_out` | File counts before vs. after filtering |
| `hours_in` / `hours_out` | Total audio hours before vs. after filtering |
| `hours_removed` | Convenience: `max(0, hours_in - hours_out)` |
| `params` | JSON blob with stage-specific knobs (threshold, etc.) |

`src/report.py` reads the CSV and emits `filter_report.md` with:

- A per-stage table (latest run only) — files, hours, % removed, params.
- A **pipeline net effect** line: total hours in vs. total hours out.
- A full-history table covering every run ever recorded.

Manual invocation (without re-running the pipeline):

```bash
python -m src.report --config_path configs/config.yaml
```

---

## Output files

For each audio segment, the pipeline generates:

```
{start}_{end}_{album_id}_{episode_id}.{ext}             # Audio chunk
{start}_{end}_{album_id}_{episode_id}_{model}.txt       # Per-model ASR
{start}_{end}_{album_id}_{episode_id}_{model}.tst       # Timestamps (when supported)
{start}_{end}_{album_id}_{episode_id}_rover.txt         # ROVER consensus
{start}_{end}_{album_id}_{episode_id}_punct.txt         # With punctuation
{start}_{end}_{album_id}_{episode_id}_accent.txt        # With accents
{start}_{end}_{album_id}_{episode_id}_rover_phonemes.txt # Phonemes
```

Dataset-level files at `podcasts_path`:

| File | Created by | Purpose |
|------|------------|---------|
| `balalaika.csv` | preprocess + crest + loudness + music/music-prob + DistillMOS + anti-spoofing + denoising | Per-clip metadata |
| `filter_summary.csv` | filtering stages | Audit log of files/hours dropped |
| `filter_report.md` | `src/report.py` | Human-readable report |
| `balalaika.parquet` | `src/collate.py` | Final aggregated metadata |

---

## Configuration

The main configuration file is `configs/config.yaml`. Key sections:

### Global parameters
- `cache_path`: Reserved/general cache path; stage code normally reads its own
  section-specific settings.
- `podcasts_path`: Dataset root set inside each section that processes data.
  Keep the values aligned unless you intentionally use separate trees.

### Runtime block

The new `runtime:` block centralises orchestration knobs that used to be
hardcoded in the shell scripts. Edit it instead of patching `base.sh`:

```yaml
runtime:
  venv_path: .dev_venv          # virtualenv activated by base.sh
  cpu_affinity: "0-24"          # taskset -c argument; empty disables pinning
  log_dir: ./logs               # directory for rotating per-stage logs
  audio_paths_source: auto      # auto/csv/rglob source for stage file lists
  trt_cache_path: ./cache/trt   # TensorRT engine cache root
  trt_workspace_bytes: 4294967296   # 4 GiB per session
  trt_fp16: True                # FP16 for TensorRT EP
```

These values are exported as `BALALAIKA_*` env vars by
`src.utils.runtime_env` and read by Python modules that need them
(`get_onnx_providers`, `setup_logging`, ...). No shell-side YAML parsing
required.

### Stage-specific configuration

Each stage has its own block. The file ships with a comment header at the top
that documents every section. Dataset and model paths should be absolute paths
for production runs; repo-local model paths such as `./models/...` are resolved
relative to where you launch the command.

---

## Running individual stages

### From `base.sh`

Use `--stage` / `--stop_stage` (see the table above). Both arguments accept
the same numeric IDs and are inclusive on both ends. Example: only run music
detection and DistillMOS:

```bash
bash base.sh --config_path configs/config.yaml --stage 4 --stop_stage 5
```

### Run scripts directly

```bash
# Activate the dev environment
source .dev_venv/bin/activate

# Run individual stages (each accepts an optional --log_dir)
python -m src.preprocess.preprocess           --config_path configs/config.yaml
python -m src.preprocess.crest_factor_remover --config_path configs/config.yaml
python -m src.preprocess.preprocess_audio     --config_path configs/config.yaml
python -m src.separation.music_detect         --config_path configs/config.yaml
python -m src.separation.distillmos_process   --config_path configs/config.yaml
python -m src.separation.distillmos_filter    --config_path configs/config.yaml
python -m src.separation.antispoofing         --config_path configs/config.yaml
python -m src.transcription.transcription     --config_path configs/config.yaml
python -m src.punctuation.punctuation         --config_path configs/config.yaml
python -m src.accents.accents                 --config_path configs/config.yaml
python -m src.phonemizer.phonemizer           --config_path configs/config.yaml
python -m src.denoising.denoising             --config_path configs/config.yaml
python -m src.collate                         --config_path configs/config.yaml
python -m src.to_webdataset                   --config_path configs/config.yaml
python -m src.report                          --config_path configs/config.yaml
```

### Processing order

1. **Download** → raw audio
2. **Preprocess** → chunking → crest filter → loudness normalization
3. **Separation** → music detection → DistillMOS scoring/filter → anti-spoofing
4. **Transcription** → per-model ASR + ROVER
5. **Punctuation** → `_rover.txt` → `_punct.txt`
6. **Accents** → `_punct.txt` → `_accent.txt`
7. **Phonemizer** → `_rover.txt` → `_rover_phonemes.txt`
8. **Denoising** → in-place 48 kHz enhanced audio
9. **Collate / export** → `balalaika.parquet`, WebDataset shards
10. **Report** → `filter_report.md`

---

## Troubleshooting

### Common issues

1. **`balalaika.csv` mentions paths that no longer exist** — rerun the
   relevant filter stage. The audit utilities prune missing rows on every CSV
   update.
2. **`filter_report.md` is empty / placeholder** — at least one filter stage
   must have completed and written to `filter_summary.csv`. Re-run a stage or
   inspect logs in `./logs/`.
3. **Chunks land as `.mp3` even though input is `.flac`** — set
   `preprocess.chunk_format: auto` (the default) or pin it to `flac`.
4. **`tensorrt` provider unavailable** — set `use_tensorrt: False` in the
   relevant stage; CUDA execution provider is used automatically.

For per-module specifics see the `src/*/README.md` files.
