# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Balalaika is an end-to-end **speech-data pipeline** for building large Russian
speech datasets. It ingests long-form audio (or pre-chunked audio), diarizes and
chunks it, filters low-quality / music / spoofed material, runs multi-model ASR
with ROVER consensus, restores punctuation and stress marks, phonemizes, denoises,
and exports to Parquet / WebDataset. Accompanies arXiv:2507.13563.

The unit of work is a single machine processing a dataset tree of millions of
short audio chunks. Design priorities, in order, are: **resumability** (any stage
can be killed and re-run), **reproducibility** (most "fast path" optimizations are
gated behind flags and pinned bit-identical by tests), and **throughput on a
shared GPU box**.

## Commands

```bash
# One-time environment setup (uv venv .dev_venv, python 3.12, ORT-GPU + TensorRT 10)
bash create_dev_env.sh

# Run the pipeline (default = stages 12..15). base.sh activates the venv,
# exports BALALAIKA_* env from configs/config.yaml, then runs each stage.
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 15
bash base.sh --config_path configs/config.yaml --stage 8 --stop_stage 8   # one stage
bash base.sh --config_path configs/config.yaml --stage 4                  # resume from 4
bash base.sh --help                                                       # stage map
bash base.sh ... --strict   # abort if any stage's status JSON reports errors > 0

# Run a single module directly (preserves venv + CPU affinity + log dir):
bash src/transcription/transcription_yaml.sh configs/config.yaml

# Per-node batch-size autotuning -> cache/node_profile.json (lets config use batch_size: auto)
python -m benchmarking.warmup --config_path configs/config.yaml

# Tests (pytest, no config file — defaults apply). Most are CPU-only and fast.
.dev_venv/bin/python -m pytest tests/ -q
.dev_venv/bin/python -m pytest tests/test_fast_rover.py -q          # single file
.dev_venv/bin/python -m pytest tests/test_fast_rover.py::test_x -q  # single test

# Syntax check after editing a module (used in CI-less workflow):
.dev_venv/bin/python -m py_compile src/path/to/module.py

# Format / lint. black is the style of record (no config file → its default
# 88-col line length). There is NO flake8/black config in the repo, so flake8's
# default 79-col E501 fires across existing code and is NOT enforced — match
# black, and use flake8 only for real defects (F401 unused, F821 undefined),
# not line length.
.dev_venv/bin/python -m black src/ tests/
.dev_venv/bin/python -m flake8 --extend-ignore=E501,E203,W503 src/
```

Note: scripts must run via the `.dev_venv` python — `base.sh`/`stage_runner.sh`
also prepend the venv's bundled NVIDIA + `tensorrt_libs` dirs to
`LD_LIBRARY_PATH` so ONNX Runtime's CUDA/TensorRT EPs can `dlopen` their libs.
Running modules with the system python will fail to load the GPU providers.

## Architecture

### Stage model

`base.sh` is the orchestrator. Stages are numbered (incl. fractional `5.5`, `6.5`)
and selected with `--stage` / `--stop_stage`. Each stage is an independent
`python -m src.<area>.<module>` invocation taking `--config_path` and `--log_dir`.
The stage map lives in `base.sh` header / README / `docs/dev.md` — keep all three
in sync when adding a stage. Stages communicate **only through the filesystem**
(the dataset tree, `balalaika.csv`, and per-chunk sidecar `.txt` files), never
in-process, which is what makes arbitrary `--stage` resume work.

| Stage | Module | Purpose |
|-------|--------|---------|
| 0 | `src.download.download` | Download source audio (Yandex Music) |
| 1 | `src.preprocess.preprocess` | Sortformer diarization + Smart Turn chunking |
| 2 | `src.preprocess.crest_factor_remover` | Crest-factor filter |
| 3 | `src.preprocess.preprocess_audio` | Loudness normalization (BS.1770-4) |
| 4 | `src.separation.music_detect` | Music probability scoring + filter |
| 5 | `src.separation.distillmos_process` | DistillMOS quality scoring |
| 5.5 | `src.separation.distillmos_filter` | DistillMOS threshold filter (deletes files) |
| 6 | `src.separation.antispoofing` | Spectra-0 anti-spoofing scores |
| 6.5 | `src.separation.antispoofing_filter` | Spoof-margin filter (deletes files) |
| 7 | `src.separation.tts_suitability` | TTS-suitability classifier scores |
| 7.5 | `src.separation.tts_suitability_filter` | not_tts-margin filter (deletes files) |
| 8 | `src.transcription.transcription` | ONNX-ASR multi-model + ROVER consensus |
| 9 | `src.punctuation.punctuation` | Punctuation restoration (RUPunct) |
| 10 | `src.accents.accents` | Accent/stress mark restoration |
| 11 | `src.phonemizer.phonemizer` | G2P phonemization |
| 12 | `src.denoising.denoising` | MossFormer2_SE_48K speech enhancement |
| 13 | `src.collate` | Merge sidecars → `balalaika.parquet` |
| 14 | `src.to_webdataset` | Export WebDataset tar shards |
| 15 | `src.report` | Build `filter_report.md` |

Stages 12–14 use top-level modules (`src/collate.py`, `src/to_webdataset.py`,
`src/report.py`) rather than the `src/<area>/` subdirectory pattern used by
earlier stages. `src/preprocess/preprocess_existing_chunks.py` is an alternate
stage-1 path for datasets that arrive pre-chunked (skips diarization, still runs
Sortformer for metadata).

### Config: one section per stage

`configs/config.yaml` has one top-level key per stage area (`runtime`, `csv`,
`download`, `preprocess`, `separation`, `transcription`, `punctuation`, `accent`
[note: key is `accent`, not `accents`], `phonemizer`, `denoising`, `export`).
Each stage calls `load_config(config_path, "<section>")` (`src/utils/utils.py`)
and sees **only its own section**. `podcasts_path` is repeated in every section
and must point at the same dataset root across stages unless you deliberately
diverge. The `config.yaml` file itself carries extensive inline docs for every
key — read it before changing behavior; many comments record measured tradeoffs
and bit-exactness caveats.

`runtime:` is special: `base.sh` reads it via `python -m src.utils.runtime_env`
and `eval`s the result into `BALALAIKA_*` shell vars (venv path, log dir, CPU
affinity, TensorRT cache/workspace/fp16, work-shard size, thread caps). Python
modules read the same block through `runtime_cfg(config_path)` so shell and Python
stay aligned. Edit `runtime:` rather than patching the shell scripts.

### Shared infrastructure (`src/utils/`) — use these, don't reinvent

This is the most important code to understand; stages are thin glue over it.
See `src/utils/README.md` and `docs/dev.md` for the authoritative API.

- **`csv_manager.py`** — single source of truth for `<podcasts_path>/balalaika.csv`
  (per-chunk metadata + quality scores). Atomic writes (`*.tmp`+`os.replace`+fsync),
  auto-bootstrap from the audio tree, per-stage column upserts keyed on `filepath`,
  and skip-already-processed resume. **Never hand-write CSV merge logic.** Key
  helpers: `ensure_main_csv`, `unprocessed_paths`, `upsert_columns`,
  `PartialCsvWriter` (workers stream rows, flushed per-row so Ctrl+C is safe),
  `absorb_partial_csvs` (fold worker partials into main), `PeriodicCsvMerger`
  (daemon thread that folds partials into `balalaika.csv` mid-stage so a kill in a
  multi-hour run loses at most `csv.flush_every_rows` rows). `preserve_existing=True`
  (the default) means null incoming values cannot erase existing columns — pass it
  explicitly at scoring/filter call sites. The top-level `csv:` block also supports
  `state_format: parquet` (keeps live state in parquet, still exports `balalaika.csv`).

- **`work_shards.py`** — disk-backed work queues. For million-file stages, do NOT
  pass giant path lists into `mp.spawn`/`mp.Process` (pickle OOM). The parent writes
  pending paths to `<podcasts_path>/.balalaika_work/<stage>/shard_*.pending`; workers
  atomically claim shards (rename to `.running.<rank>`) and mark `.done`.
  `runtime.work_shard_size` sets paths-per-shard. `prepare_length_bucketed_work_shards`
  groups files into duration buckets (one bucket per shard) so variable-length stages
  pad less per batch; pass `annotations=` to carry a per-path string (e.g. duration)
  into each shard line, read back with `read_annotated_work_shard`.

- **`parallel.py`** — GPU scheduling. `run_per_gpu_processes` (one big model per GPU,
  one `mp.Process` per GPU) and `run_per_gpu_pool` (one `ProcessPoolExecutor` per GPU
  for small models). Both handle `KeyboardInterrupt` cleanly. Check these before
  writing new GPU orchestration.

- **`gpu.py`** — `get_onnx_providers(cuda_id, use_tensorrt, config_path)` builds the
  ORT provider list with a per-GPU TensorRT engine cache shared project-wide;
  `make_session_options(config_path)` (graph opt + optional `runtime.threads_per_worker`
  caps); `apply_torch_perf_defaults()` (TF32 + flash SDP). The ONNX-inference stages
  (`antispoofing`, `music_detect`, `denoising`) share one pattern: a per-stage
  `use_tensorrt` flag, an `ensure_model()` that `hf_hub_download`s the `.onnx` on first
  use (the model ships via HF, not git), then **patch the provider opts** returned by
  `get_onnx_providers` for stage specifics — e.g. pin a TensorRT dynamic-shape profile
  (`trt_profile_{min,opt,max}_shapes`) so one engine covers the batch/length range, and
  override `trt_fp16_enable` when a model is fp16-unsafe even though `runtime.trt_fp16`
  is on globally. These stages use a **fixed `batch_size`** and rely on upstream
  chunking for bounded memory; raw long files can OOM the variable-length models.

- **`sidecars.py`** — pairing helpers for text-pipeline stages (`pending_audio_to_sidecar`,
  `pending_sidecar_chain`, `replace_in_stem`). Sidecar naming convention:
  `<chunk>_rover.txt` → `_punct.txt` → `_accent.txt`, plus `_rover_phonemes.txt`.

- **`audit.py`** + **`stage_status.py`** — filter stages append `{files_in/out,
  hours_in/out}` rows to `<podcasts_path>/filter_summary.csv` (consumed by stage 14
  `src.report` → `filter_report.md`); every stage writes
  `<log_dir>/stage_<id>_status.json` for `--strict`.

- **`logging_setup.py`** — `setup_logging("<stage>", log_dir=...)`; colored stderr +
  rotating file at `<log_dir>/<stage>_<timestamp>.log`.

- **`io_profile.py`** — detects whether `podcasts_path` is on HDD or SSD (via
  sysfs rotational flag, overridable via `runtime.io_profile` or
  `$BALALAIKA_IO_PROFILE`). Stages clamp DataLoader worker counts on HDD because
  multiple concurrent readers multiply seek distance rather than throughput.

- **`node_profile.py`** — resolves `batch_size: auto` in config against
  `cache/node_profile.json` (generated by `benchmarking/warmup.py`). Stages call
  `resolve_batch_size(config.get("batch_size", "auto"), ...)` so integer configs
  pass through unchanged.

- **`audio_durations.py`** — shared duration cache backed by `balalaika.csv`'s
  `total_duration` column. Used by length-bucketed shard preparation; probes missing
  entries once and writes them back so downstream stages reuse the cache.

### Bundled external code

`src/libs/smart_turn/` is a vendored copy of the Smart Turn library used in stage 1
for inter-sentence boundary refinement. Treat it as a dependency, not project code —
do not refactor it or apply project conventions to it.

### Dataset/DataLoader separation

Strict layering enforced by `docs/dev.md`: **Dataset/DataLoader code lives in
`src/utils/datasets/<area>.py`** (one file per pipeline area), stage orchestration
lives in `src/<area>/<module>.py`, and runtime knobs live in `config.yaml`. Stages
receive prepared batches and focus on model execution + result writing — do not
decode audio ad hoc inside the inference loop. DataLoader `__getitem__` should
**return load errors as data** (e.g. `(path, empty_tensor, 0, err_str)`) rather
than crashing workers.

### Audio stack convention

`torch` + `torchaudio` (with torchcodec) for all audio IO, resampling, STFT.
In the current `torch 2.11+cu13` build even `torchaudio.load` dispatches to
**torchcodec**, so audio decoding `dlopen`s the bundled CUDA libs (it failed here
needing `libnvrtc.so.13`) exactly like the ORT GPU providers — run stages via the
`*_yaml.sh` wrappers (which set `LD_LIBRARY_PATH`), not bare python, or decoding
itself fails. **Do not introduce `librosa`** for new code unless there is an explicit
project decision; `soundfile` is already used for the documented exceptions
(loudness/lossless writes, the antispoofing ranged-decode fast path).
GPU models are initialized **inside** the per-GPU worker process, never in the
parent before `spawn`.

### "Fast path" pattern

Several stages ship an optimized reimplementation gated behind a config flag with a
stock fallback: `use_fast_rover` (numba ROVER), `use_fast_rnnt` (batched RNN-T
greedy decode), `use_fast_accent`, `fast_g2p`, `share_decode`, `persistent_loaders`,
`sortformer_io_binding`. Each is pinned **bit/char-identical to the stock path by a
dedicated test** (`tests/test_fast_*.py`) and the stock path is the fallback. When
touching one of these, preserve the equivalence and re-run its test. Flags that are
*not* bit-exact (fp16/TensorRT, `smart_vad_batch_size>1`, `shard_order: path`)
document the divergence in `config.yaml` comments — respect those notes.

## Conventions when adding a stage

The canonical procedure is in `docs/dev.md` ("Adding A New Stage") — follow it.
In short: new module under the right package; config subsection; Dataset in
`src/utils/datasets/`; `setup_logging` + `load_config`; `csv_manager` for state;
one process per GPU; `write_stage_status`; wire into `base.sh` + a `*_yaml.sh`
wrapper; `py_compile` then run on a tiny dir before the full dataset. Multiprocessing
entrypoints set `mp.set_start_method("spawn", force=True)`.

## Reference docs

- `docs/dev.md` — authoritative developer guide (stage shape, GPU patterns, CSV/resume, pitfalls).
- `docs/guide.md`, `docs/preparing.md` — usage and dataset-layout guides.
- `src/*/README.md` and `src/utils/README.md` — per-module notes.
- `report.md` — benchmark measurements behind the optimization flags and defaults.
- `.env` (repo root, gitignored) — needs `HF_TOKEN` and `YANDEX_KEY`.
