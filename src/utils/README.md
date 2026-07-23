## Shared utilities

Cross-cutting helpers used by every pipeline stage. Stages import from this
folder instead of duplicating boilerplate.

| Module | What it provides |
|--------|------------------|
| `csv_manager.py` | Single source of truth for the `balalaika.parquet` state (bootstrap, atomic writes, partial-CSV streaming, resume, filter-stage audit). CSV state was removed — parquet only. |
| `chunk_json.py` | One `<stem>.json` per chunk for stage 8–11 text outputs: `update_chunk_json` (atomic deep-merge), `read_chunk_json`, `field_complete`/`pending_chunks` (resume by field), `ChunkJsonCache`. |
| `gpu.py` | `apply_torch_perf_defaults`, `get_onnx_providers`, `gpu_count`. |
| `parallel.py` | `run_per_gpu_pool` (one `ProcessPoolExecutor` per GPU) and `run_per_gpu_processes` (one `mp.Process` per GPU). |
| `work_shards.py` | Disk-backed work queues for huge multiprocessing stages; workers atomically claim shard files instead of receiving giant path lists. |
| `sidecars.py` | Legacy per-file sidecar helpers (`pending`, `DirNameCache`, `replace_in_stem`, …). Superseded by `chunk_json.py` for the stages; retained for `DirNameCache`/NAME_MAX semantics and tests. |
| `audit.py` | `record_stage_summary`, `safe_audio_duration`, `total_hours` — appends a row to `<podcasts_path>/filter_summary.csv` for the final report. |
| `runtime_env.py` | Reads the `runtime:` block of `configs/config.yaml`; powers `eval "$(python3 -m src.utils.runtime_env --config_path …)"` in `base.sh`. |
| `logging_setup.py` | `setup_logging(stage, log_dir=…)` — colored stderr + rotating file sink. |
| `utils.py` | Misc helpers: `load_config`, `get_audio_paths`, `get_txt_paths`, `read_file_content`, `process_token`, `normalize_text`, `load_audio`. |

## `csv_manager.py` — `balalaika.parquet` lifecycle

All filter / scoring stages collaborate around one parquet state at
`<podcasts_path>/balalaika.parquet` (CSV state was removed — parquet only).
Each stage writes one or more columns (`crest_factor`, `loudness_normalized`,
`music_prob`, `DistillMOS`, …) and filter stages may delete rows whose audio was
removed. (Function/class names keep the historical `csv`/`Csv` token; the
per-worker `*_part_*.csv` partials are transient and folded into the parquet.)

### Guarantees

* **Atomic writes.** `atomic_write_csv` writes via `*.tmp` + `os.replace` +
  `fsync`; a kill mid-write never leaves a half-written CSV. A stale `.tmp`
  is recovered transparently on the next read.
* **Auto-bootstrap.** `ensure_main_csv(podcasts_path, audio_paths=…)`
  creates the CSV from the audio tree if it doesn't exist, so any stage can
  run as the *first* CSV-touching stage.
* **Per-stage column upserts.** `upsert_columns(podcasts_path, results_df,
  value_columns, drop_missing_files=…)` merges new rows on `filepath`,
  preserves existing columns, optionally prunes rows whose audio is gone.
* **Skip-already-processed.** `unprocessed_paths(podcasts_path, column,
  audio_paths)` returns only files that don't yet have a value in `column`,
  regardless of how the previous run was killed.

### Worker-side incremental output

```python
from src.utils.csv_manager import PartialCsvWriter, absorb_partial_csvs

# Worker (called inside mp.spawn / mp.Process):
with PartialCsvWriter(podcasts_path, "crest", rank,
                      fieldnames=("filepath", "crest_factor", "duration_s", "deleted")) as writer:
    already = writer.already_done()      # resume — skip rows already on disk
    for f in shard:
        if resolve_path(f) in already:
            continue
        writer.write({"filepath": resolve_path(f), "crest_factor": ..., ...})
        # PartialCsvWriter.flush()s after every row, so Ctrl+C is safe.

# Main process (after workers join, even on KeyboardInterrupt):
partials, n = absorb_partial_csvs(
    podcasts_path, "crest",
    value_columns=["crest_factor"],
    drop_missing_files=True,
)
```

The pattern guarantees that a forced stop preserves whatever rows the workers
already produced; on the next run those rows are folded into
`balalaika.csv` before new work is scheduled.

### Disk-backed work shards

For very large datasets, stages should not pass `list[str]` with millions of
paths into `mp.spawn` or `mp.Process`. `src.utils.work_shards` writes pending
paths to `<podcasts_path>/.balalaika_work/<stage>/shard_*.pending`; workers
atomically claim shards by renaming them to `.running.<rank>` and mark them
`.done` after processing. `runtime.work_shard_size` controls the number of
paths per shard.

### Live `balalaika.csv` during long stages — `PeriodicCsvMerger`

Worker partials are flushed row-by-row, but on their own they only become
visible in `balalaika.csv` at the *end* of the stage. For multi-hour runs
that's not enough — a Ctrl+C / SIGKILL would leave the main CSV stale until
the next start-up merge. `PeriodicCsvMerger` fixes that with a deliberately
minimal design:

```python
from src.utils.csv_manager import PeriodicCsvMerger, load_csv_settings

csv_settings = load_csv_settings(args.config_path)  # reads the top-level csv:

with PeriodicCsvMerger(
    podcasts_path,
    prefix="distillmos",
    value_columns=["DistillMOS"],
    drop_missing_files=False,
    **csv_settings,
):
    mp.spawn(run_worker, args=(...,), nprocs=available_gpus, join=True)
```

What it does:

* Runs one daemon thread in the main process.
* Every `poll_interval` seconds (default 30s) counts data rows on disk
  across all `<prefix>_part_*.csv` using a cheap byte-level newline count —
  no pandas, no in-memory mirror, no buffering.
* When the count has grown by `flush_every_rows` since the last flush
  (or `flush_every_seconds` elapsed), calls the existing on-disk
  `upsert_columns` exactly once. That's a single straightforward merge.
* Never deletes partials — the post-stage `absorb_partial_csvs` still owns
  cleanup so the merger crashing mid-flush cannot lose data.

Both `flush_every_rows` and `flush_every_seconds` come from the top-level
`csv:` block of `configs/config.yaml`.

### Filter-stage audit

`audit_from_filter_partials(partials_df)` produces the
`{files_in, files_out, hours_in, hours_out, files_deleted}` dict expected by
`audit.record_stage_summary`. Used by `crest_factor_remover` and
`music_detect`.

## `gpu.py`

```python
from src.utils.gpu import apply_torch_perf_defaults, get_onnx_providers, gpu_count

apply_torch_perf_defaults()                    # TF32 + Flash/Mem-eff SDP

providers = get_onnx_providers(
    cuda_id, use_tensorrt=True, config_path="configs/config.yaml"
)  # TensorrtEP first, sharing trt_cache_<cuda_id> with the rest of the pipeline
```

`gpu_count()` is a `torch.cuda.device_count()` wrapper that returns 0 on
hosts without `torch`.

## `parallel.py`

Two orchestrators replace the per-stage GPU scaffolding:

```python
from src.utils.parallel import run_per_gpu_pool, run_per_gpu_processes

# A) one ProcessPoolExecutor per GPU (small models, many workers per GPU)
run_per_gpu_pool(
    pending_files,
    work_fn=process_file,
    initializer=init_process,
    init_args_factory=lambda gpu_id: (model_name, gpu_id, use_tensorrt, config_path),
    num_workers_per_gpu=4,
    desc="Accents",
)

# B) one mp.Process per GPU (one big model loaded per process)
run_per_gpu_processes(
    run_worker,
    num_gpus=n_gpus,
    args=(model_name, paths, config, config_path),
)
```

Both helpers handle `KeyboardInterrupt` cleanly: pools shut down with
`cancel_futures=True`, processes are terminated and joined.

## `chunk_json.py`

Stage 8–11 text outputs live in one `<stem>.json` per chunk (keys:
`asr.<model>`, `asr_ts.<model>`, `rover`, `punct`, `accent`, `rover_phonemes`).
Each stage writes its key via an atomic deep-merge and resumes on field presence.

```python
from src.utils.chunk_json import pending_chunks, get_field, read_chunk_json, \
    chunk_json_path, update_chunk_json

# Punctuation: chunks whose JSON has `rover` but not yet `punct`.
for audio in pending_chunks(podcasts_path, out_field="punct", in_field="rover",
                            config_path=config_path):
    text = get_field(read_chunk_json(chunk_json_path(audio)), "rover")
    update_chunk_json(audio, {"punct": restore_punctuation(text)})
```

## `sidecars.py` (legacy)

Superseded by `chunk_json.py` for the stages; kept for `DirNameCache` (one
`os.scandir` per directory + NAME_MAX handling) and its tests.

## `runtime_env.py`

`base.sh` evaluates the script's stdout to import the `runtime:` block as
shell variables (`BALALAIKA_VENV`, `BALALAIKA_LOG_DIR`,
`BALALAIKA_TRT_CACHE_PATH`, `BALALAIKA_TRT_WORKSPACE`,
`BALALAIKA_TRT_FP16`, `BALALAIKA_CPU_AFFINITY`). Python modules read the
same block via `runtime_cfg(config_path)` so values stay aligned between
shell and Python.

`BALALAIKA_MALLOC_TRIM_EVERY` is emitted from the `runtime.malloc_trim_every`
key (default `128`, `0` disables) and read by
`src/utils/datasets/transcription.py` to control the per-worker heap trim. Set
it in `config.yaml` like the other runtime knobs; a shell `export` still wins
for one-off overrides since the dataset reads the env var directly. See
`src/utils/datasets/README.md`.

## `logging_setup.py`

```python
from src.utils.logging_setup import setup_logging
setup_logging("crest_factor", log_dir=args.log_dir)
```

Initialises a colored stderr sink + a rotating file sink at
`<log_dir>/<stage>_<timestamp>.log` (default rotation 200 MB, retention 10).
