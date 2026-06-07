# Balalaika Developer Guide

This document explains how to add new pipeline modules to Balalaika without
breaking the existing runtime, resume logic, GPU scheduling, or dataset loading
layout.

## Pipeline Overview

The main entrypoint is `base.sh`. It runs numbered stages from
`configs/config.yaml`:

| Stage | Module | Purpose |
| --- | --- | --- |
| 0 | `src.download.download` | Download source audio. |
| 1 | `src.preprocess.preprocess` | Sortformer diarization, Smart Turn refinement, chunk export. |
| 2 | `src.preprocess.crest_factor_remover` | Crest-factor filtering. |
| 3 | `src.preprocess.preprocess_audio` | Loudness normalization. |
| 4 | `src.separation.music_detect` | Music probability filtering. |
| 5 | `src.separation.distillmos_process` | DistillMOS quality scoring. |
| 5.5 | `src.separation.distillmos_filter` | DistillMOS threshold filtering. |
| 6 | `src.separation.antispoofing` | Store raw Spectra-0 class scores. |
| 6.5 | `src.separation.antispoofing_filter` | Filter by spoof-vs-bonafide score margin. |
| 7 | `src.transcription.transcription` | ASR with `onnx-asr` and optional ROVER. |
| 8 | `src.punctuation.punctuation` | Punctuation restoration. |
| 9 | `src.accents.accents` | Accent restoration. |
| 10 | `src.phonemizer.phonemizer` | G2P / phonemization. |
| 11 | `src.denoising.denoising` | ONNX Runtime / TensorRT denoising / speech enhancement. |
| 12 | `src.collate` | Merge sidecars into parquet. |
| 13 | `src.to_webdataset` | Export WebDataset shards. |
| 14 | `src.report` | Build filter report. |

Run a single stage:

```bash
bash base.sh --config_path configs/config.yaml --stage 5 --stop_stage 5
```

Run from a checkpoint:

```bash
bash base.sh --config_path configs/config.yaml --stage 4
```

By default, `base.sh` runs stages 11..14. Use `--stage 1 --stop_stage 14`
to run the full local pipeline from preprocessing through the final report. Use
`--strict` when the orchestrator should abort after any stage writes a status
file with non-zero errors.

Small per-stage wrappers under `src/*/*_yaml.sh` use `src/stage_runner.sh`.
They are useful when you want to run one module directly while preserving the
configured virtualenv, CPU affinity, and log directory.

## Configuration Rules

All runtime configuration should live in `configs/config.yaml`.

Use one top-level section per pipeline area:

- `runtime`: virtualenv, logs, CPU affinity, TensorRT cache paths.
- `download`: downloader settings.
- `preprocess`: diarization, VAD, chunking, crest factor, loudness.
- `separation`: music detection, DistillMOS scoring/filtering, anti-spoofing.
- `transcription`: ASR models, batching, TensorRT, VAD, ROVER.
- `punctuation`, `accent`, `phonemizer`, `export`: downstream stages.
- `denoising`: MossFormer2_SE_48K ONNX Runtime / TensorRT in-place speech enhancement.

When adding a setting:

1. Put it under the stage section that owns it.
2. Read it with `load_config(args.config_path, "<section>")`.
3. Give it an explicit default in code only when that default is safe.
4. Prefer clear names like `distillmos.batch_size`,
   `diarization_loader_workers`, or `loudness_num_workers`.

Do not hide important model paths behind silent defaults. If a required model
path is missing, raise a clear error.

## Stage Module Shape

New stage modules should follow this structure:

```python
import argparse
from pathlib import Path

from loguru import logger

from src.utils.logging_setup import setup_logging
from src.utils.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    setup_logging("my_stage", log_dir=args.log_dir)
    config = load_config(args.config_path, "my_section")

    # Stage logic here.


if __name__ == "__main__":
    main()
```

For multiprocessing stages, set the start method in the executable entrypoint:

```python
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
```

## Code Style For Model Stages

Every model stage should keep three responsibilities separate:

1. **Dataset/DataLoader code** lives in `src/utils/datasets/`.
2. **Stage orchestration** lives in the stage module, for example
   `src/transcription/transcription.py`.
3. **Runtime knobs** live in `configs/config.yaml`.

Do not load model inputs ad hoc inside the inference loop. If a model consumes
audio, text, sidecar files, or tensors at scale, add a Dataset and a DataLoader
for that input path. The stage should receive already prepared batches and
focus on model execution, error handling, and writing results.

The expected shape is:

```python
# src/utils/datasets/<area>.py
class MyModelDataset(Dataset):
    ...


def create_my_model_dataloader(...) -> DataLoader:
    ...
```

```python
# src/<area>/<stage>.py
from src.utils.datasets.<area> import create_my_model_dataloader


def run_worker(gpu_id: int, world_size: int, paths: list[str], config: dict):
    model = load_model(...)
    dataloader = create_my_model_dataloader(paths, ...)

    for batch in dataloader:
        # Run model here.
        ...
```

When adding a stage, also add its config keys. At minimum, model stages usually
need:

- input root or source list, for example `podcasts_path`;
- model path or model name;
- `batch_size`;
- DataLoader worker count, for example `num_workers` or
  `<stage>_loader_workers`;
- `prefetch_factor` when `num_workers > 0`;
- backend/runtime flags such as `use_tensorrt`, quantization, thresholds, or
  cache paths when relevant.
- an explicit filter threshold for deletion stages, or a clearly documented
  manual mode like `separation.distillmos_filter.threshold: null`.

The project already has GPU parallelism helpers in `src/utils/parallel.py`:

- `run_per_gpu_processes(...)` for one large model process per GPU.
- `run_per_gpu_pool(...)` for per-GPU process pools with an initializer.

Before inventing a new GPU scheduling pattern, check whether one of these
helpers fits. For the clearest current example, read
`src/transcription/transcription.py`: it loads one ASR model per GPU worker,
builds a DataLoader through `src.utils.datasets.transcription`, and keeps the
stage logic separate from input loading.

## Dataset And DataLoader Layout

Dataset and DataLoader code belongs in `src/utils/datasets/`, not inside stage
modules.

Current layout:

- `src/utils/datasets/preprocess.py`
  - `CrestFactorDataset`
  - `LoudnessNormalizeDataset`
  - `DiarizationDataset`
- `src/utils/datasets/separation.py`
  - `DistillMOSDataset`
- `src/utils/datasets/transcription.py`
  - `TranscriptionDataset`
- `src/utils/datasets/denoising.py`
  - `DenoisingDataset`

When adding a new module, add the loader to the file that matches the pipeline
area. For example:

- Preprocess audio loading goes to `src/utils/datasets/preprocess.py`.
- Separation model loading goes to `src/utils/datasets/separation.py`.
- Transcription audio loading goes to `src/utils/datasets/transcription.py`.
- Denoising / enhancement loading goes to `src/utils/datasets/denoising.py`.

Use this pattern:

```python
class MyStageDataset(Dataset):
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            return path, waveform.to(dtype=torch.float32).contiguous(), int(sample_rate), ""
        except Exception as exc:
            return path, torch.empty(0, dtype=torch.float32), 0, str(exc)
```

Prefer returning load errors as data instead of crashing DataLoader workers.
The stage module should log the error and skip the bad file.

Use `torchaudio` and `torch` for audio IO and tensor work. Do not add `librosa`
or `soundfile` paths unless there is a very explicit project decision to do so.

### DataLoader Worker Guidance

Be careful with nested multiprocessing.

Good:

- One GPU process owns one model.
- That process uses a DataLoader for CPU-side decoding.
- DataLoader workers are used only when the stage is stable with them.

Avoid:

- Thread pool -> process pool -> multiprocessing DataLoader workers.
- Multiple GPU models sharing global model variables in the same Python process.
- `persistent_workers=True` when interruption safety matters more than speed.

For fragile GPU stages, start with `num_workers=0` in the DataLoader factory and
increase only after testing.

## GPU Stage Pattern

GPU-heavy stages should use one process per GPU. The model should be initialized
inside that process, not in the parent.

Typical pattern:

```python
def run_worker(rank: int, world_size: int, items: list[str], config: dict):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = load_model().to(device)
    model.eval()

    my_items = items[rank::world_size]
    dataloader = create_my_dataloader(my_items, ...)

    with torch.inference_mode():
        for batch in dataloader:
            # Move only model inputs to GPU.
            # Keep file paths and metadata on CPU.
            ...
```

Then launch with:

```python
mp.spawn(
    run_worker,
    args=(available_gpus, items, config),
    nprocs=available_gpus,
    join=True,
)
```

For ONNX Runtime stages, build providers through
`src.utils.gpu.get_onnx_providers(...)` so CUDA/TensorRT cache behavior stays
consistent across the project.

## CSV State And Resume Logic

Long-running stages should be resumable. Use `src.utils.csv_manager` instead of
hand-writing CSV merge logic.

Important helpers:

- `discover_audio_paths(podcasts_path)`: scan the audio tree.
- `ensure_main_csv(podcasts_path, audio_paths=...)`: create/load
  `balalaika.csv`.
- `unprocessed_paths(podcasts_path, column, audio_paths)`: skip already scored
  files.
- `PartialCsvWriter(...)`: stream worker results to
  `<prefix>_part_<rank>.csv`.
- `absorb_partial_csvs(...)`: merge partials into `balalaika.csv`.
- `upsert_columns(...)`: atomically merge result columns by `filepath`.

For stages that write one value per file, use this flow:

```python
audio_paths = discover_audio_paths(podcasts_path)
ensure_main_csv(podcasts_path, audio_paths=audio_paths)

_, absorbed = absorb_partial_csvs(
    podcasts_path,
    PARTIAL_PREFIX,
    value_columns=[COLUMN],
    bootstrap_audio_paths=audio_paths,
    preserve_existing=True,
)

pending = unprocessed_paths(podcasts_path, COLUMN, audio_paths)

# Run workers; each worker writes partial CSV rows.

absorb_partial_csvs(
    podcasts_path,
    PARTIAL_PREFIX,
    value_columns=[COLUMN],
    preserve_existing=True,
)
```

Partial CSV merges are sparse updates. They must never clear values for
filepaths absent from the partial:

- `preserve_existing=True` is the default for `upsert_columns()`,
  `absorb_partial_csvs()`, and `PeriodicCsvMerger`. Non-null incoming values
  update matching rows, while null incoming values cannot erase existing
  metadata.
- `preserve_existing=False` is reserved for an explicit recompute/overwrite.
  It may clear or replace values only for matching incoming filepaths.

Use `preserve_existing=True` explicitly in normal scoring, filtering, and
metadata-backfill stages so the retention requirement is visible at the call
site.

For filtering stages that delete files, pass `drop_missing_files=True` when
absorbing or upserting results.

If a filter consumes a score produced by a previous stage, follow
`src/separation/distillmos_filter.py`: read `balalaika.csv`, preview the effect
of the threshold, delete in workers, write partial CSV rows with enough metadata
to audit the decision, then merge/prune the main CSV.

## Avoid Full Tree Scans When Possible

`get_audio_paths()` is the raw recursive filesystem scan.
`discover_audio_paths()` follows `runtime.audio_paths_source`: `csv` trusts
`balalaika.csv`, `rglob` forces a scan, and `auto` prefers CSV with a scan
fallback. On very large datasets, prefer `csv` after `balalaika.csv` exists.

Do not pass multi-million-item path lists into `mp.spawn` / `mp.Process`. Use
`src.utils.work_shards.prepare_work_shards()` in the parent, pass only the
work directory to workers, and have workers claim shards with
`claim_work_shard()` + `mark_work_shard_done()`. This keeps multiprocessing
startup bounded and avoids pickle OOM failures.

## Logging

Every stage must call:

```python
setup_logging("stage_name", log_dir=args.log_dir)
```

Use `logger.info()` for stage progress, `logger.warning()` for recoverable
problems, and `logger.error()` for file-level failures. Use `logger.exception()`
only when the traceback is useful.

Logs go to the configured runtime log directory and should be enough to resume
or debug a failed batch run.

Stages should also call `src.utils.stage_status.write_stage_status(...)` before
exit. `base.sh --strict` reads `stage_<id>_status.json` from the log directory
and aborts the pipeline when `errors > 0`.

## Audit And Reports

Stages that remove or transform dataset size should record audit summaries with
`src.utils.audit.record_stage_summary`.

Use it when a stage changes:

- file count,
- total hours,
- filtering decisions,
- quality thresholds.

Current audit-producing stages include `preprocess`, `crest_factor`,
`music_detect`, and `distillmos_filter`.

Example:

```python
record_stage_summary(
    podcasts_path=podcasts_path,
    stage="my_filter",
    files_in=files_in,
    files_out=files_out,
    hours_in=hours_in,
    hours_out=hours_out,
    params={"threshold": threshold},
)
```

## Adding A New Stage

1. Create the module under the correct package, for example
   `src/separation/my_score.py`.
2. Add a config section or subsection in `configs/config.yaml`.
3. Put Dataset/DataLoader code in `src/utils/datasets/<area>.py`.
4. Use `setup_logging(...)` and `load_config(...)`.
5. Use `csv_manager` helpers for `balalaika.csv` state.
6. Use one process per GPU for GPU-heavy models.
7. Add the stage to `base.sh` if it should be part of the main pipeline.
8. Add or update a `*_yaml.sh` wrapper if users need a direct stage script.
9. Write a stage status file with `write_stage_status(...)` so `--strict`
   works.
10. Run syntax checks:

```bash
.dev_venv/bin/python -m py_compile src/path/to/module.py src/utils/datasets/<area>.py
```

11. Run the stage on a small limit or small test directory before launching the
    full dataset.

## Common Pitfalls

- Do not put Dataset classes inside stage modules.
- Do not use `librosa` for audio loading or feature extraction.
- Do not use `soundfile` unless the project explicitly decides to restore it.
- Do not initialize GPU models in the parent process before `spawn`.
- Do not share global model objects across GPU threads.
- Do not assume CPU-only execution is supported for GPU stages.
- Do not silently skip model path errors.
- Do not rewrite `balalaika.csv` manually; use `csv_manager`.
- Do not force full filesystem scans when `balalaika.csv` already contains the
  file list.

## Current Audio Stack

The project standard is:

- `torch` for tensors and batching.
- `torchaudio` for audio loading, saving, resampling, and STFT-like operations.
- ONNX Runtime for exported model inference.
- TensorRT EP through shared provider helpers when enabled.

Keep new code aligned with that stack.
