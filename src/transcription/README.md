## Transcription (onnx-asr)

ASR via **[onnx-asr](https://github.com/istupakov/onnx-asr)** on **ONNX Runtime**, optionally **TensorRT** — no custom PyTorch dataloaders in this repo.

### Features

- Run multiple models sequentially with **early skip** when `consensus_num` earlier models agree on normalized text.
- **ROVER** → `{stem}_rover.txt` when `use_rover: True`.
- **Word-level timestamps** → `{stem}_{model}.tst` (TSV) when `with_timestamps: True` and the model is in the supported set.
- **Multi-GPU** via `src.utils.parallel.run_per_gpu_processes` (one process per GPU; the model is loaded once per process).
- **TensorRT providers** built by `src.utils.gpu.get_onnx_providers`, sharing the engine cache root with every other ONNX-RT stage.

### Typical `model_names` (Russian)

| Config name | Backend (onnx-asr / HF id) |
|-------------|----------------------------|
| `giga_ctc` | GigaAM v3 CTC |
| `giga_rnnt` | GigaAM v3 RNN-T |
| `vosk` | Vosk Russian |
| `tone` | T-one |

Others: `parakeet_v2`, `parakeet_v3`, `canary`, `whisper_base`, `whisper_turbo`, … — see `MODEL_MAP` in `transcription.py` and comments in `configs/config.yaml`.

## Run

```bash
bash src/transcription/transcription_yaml.sh configs/config.yaml
```

## Config snippet

All keys are documented under **`transcription`** in `configs/config.yaml` (`podcasts_path`, `consensus_num`, `with_timestamps`, `use_tensorrt`, `use_vad`, `use_rover`, `model_names`, `batch_size`, plus optional `model_path`, `vosk_path`, `quantization`, `vad_params`).

## On-disk artifacts

For chunk `{stem}.mp3`:

- `{stem}_{model}.txt` — hypothesis.
- `{stem}_{model}.tst` — timestamps when enabled.
- `{stem}_rover.txt` — ROVER consensus.

## Resume / interrupt safety

Transcription does **not** touch `balalaika.csv`; per-file results live in the
`.txt` / `.tst` sidecars next to the audio:

* The shared `pending_*` helpers in `src.utils.sidecars` skip any chunk that
  already has a `{stem}_{model}.txt` (or matches the `consensus_num` early
  skip rule) so a forced stop simply resumes on the next run.
* `run_per_gpu_processes` cleanly terminates child processes on `Ctrl+C`.
* Pending audio paths are written to `.balalaika_work/transcription_<model>/`
  and claimed shard-by-shard, so large runs do not pickle huge path lists into
  GPU workers.
* ROVER also runs shard-by-shard under `.balalaika_work/transcription_rover/`
  and writes `{stem}_rover.txt` after each shard, so it does not build one
  dataset-wide CrowdKit DataFrame. `transcription.rover_shard_size` can be set
  lower than `runtime.work_shard_size` when transcripts are large.
* `transcription.rover_workers` controls how many CPU processes claim ROVER
  shards in parallel.

## Memory / OOM safety

Audio decoding (torchcodec/ffmpeg) churns the glibc heap, so each DataLoader
worker's RSS climbs to a multi-GB high-water mark and — with
`persistent_loaders: True` — never falls, which can OOM-kill workers on a
RAM-tight box (the failure surfaces as `DataLoader worker (pid …) exited
unexpectedly`). The transcription datasets call `malloc_trim(0)` every N decoded
items to return that memory to the OS, holding a worker near ~1 GB instead of
~4.5 GB.

Tune via the **`runtime.malloc_trim_every`** key in `configs/config.yaml`
(default `128`; `0` disables), exported by `base.sh` as
**`BALALAIKA_MALLOC_TRIM_EVERY`**. A shell `export` overrides it for a one-off
run since the dataset reads the env var directly:

```bash
BALALAIKA_MALLOC_TRIM_EVERY=64 bash src/transcription/transcription_yaml.sh configs/config.yaml
```

See `src/utils/datasets/README.md` for the measured numbers and rationale.

## Dependencies

`create_dev_env.sh` typically installs nightly **onnxruntime-gpu** for your CUDA, **`tensorrt-cu13`** (or matching wheel), and **`onnx-asr[gpu,hub]`** — pin versions there.
