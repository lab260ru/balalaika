# Pipeline Datasets

Put DataLoader and Dataset code here, one file per pipeline module.

Examples:
- `transcription.py` contains transcription audio decoding, collation, and batch ASR helpers.
- `separation.py` contains separation-stage datasets such as DistillMOS and Spectra-0 anti-spoofing audio loading.
- `preprocess.py` can contain preprocess-stage datasets when that stage moves to DataLoader-based input.

## Periodic heap trim (`transcription.py`)

torchcodec/ffmpeg decode churn leaves glibc holding freed memory: a DataLoader
worker's RSS ratchets up to a multi-GB high-water mark and, with
`persistent_loaders: True`, never drops — which OOM-kills workers on a
RAM-constrained box. Every `__getitem__` calls `_periodic_malloc_trim()`, which
runs `malloc_trim(0)` once per N decoded items (in whichever process decodes —
each loader worker keeps its own counter; also covers the inline single-GPU
path). Measured on the real group loader: worker heap **1451 → 221 MB**, RSS
**2218 → 988 MB**. It does not affect outputs (memory only) and is a no-op on
non-glibc libc.

**Where to set it:** the **`runtime.malloc_trim_every`** key in
`configs/config.yaml` (default `128`; `0` disables). `base.sh` /
`runtime_env.py` export it as **`BALALAIKA_MALLOC_TRIM_EVERY`**, which the
dataset reads. Because the dataset reads the env var directly, a shell `export`
overrides the config for a one-off run:

```yaml
runtime:
  malloc_trim_every: 128   # lower = tighter RAM, slightly more trims; 0 disables
```

```bash
BALALAIKA_MALLOC_TRIM_EVERY=64 bash base.sh --config_path configs/config.yaml --stage 8 --stop_stage 8
```

