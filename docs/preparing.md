# Preparing your dataset

You can either **use ready-made Balalaika exports on Hugging Face** or **run
this repo** on your own audio.

---

## Pre-built datasets (Hugging Face)

MTUCI hosts processed datasets in the **[Balalaika Dataset
collection](https://huggingface.co/collections/MTUCI/balalaika-dataset)** on
Hugging Face. Those snapshots are already segmented, filtered, and annotated —
use them directly for training or evaluation without running download →
preprocess → … locally.

To load a WebDataset-style export with the Hugging Face `datasets` library,
follow [example/README.md](../example/README.md).

---

## Running the pipeline on your own data

### 1. Directory layout

Put **`podcasts_path`** at the root of your tree (use the same root everywhere
in `configs/config.yaml`). For local annotation, a practical layout is **one
subfolder per source group**, with **one long audio file per recording** (name
= episode / clip id):

```text
dataset/                    # = podcasts_path
├── balalaika.csv           # created/updated by preprocess and later stages
├── filter_summary.csv      # audit of files/hours removed at each filter stage
├── filter_report.md        # human-readable report (src/report.py)
├── 00/
│   ├── audio1.flac
│   ├── audio2.wav
│   └── ...
├── 01/
│   ├── interview_a.mp3
│   └── ...
└── ...
```

Rules (see `src/preprocess/preprocess.py`):

- **`playlist_id`** = the **immediate parent folder name** (`00`, `01`, … or
  any stable label — album, speaker batch, etc.).
- **`podcast_id`** = the **file name without extension** (`audio1` from
  `audio1.flac`). Use **unique stems** inside each playlist folder so metadata
  doesn't collide.

Supported source extensions are whatever `get_audio_paths` collects: `.mp3`,
`.wav`, `.flac`, `.ogg`, `.opus`.

After **preprocess**, a long file is **removed** once chunks are written.
Chunks land in a subfolder named after that file's stem and **inherit the
source extension by default** (FLAC stays FLAC, no surprise transcoding):

```text
{playlist_id}/{podcast_id}/{start}_{end}_{playlist_id}_{podcast_id}.{ext}
```

Example: from `dataset/00/audio1.flac` you get files like
`dataset/00/audio1/12.50_26.30_00_audio1.flac`.

If you want a fixed container regardless of input, set
`preprocess.chunk_format` to one of `flac` / `wav` / `mp3` / `ogg` / `opus`
(`auto` is the format-preserving default).

If a file's total duration is already within the configured maximum segment
duration (`preprocess.duration`, used as `max_duration` by the chunker), it is
**left in place** and only metadata is appended to `balalaika.csv` (no chunk
subfolder for that case).

### 2. Environment

Create a `.env` in the repo root (see main [README.md](../README.md)):

- **`HF_TOKEN`** — Hugging Face token for gated models and hub downloads
  (RUPunct, onnx-asr models, etc.).
- **`YANDEX_KEY`** — only if you use the Yandex Music **download** stage.

Optional:

- Runtime defaults now live in `configs/config.yaml` under `runtime:`:
  `venv_path`, `cpu_affinity`, `log_dir`, TensorRT cache path/workspace, and
  `trt_fp16`.
- **`BALALAIKA_LOG_DIR`** can still override the log directory for direct
  module runs. When using `base.sh`, prefer `runtime.log_dir` or the per-stage
  `--log_dir <path>` flag.

### 3. Configuration

Edit **`configs/config.yaml`**: keep `podcasts_path` aligned in every section
that processes your dataset, choose batch sizes and worker counts, set quality
thresholds, define `transcription.model_names`, and choose `preprocess.chunk_format`
(`auto` preserves the source container). The file includes an inline
**parameter reference** at the top.

Important thresholds:

- `preprocess.crest_treshold`: deletes files with high peak/RMS ratio.
- `separation.music_detect.threshold`: deletes music-heavy clips.
- `separation.distillmos_filter.threshold`: deletes clips whose `DistillMOS`
  score is below the threshold. Set it to `null` if you want to choose the
  threshold interactively after seeing the MOS distribution.

**Collate note:** `src/collate.py` reads the **`download`** section for
`podcasts_path` and `num_workers`. Keep `download.podcasts_path` the same as
your working dataset root, or collate will look in the wrong place.

### 4. Run order

`base.sh` is a Kaldi-style runner with `--stage` / `--stop_stage` flags
(stages 0..13 plus stage 5.5, see [docs/guide.md](guide.md) for the table).
With no flags it runs stages 1..9: chunking through phonemization.

```bash
bash base.sh --config_path configs/config.yaml
```

Full local pipeline without Yandex download:

```bash
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 13
```

Include Yandex download:

```bash
bash base.sh --config_path configs/config.yaml --stage 0 --stop_stage 13
```

Just preprocess and audio normalization:

```bash
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3
```

Stage 10 runs ClearVoice `MossFormer2_SE_48K` denoising / speech enhancement and
overwrites audio in place at 48 kHz by default. The last stage (13) runs
`src/report.py`, which materializes
`filter_report.md` next to your dataset so you can see how much audio (in
hours) was filtered at each step.

To run only the DistillMOS quality filter after scoring:

```bash
bash base.sh --config_path configs/config.yaml --stage 5.5 --stop_stage 5.5
```

---

## More reading

- [Usage Guide](guide.md) — stages, artifacts, logging layout, filter report.
- [example/README.md](../example/README.md) — WebDataset + `datasets` loading.
