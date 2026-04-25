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

If a file is **shorter than** `preprocess.duration`, it is **left in place**
and only metrics are appended to `balalaika.csv` (no chunk subfolder for that
case).

### 2. Environment

Create a `.env` in the repo root (see main [README.md](../README.md)):

- **`HF_TOKEN`** — Hugging Face token for gated models and hub downloads
  (RUPunct, onnx-asr models, etc.).
- **`YANDEX_KEY`** — only if you use the Yandex Music **download** stage.

Optional:

- **`BALALAIKA_LOG_DIR`** — directory for per-stage rotating log files
  (defaults to `./logs`). Each stage also accepts `--log_dir <path>` to
  override per invocation.

### 3. Configuration

Edit **`configs/config.yaml`**: absolute `podcasts_path` everywhere you
process data, batch sizes, thresholds, `model_names` for transcription and
the new `chunk_format` knob. The file includes an inline **parameter
reference** at the top.

**Collate note:** `src/collate.py` reads the **`download`** section for
`podcasts_path` and `num_workers`. Keep `download.podcasts_path` the same as
your working dataset root, or collate will look in the wrong place.

### 4. Run order

`base.sh` is a Kaldi-style runner with `--stage` / `--stop_stage` flags
(stages 0..12, see [docs/guide.md](guide.md) for the table). Full pipeline:

```bash
bash base.sh --config_path configs/config.yaml
```

Just preprocess:

```bash
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3
```

The last stage (12) runs `src/report.py`, which materializes
`filter_report.md` next to your dataset so you can see how much audio (in
hours) was filtered at each step.

---

## More reading

- [Usage Guide](guide.md) — stages, artifacts, logging layout, filter report.
- [example/README.md](../example/README.md) — WebDataset + `datasets` loading.
