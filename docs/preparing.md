# Preparing your dataset

You can either **use ready-made Balalaika exports on Hugging Face** or **run this repo** on your own audio.

---

## Pre-built datasets (Hugging Face)

MTUCI hosts processed datasets in the **[Balalaika Dataset collection](https://huggingface.co/collections/MTUCI/balalaika-dataset)** on Hugging Face. Those snapshots are already segmented, filtered, and annotated through the Balalaika-style pipeline—use them directly for training or evaluation without running download → preprocess → … locally.

To load a WebDataset-style export with the Hugging Face `datasets` library, follow [example/README.md](../example/README.md).

---

## Running the pipeline on your own data

### 1. Directory layout (for annotating your own corpus)

Put **`podcasts_path`** at the root of your tree (use the same root everywhere in `configs/config.yaml`). For local annotation, a practical layout is **one subfolder per source group**, with **one long audio file per recording** (name = episode / clip id):

```text
dataset/                    # = podcasts_path
├── balalaika.csv           # created/updated by preprocess and later stages (may appear after first run)
├── 00/
│   ├── audio1.mp3
│   ├── audio2.mp3
│   └── ...
├── 01/
│   ├── interview_a.mp3
│   └── ...
└── ...
```

Rules (see `src/preprocess/preprocess.py`):

- **`playlist_id`** = the **immediate parent folder name** (`00`, `01`, … or any stable label—album, speaker batch, etc.).
- **`podcast_id`** = the **file name without extension** (`audio1` from `audio1.mp3`). Use **unique stems** inside each playlist folder so metadata does not collide.

Supported extensions are whatever `get_audio_paths` collects (e.g. `.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`).

After **preprocess**, a long file is **removed** once chunks are written. Chunks land in a subfolder named after that file’s stem:

```text
{playlist_id}/{podcast_id}/{start}_{end}_{playlist_id}_{podcast_id}.mp3
```

Example: from `dataset/00/audio1.mp3` you get files like `dataset/00/audio1/12.50_26.30_00_audio1.mp3`.

If a file is **shorter than** `preprocess.duration`, it is **left in place** and only metrics are appended to `balalaika.csv` (no chunk subfolder for that case).

### 2. Environment

Create a `.env` in the repo root (see main [README.md](../README.md)):

- **`HF_TOKEN`** — Hugging Face token for gated models and hub downloads (RUPunct, onnx-asr models, etc.).
- **`YANDEX_KEY`** — only if you use the Yandex Music **download** stage.

### 3. Configuration

Edit **`configs/config.yaml`**: absolute `podcasts_path` everywhere you process data, batch sizes, thresholds, and `model_names` for transcription. The file includes an inline **parameter reference** at the top.

**Collate note:** `src/collate.py` reads the **`download`** section for `podcasts_path` and `num_workers`. Keep `download.podcasts_path` the same as your working dataset root, or collate will look in the wrong place.

### 4. Run order

Use [docs/guide.md](guide.md) for stage-by-stage commands, or uncomment stages in `base.sh` and run:

```bash
bash base.sh configs/config.yaml
```

---

## More reading

- [Usage Guide](guide.md) — stages and artifacts (some sections may still mention older tooling; prefer `src/*/README.md` and `configs/config.yaml` for the current stack).
- [example/README.md](../example/README.md) — WebDataset + `datasets` loading example.
