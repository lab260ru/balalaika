# Balalaika WebDataset (Hugging Face)

Pipeline export is packed as [WebDataset](https://github.com/webdataset/webdataset) `.tar` shards. Load with Hugging Face [`datasets`](https://huggingface.co/docs/datasets) and `streaming=True` to avoid holding the full corpus in RAM.

## Install

```bash
pip install datasets webdataset
```

You may also need `torchaudio` (or another backend) depending on your `datasets` version for decoding `mp3` / `wav` columns.

## Loading

After `src/collate_yamls.sh`, shards are written to  
`{parent of podcasts_path}/{dataset_folder_name}_webdataset/train/`  
(see `src/to_webdataset.py`). Point `data_dir` at that `train` folder.

[`example.py`](example.py) shows a minimal loop.

```python
from datasets import load_dataset

dataset = load_dataset(
    "webdataset",
    data_dir="/path/to/your_dataset_webdataset/train",
    split="train",
    streaming=True,
)

for item in dataset:
    print(item["__key__"])
    audio_key = next(k for k in item if k in ("mp3", "wav", "flac", "ogg"))
    audio = item[audio_key]
    print(audio["array"].shape, audio["sampling_rate"])
    print(item["json"])  # dict: CSV metadata + all sidecar texts
```

## Sample layout

| Field | Type | Description |
|--------|------|-------------|
| `__key__` | `str` | Sample id; dots in the stem are replaced with `_` for HF / WebDataset parsing. |
| `mp3` / `wav` / … | `dict` | Audio: NumPy `array`, `sampling_rate`. Extension matches the chunk on disk. |
| `json` | `dict` | Merged metadata from `balalaika.csv` plus every sidecar file next to the chunk. |

## Typical `json` keys (full run)

Keys mirror CSV columns and filenames `{stem}_{postfix}` → JSON key `postfix`.

**From `balalaika.csv`:** e.g. `start`, `end`, `total_duration`, `speaker_id`, `playlist_id`, `podcast_id`, `silence_percent`, `max_silence_duration`, `crest_factor`, `music_prob`, `DistillMOS`, and optionally `is_single_speaker`, etc. Only files that passed all filters are in the dataset.

**Text sidecars** (depend on `transcription.model_names` and which stages you ran):

| Key | Content |
|-----|---------|
| `giga_ctc.txt`, `giga_rnnt.txt`, `vosk.txt`, `tone.txt`, … | Raw ASR text per model. |
| `giga_ctc.tst`, `tone.tst`, … | Word-level TSV: `start_sec\tend_sec\tword` per line if timestamps enabled. |
| `rover.txt` | Multi-model consensus (ROVER). |
| `punct.txt` | Punctuation restoration (RUPunct). |
| `accent.txt` | Stress marks + normalized text (ruAccent). |
| `rover_phonemes.txt` | IPA string from consensus text (TryIParu `G2PModel`). |

## Why WebDataset

- **Throughput**: sequential read of large `.tar` files instead of millions of tiny files.
- **Streaming**: train without fully unpacking to disk.
- **HF-friendly**: same loader pattern for local folders or Hub-hosted shards.
