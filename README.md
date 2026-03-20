# Balalaika Pipeline

End-to-end speech data processing: ingest, segmentation, quality filtering, multi-model ASR with ROVER, punctuation, lexical stress, G2P, and export to Parquet / WebDataset.

Works with Yandex Music podcasts out of the box, or **your own corpus** if you follow the expected layout (see [Preparing your dataset](docs/preparing.md)).

**Pre-built processed datasets** (segmented, filtered, annotated) are published on Hugging Face: **[Balalaika Dataset — MTUCI collection](https://huggingface.co/collections/MTUCI/balalaika-dataset)**.

---

## Quick Start

### Prerequisites

```bash
sudo apt update && sudo apt install -y ffmpeg
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
git clone https://github.com/mtuciru/balalaika
cd balalaika
bash create_dev_env.sh   # full stack for running the pipeline
# or
bash create_user_env.sh  # consume pre-built datasets only
```

### Basic setup

1. Create `.env`:

```ini
HF_TOKEN=<your_huggingface_token>
YANDEX_KEY=<your_yandex_music_token>
```

2. Edit `configs/config.yaml`: set absolute paths (`podcasts_path`, model files under `models/`, etc.).

3. Run stages (see [Usage Guide](docs/guide.md)). Sequential wrapper:

```bash
bash base.sh configs/config.yaml
```

Note: `base.sh` may have early stages commented out—uncomment what you need.

---

## Documentation

- **[Preparing your dataset](docs/preparing.md)** — HF collection vs. local pipeline, folder layout, models, config.
- **[Usage Guide](docs/guide.md)** — stages, artifacts, per-step commands.
- **[example/README.md](example/README.md)** — loading the WebDataset with Hugging Face `datasets`.

Per-module notes live under `src/*/README.md` (aligned with `configs/config.yaml`).

---

## Pipeline overview

1. **Download** — optional episode fetch.
2. **Preprocess** — crest-factor pruning, **Sortformer (ONNX)** diarization, single-speaker selection, **Smart Turn** boundary refinement, chunking + `balalaika.csv`; long source files are removed after successful chunking; **EBU R128-style** loudness normalization (see `preprocess_yaml.sh` order).
3. **Separation** — **music detection** (WavLM-based weights in `music_detection.safetensors`), **DistillMOS** → column in `balalaika.csv`.
4. **Transcription** — **[onnx-asr](https://github.com/istupakov/onnx-asr)** (ONNX Runtime / optional TensorRT), **ROVER** consensus, optional word-level `.tst`.
5. **Punctuation** — RUPunct.
6. **Accents** — ruAccent (e.g. `turbo3.1`).
7. **Phonemization** — **TryIParu** `G2PModel` → `*_rover_phonemes.txt`.
8. **Collate / export** — `balalaika.parquet` and WebDataset shards via `src/collate_yamls.sh`.

---

## Citation

```bibtex
@article{borodin2025datacentric,
  title={A Data-Centric Framework for Addressing Phonetic and Prosodic Challenges in Russian Speech Generative Models},
  author={Borodin, Kirill and Vasiliev, Nikita and Kudryavtsev, Vasiliy and Maslov, Maxim and Gorodnichev, Mikhail and Rogov, Oleg and Mkrtchian, Grach},
  journal={arXiv preprint arXiv:2507.13563},
  year={2025}
}
```

**Paper**: [arXiv:2507.13563](https://arxiv.org/abs/2507.13563)  
**DOI**: [10.48550/arXiv.2507.13563](https://doi.org/10.48550/arXiv.2507.13563)

---

## Models & tooling

| Piece | Role |
|--------|------|
| **Sortformer** (ONNX) | streaming diarization, single-speaker slices |
| **[Smart Turn](https://github.com/pipecat-ai/smart-turn)** (`smart-turn-v3.0.onnx`) | end-of-speech / turn boundaries |
| **Music detector** (`music_detection.safetensors`) | drop music-heavy chunks |
| **DistillMOS** | predicted MOS in `balalaika.csv` |
| **[onnx-asr](https://github.com/istupakov/onnx-asr)** | GigaAM v3 CTC/RNNT, Vosk, T-one, Parakeet, Canary, Whisper, … |
| **[RUPunct](https://huggingface.co/RUPunct/RUPunct_big)** | punctuation |
| **[ruAccent](https://github.com/Den4ikAI/ruaccent)** | stress marks |
| **TryIParu** (`tryiparu`) | grapheme → IPA |

---

## License

See [LICENSE](LICENSE).