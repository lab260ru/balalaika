# Balalaika Pipeline

A complete production-ready pipeline for processing audio data, from download to feature extraction. This pipeline handles audio preprocessing, speaker diarization, transcription, punctuation restoration, accent restoration, and phonemization.

The pipeline is designed to work with podcasts from Yandex Music, but **can also process your own audio datasets** by organizing them in the expected format (see [Preparing Your Dataset](docs/preparing.md)).

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
bash create_dev_env.sh  # For development/annotation
# or
bash create_user_env.sh  # For using pre-annotated dataset only
```

### Basic Setup

1. Create `.env` file:
```ini
HF_TOKEN=<your_huggingface_token>
YANDEX_KEY=<your_yandex_music_token>
```

2. Configure `configs/config.yaml`:
```yaml
podcasts_path: /absolute/path/to/your/data
```

3. Place required models in `models/` directory (see [Preparing Your Dataset](docs/preparing.md))

4. Run the pipeline:
```bash
bash base.sh configs/config.yaml
```

---

## Documentation

- **[Usage Guide](docs/guide.md)**: Detailed guide on how to use the pipeline, what files are created, and how to run individual stages
- **[Preparing Your Dataset](docs/preparing.md)**: Setup instructions and how to prepare your own datasets for processing

---

## Pipeline Overview

The pipeline consists of the following stages:

1. **Download** - Downloads podcast episodes (optional, if using your own data)
2. **Preprocess** - Audio quality filtering and normalization:
   - **Crest Factor Removal** - Removes files with excessive peak-to-RMS ratio
   - **Loudness Normalization** - Normalizes audio loudness (ITU-R BS.1770-4)
   - **Audio Segmentation** - Segments audio into chunks using VAD
3. **Separation** - Speaker diarization, quality assessment, music detection
4. **Transcription** - Multi-model ASR with ROVER consensus (optimized with early stopping when models agree)
5. **Punctuation** - Punctuation restoration
6. **Accents** - Accent restoration
7. **Phonemization** - Text-to-phoneme conversion
8. **Collate** - Aggregates metadata into Parquet

For detailed information about each stage, see [Usage Guide](docs/guide.md).

---

## Citation

If you use this pipeline or the Balalaika dataset in your research, please cite:

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

## Models

The pipeline integrates the following models and tools:

- **[NISQA](https://github.com/gabrielmittag/NISQA)**: Audio quality assessment
- **[GigaAM](https://github.com/salute-developers/GigaAM)**: ASR models (CTC, RNNT, CTC+LM)
- **[ruAccent](https://github.com/Den4ikAI/ruaccent)**: Accent restoration
- **[RUPunct](https://huggingface.co/RUPunct/RUPunct_big)**: Punctuation restoration
- **[Vosk](https://alphacephei.com/vosk/)**: ASR model
- **[T-one](https://github.com/voicekit-team/T-one)**: ASR model
- **[TryIPaG2P](https://github.com/mtuciru/IpaG2p)**: Phonemization
- **[Speaker Diarization](https://github.com/pyannote/pyannote-audio)**: Speaker diarization
- **[Smart Turn VAD](https://github.com/pipecat-ai/smart-turn)**: Voice Activity Detection

---

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Thanks to all the developers and contributors who made this project possible, including the teams behind GigaAM, ruAccent, RUPunct, Vosk, and other integrated tools.
