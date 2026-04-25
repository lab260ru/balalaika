# Balalaika

Balalaika is an end-to-end speech data pipeline: download or ingest audio,
diarize and chunk it, filter low-quality material, run multi-model ASR with
ROVER, restore punctuation and stress marks, phonemize text, and export the
result to Parquet / WebDataset.

The current runner is stage-based (`--stage` / `--stop_stage`), so long jobs can
be resumed from any point without editing shell scripts.

---

## Quick Start

Install system tools and create the development environment:

```bash
sudo apt update && sudo apt install -y ffmpeg
wget -qO- https://astral.sh/uv/install.sh | sh

git clone https://github.com/mtuciru/balalaika
cd balalaika
bash create_dev_env.sh
```

Create `.env` in the repo root:

```ini
HF_TOKEN=<your_huggingface_token>
YANDEX_KEY=<your_yandex_music_token>
```

Edit `configs/config.yaml`:

- set `podcasts_path` in every stage you plan to run;
- set model paths under `preprocess`, `separation`, etc.;
- tune `runtime:` (`venv_path`, `cpu_affinity`, `log_dir`, TensorRT cache).

Run the full pipeline:

```bash
bash base.sh --config_path configs/config.yaml
```

Run only a range of stages:

```bash
# Preprocess only: chunking -> crest filter -> loudness normalization
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3

# Transcription only
bash base.sh --config_path configs/config.yaml --stage 6 --stop_stage 6

# Regenerate the filtering report only
bash base.sh --config_path configs/config.yaml --stage 12 --stop_stage 12
```

---

## Stage Map

| ID | Stage | Module |
|----|-------|--------|
| 0 | Download from Yandex Music | `src.download.download` |
| 1 | Preprocess: diarization + chunking | `src.preprocess.preprocess` |
| 2 | Preprocess: crest-factor filter | `src.preprocess.crest_factor_remover` |
| 3 | Preprocess: loudness normalization | `src.preprocess.preprocess_audio` |
| 4 | Separation: music detection | `src.separation.music_detect` |
| 5 | Separation: DistillMOS scoring | `src.separation.distillmos_process` |
| 6 | Transcription + ROVER | `src.transcription.transcription` |
| 7 | Punctuation | `src.punctuation.punctuation` |
| 8 | Stress marks / accents | `src.accents.accents` |
| 9 | Phonemization | `src.phonemizer.phonemizer` |
| 10 | Collate to Parquet | `src.collate` |
| 11 | Export to WebDataset | `src.to_webdataset` |
| 12 | Filtering report | `src.report` |

`base.sh --help` prints the same map.

---

## Runtime Configuration

Shell-level values live in `configs/config.yaml` under `runtime:` instead of
being hardcoded in `base.sh`:

```yaml
runtime:
  venv_path: .dev_venv
  cpu_affinity: "0-24"          # empty string disables taskset
  log_dir: ./logs
  trt_cache_path: ./cache/trt
  trt_workspace_bytes: 4294967296
  trt_fp16: True
```

`base.sh` reads this block through `src.utils.runtime_env` and exports
`BALALAIKA_*` variables before running stages. Python modules use the same
values for logging and TensorRT provider setup.

---

## Audio Quality

Balalaika avoids silent quality degradation:

- `preprocess.chunk_format: auto` preserves the source container when chunking
  (`.flac` input produces `.flac` chunks, `.wav` stays `.wav`).
- Set `preprocess.chunk_format` to `flac`, `wav`, `mp3`, `ogg`, or `opus` only
  when you explicitly want a fixed output container.
- Loudness normalization writes FLAC/WAV through `soundfile` as lossless
  containers. Lossy formats are handled by `torchaudio.save`.
- WebDataset export copies the produced audio bytes as-is.

---

## Logs And Filtering Report

Every stage writes a timestamped log file:

```text
<runtime.log_dir>/<stage>_YYYYMMDD-HHMMSS.log
```

Filtering stages append audit rows to:

```text
<podcasts_path>/filter_summary.csv
```

The final report stage reads this CSV and writes:

```text
<podcasts_path>/filter_report.md
```

The report shows files and hours kept / removed at each filtering step.

---

## Outputs

Main dataset-level files:

| File | Purpose |
|------|---------|
| `balalaika.csv` | Per-chunk metadata and quality scores |
| `filter_summary.csv` | Machine-readable filtering audit |
| `filter_report.md` | Human-readable filtering report in hours |
| `balalaika.parquet` | Aggregated metadata for downstream training |
| WebDataset shards | Audio + JSON samples for large-scale loading |

Per-chunk sidecars include model ASR text, optional timestamps, ROVER output,
punctuated text, accent-marked text, and phonemes.

---

## Documentation

- [Preparing your dataset](docs/preparing.md) — expected folder layout and setup.
- [Usage Guide](docs/guide.md) — detailed stage behavior and commands.
- [example/README.md](example/README.md) — loading exported WebDataset shards.

Per-module notes live under `src/*/README.md`.

---

## Models & Tooling

| Piece | Role |
|-------|------|
| Sortformer (ONNX) | streaming diarization and speaker turns |
| Smart Turn (`smart-turn-v3.0.onnx`) | end-of-turn refinement |
| WavLM music detector | removes music-heavy chunks |
| DistillMOS | predicts speech quality score |
| [onnx-asr](https://github.com/istupakov/onnx-asr) | ASR models and optional TensorRT |
| RUPunct | punctuation restoration |
| ruAccent | lexical stress marks |
| TryIParu | grapheme-to-phoneme conversion |

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