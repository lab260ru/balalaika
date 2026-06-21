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
sudo apt update && sudo apt install -y \
  ffmpeg \                 # video/audio toolkit
  python3 \                # Python
  python3-pip \            # Pip package manager
  python3-venv \           # std-lib virtual-env support
  python3-dev \            # headers for compiling native wheels
  python-is-python3
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

Run the default tail of the pipeline (stages 12..15):

```bash
bash base.sh --config_path configs/config.yaml
```

Run only a range of stages:

```bash
# Preprocess only: chunking -> crest filter -> loudness normalization
bash base.sh --config_path configs/config.yaml --stage 1 --stop_stage 3

# Transcription only
bash base.sh --config_path configs/config.yaml --stage 8 --stop_stage 8

# Regenerate the filtering report only
bash base.sh --config_path configs/config.yaml --stage 15 --stop_stage 15
```

---

## Stage Map

| ID | Stage | Module |
|----|-------|--------|
| 0 | Download from Yandex Music | `src.download.download` |
| 1 | Preprocess: diarization + chunking | `src.preprocess.preprocess` |
| 2 | Preprocess: crest-factor filter | `src.preprocess.crest_factor_remover` |
| 3 | Preprocess: loudness normalization | `src.preprocess.preprocess_audio` |
| 4 | Separation: music scoring | `src.separation.music_detect` |
| 4.5 | Separation: music filter | `src.separation.music_detect_filter` |
| 5 | Separation: DistillMOS scoring | `src.separation.distillmos_process` |
| 5.5 | Separation: DistillMOS filter | `src.separation.distillmos_filter` |
| 6 | Separation: Spectra-0 raw scoring | `src.separation.antispoofing` |
| 6.5 | Separation: anti-spoofing filter | `src.separation.antispoofing_filter` |
| 7 | Separation: TTS-suitability scoring | `src.separation.tts_suitability` |
| 7.5 | Separation: TTS-suitability filter | `src.separation.tts_suitability_filter` |
| 8 | Transcription + ROVER | `src.transcription.transcription` |
| 9 | Punctuation | `src.punctuation.punctuation` |
| 10 | Stress marks / accents | `src.accents.accents` |
| 11 | Phonemization | `src.phonemizer.phonemizer` |
| 12 | Denoising / enhancement | `src.denoising.denoising` |
| 13 | Collate to Parquet | `src.collate` |
| 14 | Export to WebDataset | `src.to_webdataset` |
| 15 | Filtering report | `src.report` |

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
  audio_paths_source: csv        # csv avoids repeated rglob after balalaika.csv exists
  work_shard_size: 10000         # file paths per on-disk multiprocessing shard
  trt_cache_path: ./cache/trt
  trt_workspace_bytes: 4294967296
  trt_fp16: True
```

`base.sh` reads this block through `src.utils.runtime_env` and exports
`BALALAIKA_*` variables before running stages. Python modules use the same
values for logging, TensorRT provider setup, CSV-backed file discovery, and
on-disk work-shard sizing. Heavy stages write work queues under
`<podcasts_path>/.balalaika_work/<stage>/` so multiprocessing workers claim
small shard files instead of receiving millions of paths through pickle.

### Per-node batch-size autotuning

Run once on every new machine:

```bash
python -m benchmarking.warmup --config_path configs/config.yaml
```

This probes each tunable model with growing batch sizes (respecting free
VRAM, safe even while other jobs share the GPU) and writes
`cache/node_profile.json`. Any model `batch_size` in the config can then be
set to `auto` to use the profiled optimum; plain integers keep working as
before. Transcription resolves per-model optima (`transcription.<model>`),
which matters: on one test node `tone` was 29x faster at batch 64 while
`giga_rnnt` was fastest at batch 1. See `report.md` for measurements.

---

## Audio Quality

Balalaika avoids silent quality degradation:

- `preprocess.chunk_format: auto` preserves the source container when chunking
  (`.flac` input produces `.flac` chunks, `.wav` stays `.wav`).
- Set `preprocess.chunk_format` to `flac`, `wav`, `mp3`, `ogg`, or `opus` only
  when you explicitly want a fixed output container.
- Loudness normalization writes FLAC/WAV through `soundfile` as lossless
  containers. Lossy formats are handled by `torchaudio.save`.
- Denoising uses a dynamic ONNX export of ClearerVoice-Studio
  `MossFormer2_SE_48K`, converts clips to 48 kHz mono, and overwrites audio
  in place before export.
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
| [Spectra-0 anti-spoofing](https://huggingface.co/lab260/spectra_0) (ONNX) | removes generated / spoofed speech |
| [onnx-asr](https://github.com/istupakov/onnx-asr) | ASR models and optional TensorRT |
| RUPunct | punctuation restoration |
| ruAccent | lexical stress marks |
| TryIParu | grapheme-to-phoneme conversion |
| [ClearerVoice-Studio MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K) (ONNX export in pipeline) | denoising / speech enhancement |

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
| **[Spectra-0](https://huggingface.co/lab260/spectra_0)** | generated / spoofed speech detection |
| **[MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)** | 48 kHz denoising / speech enhancement via ONNX Runtime |

---

## License

See [LICENSE](LICENSE).
