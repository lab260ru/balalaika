# Balalaika Pipeline Performance Report

Date: 2026-06-11 · Branch: `claude` · All numbers measured on this node.

## 1. Node and environment

| Component | Value |
|---|---|
| GPUs | 2× NVIDIA RTX 4060 Ti 16 GB (driver 575.64.03, CUDA 12.9) — **both shared with a running training job during all GPU measurements** |
| CPU | 2× Xeon E5-2690 v3, 48 threads, AVX2 (no AVX-512), 2 NUMA nodes |
| RAM | 31 GB (low — treated as a first-class constraint) |
| Working env | `.dev_venv`: torch 2.8.0+cu128, onnxruntime-gpu 1.26, TensorRT 10.16, torchcodec 0.7 |

GPU numbers below are **depressed and noisy** because your training job occupied
both GPUs (61–100 % util, 11–14 GB VRAM) throughout. Relative comparisons and
batch-size *shapes* are still informative; absolute throughput will be higher on
an idle GPU. CPU-side numbers are clean.

### Environment bugs fixed (these blocked the pipeline on this node entirely)

1. **`create_dev_env.sh` installed a CUDA-13 nightly onnxruntime-gpu + `tensorrt-cu13`** —
   cannot load on any CUDA 12.x driver node (needs driver ≥ 580). Now installs the
   stable CUDA-12 ORT.
2. **`pip install tensorrt-cu12` now resolves to TensorRT 11**, which ships
   `libnvinfer.so.11`; ORT links `libnvinfer.so.10`, so the TRT provider *silently*
   fell back to CPU. Pinned `tensorrt-cu12==10.*`; verified both CUDA and TRT
   execution providers actually load and run.
3. **`requirements_dev.cuda128.txt` was a `pip freeze` of a system Python** — it
   pinned `python-apt`, `systemd-python`, `ubuntu-pro-client`, etc., and was not
   installable anywhere. Regenerated as a real lockfile from the verified working
   venv (CPU `onnxruntime` deliberately excluded so the GPU build owns the module).
4. **`src/stage_runner.sh` lacked the nvidia/TensorRT `LD_LIBRARY_PATH` block** that
   `base.sh` builds, so any stage launched via its `*_yaml.sh` wrapper ran without
   CUDA/TRT ONNX providers. Also had CRLF line endings — `bash -n` failed on the
   committed file. Both fixed.

## 2. Warmup script: per-node batch-size autotuning (your request #1)

New: **`python -m benchmarking.warmup --config_path configs/config.yaml`**

- Probes each tunable model with a doubling batch ladder, measures items/s and
  audio-seconds/s, watches free VRAM (safe to run while another job owns most of
  the GPU — it stopped itself at 85 % of free memory during validation), detects
  throughput plateaus, and writes **`cache/node_profile.json`** plus a
  human-readable `cache/node_profile.suggested.yaml`.
- Run it once per machine — that's the "use it on other nodes" story. Profiles
  carry hostname/GPU/`"contended"` flags so a profile taken on a busy GPU is
  marked as such.
- Probed: DistillMOS, Spectra-0 (auto-downloads), MossFormer2 (if ONNX present),
  every configured onnx-asr model, RUPunct, music_detect (if its package is
  installed). Skipped-by-design with documented reasons: Sortformer (stateful
  streaming, batch is structurally 1), SmartTurn (called per segment), ruAccent
  (no batch API).
- Stages now accept **`batch_size: auto`** (resolved by `src/utils/node_profile.py`;
  plain integers behave exactly as before). Transcription resolves
  **per-model** keys (`transcription.giga_rnnt`, …) before the flat fallback.

### Measured on this node (GPU shared, 8 s clips)

| Model | bs=1 | best | best bs | note |
|---|---|---|---|---|
| DistillMOS | 57.8 it/s | 60.5 it/s | 2 | GPU already saturated by the other job |
| Spectra-0 | 32.1 it/s | 37.0 it/s | 8 | VRAM guard stopped the sweep at 16 |
| gigaam-v3-e2e-ctc | 31.8 it/s | **61.8 it/s** | 8 | degrades again at 16/32 |
| giga_ctc | 38.2 it/s | **60.8 it/s** | 8 | |
| giga_rnnt | **7.3 it/s** | 7.3 it/s | **1** | batching *hurts* (sequential RNNT decode) |
| vosk | 7.4 it/s | 10.3 it/s | 2 | |
| tone | 0.42 it/s | **12.1 it/s** | **64+** | **29× from batching, still climbing at the cap** |
| RUPunct | 48.8 texts/s | **252 texts/s** | 64 | motivated the stage-8 batching below |

**Key insight:** one flat `transcription.batch_size` is badly wrong for this model
mix — `tone` wants ≥ 64 while `giga_rnnt` wants 1. With a node profile present,
each ASR worker now picks its own batch size.

## 3. Pipeline-wide analysis (your request #2)

Eleven parallel reviewers mapped every stage file (raw per-area findings with
file:line references are kept in `.claude/analysis/*.json`). The dominant cost
pattern across the whole pipeline was **state/IO management, not model math**:

- Every stage start: discover paths → ensure CSV → absorb partials → compute
  pending. On a 2 M-row `balalaika.csv` that was ~80 s *per stage* before this
  work (scales linearly; production datasets are larger).
- During stages: the periodic flush re-read all partials, re-read the full main
  CSV, made a **full byte-copy backup**, normalized paths row-by-row through
  `Path()` objects under per-row tqdm, and rewrote the whole CSV.
- Sidecar/file discovery: five `rglob` passes per tree scan; 2 stat calls per
  audio file per text stage; two `glob` calls per file in webdataset export;
  13 `open()` attempts per file in collate.

## 4. Optimizations applied, with before → after measurements (requests #3–#4)

### 4.1 CSV state layer (`src/utils/csv_manager.py`) — affects *every* stage

2 M-row fixture (10 columns, 4×50 k-row partials), `benchmarking/micro/bench_csv_ops.py`,
avg of 3 runs:

| Operation | Before | After | Speedup | What changed |
|---|---|---|---|---|
| `load_main_csv` | 10.50 s | 3.06 s | **3.4×** | pyarrow parser + vectorized path normalize |
| `atomic_write_csv` | 14.98 s | 3.63 s | **4.1×** | pyarrow writer + hardlink `.bak` instead of byte copy |
| periodic flush cycle | 33.83 s | 14.73 s | **2.3×** | all of the above |
| upsert + drop_missing | 44.98 s | 15.96 s | **2.8×** | scandir name-set cache instead of per-row `stat` |
| `unprocessed_paths` | 15.28 s | 4.78 s | **3.2×** | vectorized done-set, no per-row tqdm |
| discover paths from CSV | 20.62 s | 4.57 s | **4.5×** | string-ops dedupe, no `Path()` per row |
| `read_partial_csvs` | 0.71 s | 0.38 s | 1.9× | pyarrow parser |
| `get_audio_paths` (50 k tree) | 0.80 s | 0.55 s | 1.5× | one `os.walk` instead of five `rglob`s |

Net effect at 2 M rows: stage-start overhead ~80 s → ~25 s; flushes during a stage
2.3× cheaper. Savings scale linearly with dataset size (≈ ×5 at 10 M rows).

Knob: `BALALAIKA_CSV_ENGINE=c` forces the old pandas parser/writer — use on very
low-RAM nodes (pyarrow conversion briefly holds a second copy of the table) or
for byte-conservative output (pyarrow quotes string fields per RFC 4180; values
round-trip identically — verified — but raw bytes differ).

Documented tradeoff: pyarrow's float parser can differ from pandas' by 1 ULP on
full-precision floats (e.g. `…015777` vs `…015775`). Bounded, non-cumulative,
orders of magnitude below score noise; not observable for the `round(x, 4)`
values stages write.

### 4.2 Collate (stage 12)

| Operation | Before | After | Speedup |
|---|---|---|---|
| ASR-consistency column, 1 M rows | 20.07 s | 10.39 s | **1.9×** (row-wise `df.apply` → factorized pairwise compare) |
| Sidecar collection, 50 k files | 14.99 s | 0.85 s | **17.6×** (per-file futures + 13 opens → slab map + one scandir per dir) |

Equivalence pinned by `tests/test_collate_consistency.py` (vectorized vs original
row-wise on edge cases: NaN, numeric cells, ties, unicode, vosk/vosk_small dedup).

### 4.3 WebDataset export (stage 13)

- Metadata load: `iterrows` (~46 s per 500 k rows, extrapolated from a 50 k
  sample) → vectorized records (**1.5 s for the full 500 k including read**).
- Sibling sidecar discovery: two `glob` calls per *file* → one cached scandir per
  *directory*. Round-trip correctness pinned by `tests/test_to_webdataset.py`
  (including a shared-prefix decoy chunk).

### 4.4 Punctuation (stage 8) — 5× from batching

RUPunct was invoked one file at a time. Files are now processed in
`batch_size` slabs through a single pipeline call (`batch_size: auto` supported;
profile says 64 on this node → measured 48.8 → 252 texts/s, **5.2×**).
**Proof of unchanged outputs:** batched vs per-file outputs are character-identical
on 60 varied Russian texts (run on the real RUPunct_big on GPU). Per-file fallback
inside each slab keeps one bad file from failing its neighbors; stage status now
counts actually-produced sidecars.

### 4.5 DistillMOS filter (stage 5.5)

Every deletion worker loaded the **entire** main CSV (×8 wall-clock and ×8 RAM —
an OOM risk at 31 GB). The parent now reads the CSV once and ships per-shard
`(path, mos, duration)` tuples. With `load_main_csv` at 3 s / 2 M rows, that's
~21 s and ~×8 peak RAM saved per run at 2 M rows, much more at production scale.
Behavior pinned by `tests/test_distillmos_filter.py`.

### 4.6 Denoising (stage 11)

Each worker built a throwaway **CPU InferenceSession of the full MossFormer2
model** just to read the input tensor name (seconds of load + optimization per
worker). Replaced with an ONNX graph-metadata read (milliseconds).

### 4.7 Text-stage pending scans (stages 7–10 startup)

Two `stat` calls per audio file → one scandir per directory (`DirNameCache` in
`src/utils/sidecars.py`). Identical pending sets verified on a 50 k fixture.
Warm-cache SSD timing is parity; the win is on cold caches and the production
HDD (`/mnt/hdd_6tb_1`), where per-file stats seek and directory reads batch.

### 4.8 Dataloaders

`pin_memory` enabled for the two torch-fed loaders (DistillMOS, music_detect):
measured 10 MB batch H2D 2.54 ms → 1.69 ms, and it makes the stages' existing
`non_blocking=True` an actual async copy instead of a silent synchronous one.
ORT-fed loaders intentionally left unpinned (inputs convert to numpy; pinning
does nothing there). Cost: a few MB of page-locked RAM per prefetched batch.

### 4.9 Phonemizer / TryIParu G2P (stage 10) — batched OOV decode

Stock `tryiparu.G2PModel` greedy-decodes every out-of-dictionary word
one-at-a-time: ≤63 sequential decoder passes per word, a GPU sync per step,
fresh CPU tensors per token, and a `torch.compile(mode="max-autotune")` call
that costs ~2.8 s per worker yet compiles nothing (it wraps `forward`, while
inference goes through the uncompiled `.encode`/`.decode` bound methods).
`src/phonemizer/fast_g2p.py` keeps the weights, tokenizer and rules and fixes
only the mechanics; the stage now also reuses rule splits per unique word and
persists OOV decodes across runs/workers (`oov_cache_path`, weights-fingerprint
keyed, fcntl lock-merge so concurrent workers can't clobber each other).

`benchmarking/micro/bench_g2p.py`, 30 ASR-like fixture texts (24×250 words @2%
OOV, 4×200 @15%, 2×20 — built from tryiparu's own 398k-word dictionary plus
deterministic pseudo-words), GPU = RTX 4060 Ti running a training job:

| Metric | Stock GPU | Stock CPU | Fast GPU | Fast CPU |
|---|---|---|---|---|
| OOV decode, ms/word | 123.8 | 82.0 | **5.7** | **8.4** |
| 30 fixture texts | 18.2 s | 12.5 s | **4.7 s** | **3.9 s** |
| mean / max per text | 0.61 / 1.74 s | 0.42 / 1.23 s | 0.16 / 0.26 s | 0.13 / 0.24 s |
| worker init | 8.2 s | 5.9 s | 2.5 s | 2.3 s |

On a 1200-word fresh-OOV batch the gap widens (per-text batches are small):
stock 140 ms/word vs fast 3.8 ms/word = **37×**; CPU 86.6 → 8.2 ms/word =
10.5×. Init is faster because the no-op compile is gone and the dictionary CSV
parses once per node into `cache/g2p_dict.pkl` (0.94 s pandas → 0.25 s pickle
per worker). A fully-cached text drops 9 ms → ~2 ms via the rule-split memo.

**Proof of equivalence**: 0/1200 token mismatches vs the stock fp32 reference
on CPU and on GPU at batch sizes 64/5/1 (run under production ambient flags);
0/30 fixture texts differ across all impl×device combinations; identical
`ValueError` (message and words-cached-before-raise semantics) on oversize
words; the csv-module dictionary load compared equal to pandas
`set_index().to_dict()` on all 397,782 keys. Pinned by 11 tests in
`tests/test_phonemizer_fast_g2p.py`. Decode runs with TF32 saved/disabled/
restored around it, because under the pipeline-wide TF32 default even the
STOCK model flips argmax ties on knife-edge pseudo-words (3 of 1200 flip when
toggling TF32 alone, batch size 1) — fp32 is the only stable reference, and at
d_model=128 it is free (3.3 vs 3.4 ms/word measured).

Two deliberate divergences, both pinned by tests: (1) a word encoding to
exactly 62 BPE ids *crashes* stock (its zero-length pad segment is a float32
empty tensor; `torch.cat` promotes the whole encoder input and
`nn.Embedding` rejects it) — FastG2P decodes it normally, so such files
produce phonemes instead of erroring; (2) TF32 tie-flips on synthetic
pseudo-words under the old always-on TF32 regime no longer occur (fp32 is
enforced during decode).

A 28-agent adversarial review (4 lenses × verify-each-finding) then hardened
the implementation: TF32 flip scoped to the decode call instead of leaking
process-global state; GPU sync (`finished.all()`) polled every 4th step
instead of every step (post-`<eos>` tokens are truncated, outputs identical);
cache fingerprints switched from `(size, mtime)` to `(size, blake2b)` (mtime
survives `cp -p`/rsync across different files); dict-cache build serialized
under flock (was a thundering-herd CSV parse across workers); wrong-typed
cache payloads ignored instead of bricking worker init; OOV pending list not
accumulated when persistence is off; inline flush backs off geometrically;
`device: cpu` no longer dies on CPU-only nodes (and runs ONE pool, not one
per GPU); `_rover_phonemes.txt` written atomically (a killed worker used to
leave a truncated sidecar that resume scans accept forever — pre-existing
bug); `_rules_cache` RAM-bounded. Rejected as not-issues after verification:
per-device OOV cache keys, decoder-input preallocation, sorting words by
length before chunking (changes error-path cache semantics for <1 batch of
typical work).

Low-resource note: `phonemizer.device: cpu` runs the whole stage on CPU at
~8 ms/word batched — on this node that beats the *busy* GPU and uses zero
VRAM; keep `cuda` (default) when GPUs are idle.

## 5. End-to-end validation on real audio (proof nothing broke)

250 real Russian wavs (OpenSTT), old commit `410de9b` in a worktree vs HEAD,
same venv, benchmark harness copies per repeat:

| Check | Result |
|---|---|
| Crest stage output rows | 247 = 247 (same 3 files deleted by threshold) |
| `crest_factor` values | identical |
| `total_duration` values | identical |
| `loudness_normalized` markers | identical |
| **Loudness-normalized audio files** | **byte-identical MD5 for all 250 files** |
| Behavior tests (written against the ORIGINAL code, then run on HEAD) | 76 passed |

Stage wall-clock at this scale is parity (~7 s steady-state both sides — fixed
~4 s of torch import dominates 250 files); the CSV-layer wins above only become
visible at realistic row counts, which the micro-benchmarks quantify.

DistillMOS GPU stage before/after: see §7 (run while GPUs were shared).

## 6. Bugs found and fixed (request P.S.)

Beyond the four environment bugs in §1:

| # | Bug | Severity | Fix |
|---|---|---|---|
| 1 | `run_per_gpu_pool` returned a 3-tuple on the empty-items path, 2-tuple otherwise — every caller unpacks 2, so a stage with zero pending items **crashed** | high | consistent `(error_count, error_details)` |
| 2 | Pool error reporting attributed every failure to the *last submitted* item (closure over loop var) | medium | future→item map |
| 3 | `process_token` returned `None` for unknown punctuation labels → `TypeError` crash for the whole file in stage 8 | medium | unknown labels return the token unchanged |
| 4 | `smart_turn/inference.py` used `np.*` with no numpy import (dormant `NameError`) | low | import added |
| 5 | Loudness stage status lost the discovery-time skipped count to variable shadowing | low | renamed + summed |
| 6 | `cpu_affinity: "0-80"` on a 48-CPU node (silently clamped; crashes/mis-pins on other machines) | medium | default `""` + documented NUMA guidance |
| 7 | Benchmark harness: `--batch-size-override` was a silent no-op for **all transcription targets** (wrote into nonexistent `giga`/`vosk` subsections) and for distillmos | high | writes the keys stages actually read; pinned by `tests/test_benchmark_targets.py` |
| 8 | Harness referenced deleted stages (`nisqa_process`, `diarization`, `silence_detect`); `pipeline.base` missing current stages; no antispoofing/denoising targets | high | targets realigned with `base.sh`; new targets added |
| 9 | `python -m benchmarking.cli` was a no-op (no `__main__` guard) — this is why the "before" benchmark legs initially produced no output | medium | entrypoint added |
| 10 | Benchmarks with `runtime.audio_paths_source: csv` no-opped on fresh dataset copies (no balalaika.csv yet → "No audio files found") | medium | harness forces `auto` on benchmark copies |
| 11 | Crest audit fallback probed **every** audio file serially when workers wrote nothing (hours, for a report-only number) | medium | bounded 2000-file sample + extrapolation |
| 12 | `gpu.py` docstring documented a `gpu_count` helper that doesn't exist | low | docstring fixed |
| 13 | Config comment said flushes happen every "10 000 rows" while the value is 100 000 | low | comment fixed |

Investigated and **rejected** (not bugs): `format_timestamps` trailing-word index
(uses the last token's timestamp of the same word — semantically right);
`to_webdataset` worker-count overflow claim (ceil math caps at `num_workers`).

Known issues documented but deliberately **not** changed (behavior-altering;
your call): the `peak` parameter of loudness normalization is accepted but
ignored (`_ = peak`) — implementing true-peak limiting would change produced
audio; Spectra-0/antispoofing uses a *random* crop so scores differ between
reruns (a fixed/center crop would be deterministic but changes scores);
`vosk`/`vosk_small` share one sidecar suffix and would overwrite each other if
both were configured.

## 7. GPU stage validation + TensorRT

### DistillMOS stage end-to-end (old commit vs HEAD, 250 real files, GPU 1)

| | Before (410de9b) | After (HEAD) |
|---|---|---|
| Wall time | 44.9 s | 45.6 s (parity — GPU was 73–85 % busy with the other job) |
| **DistillMOS scores** | — | **max abs delta 0.000000 across all 250 files** |

The GPU data path (loader, resampling, batching, model, CSV writes) produces
bit-identical scores after all changes.

### TensorRT conversion experiment: Spectra-0 (your request #3)

Fixed-shape input makes it the ideal TRT candidate. Measured on the *shared*
GPU, batch 8 × 64 600 samples:

| Engine | ms/batch | items/s | Speedup |
|---|---|---|---|
| CUDA EP fp32 (old default) | 224.2 | 35.7 | — |
| **TensorRT EP fp16** | **52.7** | **151.8** | **4.25×** |

Score fidelity: logit max abs diff **0.0085** (mean 0.0045); per-clip spoof
margins shift by ~0.005–0.012 — an order of magnitude below the stage's own
run-to-run variation from its random crop. **Enabled in config**
(`separation.antispoofing.use_tensorrt: True`) with a documented opt-out.
Engine build is one-time (~8 min, cached under `runtime.trt_cache_path`);
`create_session` now pins a dynamic-batch TRT profile (1..batch_size) so
trailing partial batches don't each trigger another build.

TRT for MossFormer2/ASR was deliberately *not* swept here: those models build
one engine per (batch, length) profile — engines rebuild at pipeline runtime
anyway, and the warmup sweep uses CUDA EP so it finishes in minutes, not hours.

## 8. Low-resource tradeoffs (request: "every type of hardware")

- **Low RAM**: the distillmos-filter rework removes an ×N-workers CSV RAM
  multiplier (the worst spike). `BALALAIKA_CSV_ENGINE=c` trades CSV speed for a
  smaller peak during reads/writes. `csv.flush_every_rows` can be raised to trade
  crash-freshness for fewer rewrite cycles. webdataset workers now receive only
  their shard's metadata instead of the full dict.
- **Low CPU**: pyarrow CSV ops use all cores but degrade gracefully to single
  core; the scandir caches *reduce* syscall counts rather than parallelize, so
  they help weak CPUs most. `cpu_affinity` now defaults to off (no stale-range
  crashes on small machines).
- **Single GPU**: everything runs with one visible device; the warmup profile's
  VRAM guard adapts batch sizes to whatever memory the node actually has — that,
  not hardcoded batch sizes, is the portability mechanism. `batch_size: auto` +
  a per-node `cache/node_profile.json` is the recommended setup.
- **HDD datasets**: the scandir-cache changes (sidecars, collate, webdataset,
  drop_missing) replace per-file random stats with sequential directory reads —
  this is the dominant win on spinning disks.

## 9. Recommended next steps (analyzed, not applied — need either model files
absent on this node or larger refactors)

1. **Transcription stage restructure**: decode/resample each file once and share
   across the 5 ASR models (currently 5 full decodes per file), and keep workers
   alive across models instead of respawning per model. Largest remaining win in
   the pipeline (decode is ~5× redundant), but a structural change to stage 7.
2. **Sortformer/SmartTurn (stage 1)**: batch SmartVAD calls across segments; ORT
   IOBinding to kill per-window GPU→CPU→GPU round-trips; vectorize `_binarize`
   and spkcache compression (file:line details in `.claude/analysis/preprocess-*.json`).
   Untestable here — the ONNX model files are not on this node.
3. ~~**Phonemizer**: persist the word→phoneme cache across runs and batch
   `greedy_decode` over unique words.~~ Done — see §4.9.
4. **tone batch size**: raise beyond 64 (still climbing at the sweep cap) once
   measured on an idle GPU.
5. Consider Parquet for pipeline *state* (keeping balalaika.csv as an export) —
   removes CSV parse cost entirely; bigger format decision, not taken unilaterally.

## 10. How to reproduce every number

```bash
source .dev_venv/bin/activate
python -m benchmarking.micro.make_fixtures                  # once (~2 min)
python -m benchmarking.micro.bench_csv_ops --label check    # §4.1 table
python -m benchmarking.micro.bench_collate --label check    # §4.2
python -m benchmarking.micro.bench_g2p --make-fixtures      # §4.9 (once)
python -m benchmarking.micro.bench_g2p --impl fast --label check   # §4.9
python -m benchmarking.warmup --config_path configs/config.yaml   # §2 (per node)
python -m pytest tests/ -q                                  # 76+ behavior tests
# stage-level before/after harness runs: benchmarking/reports/2026*/report.json
```
