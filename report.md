# Balalaika Pipeline Performance Report

Date: 2026-06-11 (two passes; second pass = §4.10–4.13) · Branch: `claude` ·
All numbers measured on this node.

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

### 4.10 Transcription stage restructure: shared decode (stage 7) — 1.39×, second pass 2026-06-11

The stage ran its 5 ASR models strictly sequentially, and **each model
re-read and re-decoded every audio file from disk** through its own
DataLoader (`src/utils/datasets/transcription.py`), plus respawned GPU
workers per model. The first `consensus_num` models always process every
pending file (the consensus filter only engages after that many models
exist), so they now run as one **shared-decode group**: each file is read,
decoded and resampled once per GPU worker, and every grouped model consumes
the same macro-batch through per-model sub-batching (each model keeps its
own `batch_size: auto` value; per-chunk re-trimming keeps tensor shapes
identical to the sequential flow). The remaining models keep the exact
sequential flow because their pending sets depend on the previous model's
outputs. Work shards now carry per-path "which models still need me"
annotations, so resume semantics are preserved per model.

Knob: `transcription.share_decode` (default True; False restores the old
flow bit-for-bit). VRAM cost: the grouped models' sum on each GPU — on very
tight VRAM set False.

Measured, 250 real files × 5 models + ROVER, GPU 1 shared with training:

| Leg | Wall | Note |
|---|---|---|
| Harness before (warm TRT) | 243.6 s | `transcription.stage`, repeat 2 |
| Harness after (2 repeats) | 168.5 / 184.2 s | **1.38×** |
| Same-conditions A/B, sequential | 254.2 s | `share_decode: false` |
| Same-conditions A/B, shared | 183.5 s | **1.39×** |

On this SSD-and-16-loader-workers node the saving is mostly CPU decode +
worker respawn; on the production HDD the same change removes **2 of 3 full
passes of random reads** for the grouped models (decode count measured in
tests: `2·N+odd → N+odd` for 3 models with consensus 2), which is the
dominant cost there. Avg RSS during the stage also dropped 13.0 → 11.8 GB.

**Proof of unchanged outputs**: `tests/test_transcription_share_decode.py`
pins byte-identical sidecars between both modes with deterministic fake
models (orchestration-level equivalence: pending-set union, sub-batching,
timestamps, consensus skipping, resume). On the real 250-file set with real
models: 1749 = 1749 sidecars, **1742/1749 byte-identical, including every
`_rover.txt` (0/250 differ — the stage's downstream product is unchanged)**.
The 7 diffs (0.4 %) are single-token flips confined to one model
(giga_ctc: 5 `.tst` + 2 `.txt` of 250). Controls: each mode is internally
**deterministic** — sequential run twice → 0/1749 differ; shared run twice
→ 0/1749 differ — so this is not run noise but a stable mode-dependent
numeric divergence: three ORT/TRT sessions sharing one GPU process select
different kernels/workspaces for giga_ctc, shifting float rounding on
knife-edge frames. Same class of bounded divergence as the already-accepted
Spectra-0 TRT fp16 (§7) and G2P TF32 pins (§4.9); `share_decode: false`
restores the old flow bit-for-bit where strict reproducibility matters.
Bonus robustness fix: ASR sidecars are now written atomically (tmp+rename)
— a killed worker previously left a truncated `.txt` that resume scans
accepted forever.

### 4.11 ROVER aggregation: numba fast path — 12.2×

crowd-kit's `ROVER._align` is an O(n·m) dynamic program in pure Python with
per-cell tuple/list allocations, attrs objects, and a `deepcopy` in the
traceback, run (models−1) times per file. `src/transcription/fast_rover.py`
keeps the exact algorithm — same costs, same option-order tie-breaking,
same zero-cost deletion against empty-token edge sets, same
`(count, len(word), word)` voting — but runs the DP in a cached numba
kernel over integer word ids; only the O(n+m) traceback/voting stays in
Python.

`benchmarking/micro/bench_rover.py`, 2000 ASR-like tasks (9424 hypothesis
rows, 5 models, word-level corruptions):

| Impl | tasks/s | per file |
|---|---|---|
| crowd-kit | 70 | 14.2 ms |
| **FastROVER** | **857** | **1.2 ms** |

At 500 k files this turns ~30 min of ROVER DP (4 workers) into ~2.5 min.
**Output equality: 0/2000 aggregated strings differ**; pinned further by 13
tests in `tests/test_fast_rover.py` (randomized corpora + every tie-break
edge case). Knob: `transcription.use_fast_rover` (default True; crowd-kit
fallback also engages automatically if numba is unavailable). Worker
startup pays a one-time ~0.4 s numba cache load (`cache=True`; compile
happens once per machine).

### 4.12 BS.1770 loudness gating: bit-exact vectorization (stages 1+3)

pyloudnorm's gating loop squares every sample ~4× (75 % overlapping blocks,
fresh temp per block) and runs the gating passes as per-block Python list
comprehensions. `_integrated_loudness_fast` in
`src/preprocess/audio_postprocessing.py` squares each channel once and
keeps every reduction's element order/length/dtype identical (numpy pairwise
summation depends only on those), including the original's per-block
`int()` truncation of block bounds — so the LUFS float, and therefore the
normalized audio bytes, are **identical** (the §5 byte-identical bar
still holds). Falls back to the stock meter on anything unexpected.

Honest numbers: the measure step is dominated by the two scipy `lfilter`
K-weighting passes, so the end-to-end win is modest — 1.22× on 100 real
clips, 1.15× on 15–60 s chunks, plus removed allocation churn in the
stage-1 fused path. 17 tests (`tests/test_fast_loudness.py`) pin LUFS
equality and array equality on real + synthetic edge-case audio; 0/100
mismatches in `benchmarking/micro/bench_loudness.py`. A further ~2×
(fusing both biquads into one `sosfilt` pass) is possible but **not**
bit-exact — left as a knob-gated option, not taken.

### 4.13 Cross-stage quick wins (second pass)

| Fix | Measured | Where it shows |
|---|---|---|
| Dead `import torch` removed from `src/utils/utils.py` | 2.5 s / 626 MB → 0.2 s / 24 MB per process | every CPU-only stage (download, both filters, collate, webdataset, report) **and each of their spawned workers** |
| Duration probes: soundfile-first for wav/flac/ogg/opus (`safe_audio_duration`) | 4.6 ms → 0.07 ms per file (**66×**), 0/250 value mismatches | duration backfills in transcription/distillmos/music_detect; probe workers no longer import torch |
| DistillMOS per-shard re-probe removed (`assume_sorted`) | probe pass gone (6 → 0 occurrences in stage logs), 48.6 → 44.2 s on 250 files; **scores bit-identical: max abs delta 0.0 across all 250 files** (real model, GPU 0) | stage 5 worker startup per shard (~24 s per 10 k-file shard on the HDD) + removes a racy shared JSON cache; knob `distillmos.sort_in_loader: true` restores old behavior |
| music_detect durations hoisted out of workers (per-path values ride shard annotations, exact `str(float)` round-trip) | removes a full-CSV read (+ possible rewrite) **per claimed shard** inside GPU workers | stage 4 (pattern mirrors distillmos; not end-to-end runnable on this node — model weights absent; round-trip pinned by tests) |
| `unprocessed_paths` narrow read (header sniff + `usecols=[filepath, column]`; missing column now skips the body read entirely) | 17-col 2 M-row fixture: 5.09 s → 4.10 s and ~0.5 GB less transient RAM; 10-col fixture 4.78 → 4.43 s; identical pending sets | every stage startup; wins grow with CSV width (production CSVs carry text columns) |
| Collate main-CSV read switched to `fast_read_csv` (pyarrow) | ~3.4× on this read at 2 M rows (per §4.1 measurements) | stage 12 startup |
| WebDataset workers now receive **only their chunk's metadata** | full dict is GBs at 2 M rows, was pickled once per worker | stage 13; **this corrects §8 below** — the prior report claimed it was already done, but commit 5c124e5 only vectorized the load; now actually implemented |
| Spawned workers honor the configured log level (`BALALAIKA_LOG_LEVEL` exported; DataLoader `worker_init_fn`) | emitted debug line 36.5 µs → 0.9 µs suppressed; stage/loader workers previously ran loguru's default DEBUG-to-stderr sink | every per-file `logger.debug` in dataset `__getitem__` at production scale (≈2 CPU-min per million files per call site, plus stderr noise) |

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

1. ~~**Transcription stage restructure**: decode/resample each file once and
   share across the ASR models.~~ Done — see §4.10 (1.39×). Remaining ideas
   for stage 7: keep DataLoader workers alive across shards (GPU idles a few
   seconds per 10 k-file shard), and the **RNNT decode loop** — giga_rnnt runs
   200+ sequential batch-1 ONNX decoder calls per file inside onnx-asr (why it
   sits at 7 it/s vs ~60 for CTC and sets the stage critical path); fixing it
   means batched/stateful decode inside the onnx-asr Kaldi adapter (upstream
   surgery, high effort, high payoff).
2. **Sortformer/SmartTurn (stage 1)**: batch SmartVAD calls across segments; ORT
   IOBinding to kill per-window GPU→CPU→GPU round-trips; vectorize `_binarize`
   and spkcache compression (file:line details in `.claude/analysis/preprocess-*.json`).
   Untestable here — the ONNX model files are not on this node. The 2026-06-11
   second-pass audit additionally flagged: the ACTIVE existing_chunks+fuse path
   decodes every chunk twice; diarization decode is fully serialized with GPU
   inference (`diarization_loader_workers: 0`); raw mode accumulates all
   chunk-row dicts in RAM. Full details: `.claude/analysis/audit2_findings.json`.
3. ~~**Phonemizer**: persist the word→phoneme cache across runs and batch
   `greedy_decode` over unique words.~~ Done — see §4.9.
4. **tone batch size**: raise beyond 64 (still climbing at the sweep cap) once
   measured on an idle GPU.
5. Consider Parquet for pipeline *state* (keeping balalaika.csv as an export) —
   removes CSV parse cost entirely; bigger format decision, not taken unilaterally.
6. **Accents stage (9)** is the slowest text stage per the audit: ruAccent runs
   3-6 batch-1 ONNX calls per sentence with no batching, loads ~200 MB of
   rule-engine assets per worker, and re-runs OOV words per sentence. Batching
   inside ruAccent is upstream work; a word-level memo would change homograph
   handling (context-dependent) — needs a careful equivalence study first.
7. **Antispoofing decode** (stage 6): the loader decodes + preemphasizes the
   full clip then keeps a random 4.04 s window — a seek-bounded read would cut
   ~3× of that decode CPU; exactness depends on container seek semantics
   (bit-exact for PCM wav/FLAC, needs verification per format).
8. **Collate RAM**: stage 12 holds every sidecar text ~3× over (records list →
   DataFrame → Arrow) while writing the parquet; chunked assembly would cap
   peak RSS on low-RAM nodes.
9. Remaining smaller audit findings (with verifier verdicts where the budget
   allowed) are preserved in `.claude/analysis/audit2_findings.json`.

## 10. How to reproduce every number

```bash
source .dev_venv/bin/activate
python -m benchmarking.micro.make_fixtures                  # once (~2 min)
python -m benchmarking.micro.bench_csv_ops --label check    # §4.1 table
python -m benchmarking.micro.bench_collate --label check    # §4.2
python -m benchmarking.micro.bench_g2p --make-fixtures      # §4.9 (once)
python -m benchmarking.micro.bench_g2p --impl fast --label check   # §4.9
python -m benchmarking.micro.bench_rover --label check --impl both # §4.11 (also proves 0 mismatches)
python -m benchmarking.micro.bench_loudness --label check          # §4.12 (also proves 0 mismatches)
TARGET=transcription.stage DATASET=cache/bench_sample/audio NUM_SAMPLES=250 \
  REPEATS=2 GPU_IDS=1 benchmarking/run_benchmark.sh                # §4.10 stage legs
python -m benchmarking.warmup --config_path configs/config.yaml   # §2 (per node)
python -m pytest tests/ -q                                  # 120+ behavior tests
# stage-level before/after harness runs: benchmarking/reports/2026*/report.json
```
