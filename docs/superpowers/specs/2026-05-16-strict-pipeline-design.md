# Strict Pipeline Mode — Design Spec

**Date:** 2026-05-16
**Status:** draft

## Problem

When a pipeline stage encounters Python errors, `base.sh` does not stop. All
Python stage modules catch exceptions internally, log them, and return exit
code 0. Combined with `set -euo pipefail` this means the pipeline always
completes — silently skipping failures. The current philosophy is "resilient by
resume" (check logs, re-run), but there is no way to make the pipeline
fail-fast on errors.

## Design

### 1. Status File Contract

Each stage writes a JSON status file as its final action:

```
${BALALAIKA_LOG_DIR}/stage_<N>_status.json
```

Format:

```json
{
  "stage": 3,
  "stage_name": "preprocess_audio",
  "processed": 1234,
  "skipped": 5,
  "errors": 2,
  "error_details": [
    {"file": "audio_42.wav", "reason": "CUDA OOM during loudness"},
    {"file": "podcast_7/chunk_3.wav", "reason": "FileNotFoundError"}
  ]
}
```

- `processed` — files successfully handled.
- `skipped` — files intentionally excluded (too short, unsupported format,
  etc.). Not failures, do not trigger a stop.
- `errors` — files that hit an unexpected Python exception. Failures, trigger
  a stop in strict mode.
- `error_details` — optional array, capped at 50 entries.

A shared utility function in `src/utils/stage_status.py` handles writing:

```python
def write_stage_status(stage: int, stage_name: str, log_dir: str,
                       processed: int, skipped: int, errors: int,
                       error_details: list[dict] | None = None) -> None:
```

This is imported by each stage module.

### 2. Stage Module Changes

Every stage's `main()` must:

1. Track separate counters: `processed`, `skipped`, `errors`.
2. Distinguish intentional skips from unexpected exceptions in error handling.
3. Call `write_stage_status(...)` as the last action before returning.

Each stage currently uses a broad `except Exception as e: logger.error(...)`
pattern. These must be refactored into two paths:

- **Known skips** (bad format, too short, already processed, intentionally
  filtered) → increment `skipped`.
- **Unexpected errors** (CUDA OOM, FileNotFoundError, model crash, any
  unhandled exception) → increment `errors`, append to `error_details`.

Stages that crash before writing the status file are treated as failures by
`base.sh` (missing status file → stage did not complete).

### 3. base.sh Changes

#### New CLI flag

```
--strict    Enable fail-fast: stop the pipeline if any stage has errors.
```

Parsed alongside existing flags in the `while` loop. Default: off (current
behavior preserved).

#### New helper function

```bash
check_stage_status() {
    local s="$1"
    local status_file="${BALALAIKA_LOG_DIR:-./logs}/stage_${s}_status.json"

    if [[ "${strict_mode:-0}" != "1" ]]; then
        return 0
    fi

    if [[ ! -f "$status_file" ]]; then
        echo -e "\033[1;31m[FAIL] Stage $s: status file not found (stage did not complete)\033[0m" >&2
        exit 1
    fi

    local errors
    errors=$(python3 -c "import json; print(json.load(open('$status_file'))['errors'])")
    if [[ "$errors" -gt 0 ]]; then
        echo -e "\033[1;31m[FAIL] Stage $s: $errors error(s). Pipeline aborted.\033[0m" >&2
        echo "See: $status_file" >&2
        exit 1
    fi
}
```

#### Pipeline section

Each stage block adds `check_stage_status <N>` after `run_python`:

```bash
if stage_active 3; then
    echo "Stage 3: Preprocess — loudness normalization (BS.1770-4)"
    run_python src.preprocess.preprocess_audio
    check_stage_status 3
fi
```

### 4. Affected Files

| File | Change |
|------|--------|
| `base.sh` | Add `--strict` flag, `check_stage_status` function, call after each stage |
| `src/utils/stage_status.py` | **New.** Shared `write_stage_status()` utility |
| `src/download/download.py` | Track counters, call `write_stage_status` at end |
| `src/preprocess/preprocess.py` | Track counters, call `write_stage_status` at end |
| `src/preprocess/crest_factor_remover.py` | Track counters, call `write_stage_status` at end |
| `src/preprocess/preprocess_audio.py` | Track counters, call `write_stage_status` at end |
| `src/separation/music_detect.py` | Track counters, call `write_stage_status` at end |
| `src/separation/distillmos_process.py` | Track counters, call `write_stage_status` at end |
| `src/transcription/transcription.py` | Track counters, call `write_stage_status` at end |
| `src/punctuation/punctuation.py` | Track counters, call `write_stage_status` at end |
| `src/accents/accents.py` | Track counters, call `write_stage_status` at end |
| `src/phonemizer/phonemizer.py` | Track counters, call `write_stage_status` at end |
| `src/collate.py` | Track counters, call `write_stage_status` at end |
| `src/to_webdataset.py` | Track counters, call `write_stage_status` at end |
| `src/report.py` | Track counters, call `write_stage_status` at end |

### 5. Error Handling Semantics

| Signal | Strict mode | Default mode |
|--------|------------|--------------|
| Status file missing after stage | Stop | Continue |
| `errors > 0` in status file | Stop | Continue |
| `errors == 0` | Continue | Continue |
| Python exit code ≠ 0 | Stop (set -e) | Stop (set -e) |
| Stage skipped (not in [stage, stop_stage]) | Continue | Continue |

In default mode (no `--strict`), behavior is identical to current `base.sh`.

### 6. What Qualifies as an Error vs. Skip

Guidance for the per-stage refactor:

- **Error (increment `errors`):** CUDA OOM, ONNX session crash, file not found
  when it should exist, model load failure, multiprocessing crash, any
  unhandled `Exception` in the processing path.
- **Skip (increment `skipped`):** File too short for the stage threshold,
  unsupported audio format, already processed (resume skip), filtered out by
  crest factor / music detection (intentional removal), empty transcription
  result (valid outcome).
- **Processed (increment `processed`):** File was handled and the stage wrote
  its output successfully.

## Verification

No test suite exists. Manual verification:

1. Run a single stage with a known-bad file to trigger an error, with
   `--strict` — confirm the pipeline stops and the status file is written.
2. Run a single stage with a known-bad file, without `--strict` — confirm the
   pipeline continues and the status file is written.
3. Run a clean stage with all good files — confirm `errors == 0` and the
   pipeline proceeds.
4. Run the full pipeline with `--strict` on a small dataset — confirm it
   either completes cleanly or stops at the first real error.
