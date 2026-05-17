# Strict Pipeline Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--strict` flag to `base.sh` that stops the pipeline when any stage has processing errors, using per-stage JSON status files as the failure signal.

**Architecture:** A new shared utility `src/utils/stage_status.py` writes a JSON status file per stage. Each stage module tracks counters (processed/skipped/errors) and calls the utility at the end of `main()`. `base.sh` reads the status file after each stage and exits on errors when `--strict` is set.

**Tech Stack:** Bash (base.sh), Python with existing torch/multiprocessing/loguru stack.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/utils/stage_status.py` | **Create** | Shared `write_stage_status()` function |
| `base.sh` | **Modify** | `--strict` flag, `check_stage_status()`, call after each stage |
| `src/utils/parallel.py` | **Modify** | Return error counts from `run_per_gpu_pool` and `run_per_gpu_processes` |
| `src/report.py` | **Modify** | Single process — track counters, call `write_stage_status` |
| `src/download/download.py` | **Modify** | ThreadPoolExecutor — track via shared counters, call `write_stage_status` |
| `src/collate.py` | **Modify** | ThreadPoolExecutor — track via shared counters, call `write_stage_status` |
| `src/to_webdataset.py` | **Modify** | ProcessPoolExecutor — track via shared counters, call `write_stage_status` |
| `src/preprocess/preprocess.py` | **Modify** | Complex two-level executor — track per-file, call `write_stage_status` |
| `src/preprocess/crest_factor_remover.py` | **Modify** | mp.spawn — shared counters, call `write_stage_status` |
| `src/preprocess/preprocess_audio.py` | **Modify** | mp.spawn — shared counters, call `write_stage_status` |
| `src/separation/music_detect.py` | **Modify** | mp.spawn — shared counters, call `write_stage_status` |
| `src/separation/distillmos_process.py` | **Modify** | mp.spawn — shared counters, call `write_stage_status` |
| `src/transcription/transcription.py` | **Modify** | `run_per_gpu_processes` — use returned error info, call `write_stage_status` |
| `src/punctuation/punctuation.py` | **Modify** | `run_per_gpu_pool` — use returned error info, call `write_stage_status` |
| `src/accents/accents.py` | **Modify** | `run_per_gpu_pool` — use returned error info, call `write_stage_status` |
| `src/phonemizer/phonemizer.py` | **Modify** | `run_per_gpu_pool` — use returned error info, call `write_stage_status` |

---

### Task 1: Create `src/utils/stage_status.py` shared utility

**Files:**
- Create: `src/utils/stage_status.py`

- [ ] **Step 1: Write the module**

```python
"""Shared utility for writing per-stage status files consumed by base.sh --strict."""

import json
from pathlib import Path


def write_stage_status(
    stage: int,
    stage_name: str,
    log_dir: str,
    processed: int,
    skipped: int,
    errors: int,
    error_details: list[dict] | None = None,
) -> None:
    """Write stage_N_status.json to log_dir.

    base.sh reads this file after each stage. When --strict is set and
    ``errors > 0``, the pipeline aborts.

    ``error_details`` is capped at 50 entries to keep the file small.
    """
    if error_details is None:
        error_details = []

    capped = error_details[:50]
    status = {
        "stage": stage,
        "stage_name": stage_name,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "error_details": capped,
    }

    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stage_{stage}_status.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import src.utils.stage_status; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/utils/stage_status.py
git commit -m "feat: add stage_status shared utility for strict pipeline mode"
```

---

### Task 2: Modify `base.sh` — add `--strict` flag and `check_stage_status`

**Files:**
- Modify: `base.sh`

- [ ] **Step 1: Add `strict_mode` variable and `--strict` flag parsing**

In the defaults section (after `stop_stage=9`), add:

```bash
strict_mode=0
```

In the `while` loop, before `--help|-h)`, add:

```bash
        --strict)
            strict_mode=1; shift ;;
```

- [ ] **Step 2: Add `check_stage_status` helper function**

After the `stage_active()` function (before `# ---- pipeline`), add:

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

- [ ] **Step 3: Add `check_stage_status` after each stage**

Add `check_stage_status <N>` after each `run_python` call in the pipeline section. Example for stage 3 (apply the same pattern to all 13 stages):

Before:
```bash
if stage_active 3; then
    echo "Stage 3: Preprocess — loudness normalization (BS.1770-4)"
    run_python src.preprocess.preprocess_audio
fi
```

After:
```bash
if stage_active 3; then
    echo "Stage 3: Preprocess — loudness normalization (BS.1770-4)"
    run_python src.preprocess.preprocess_audio
    check_stage_status 3
fi
```

Apply `check_stage_status <N>` to every stage (0 through 12).

- [ ] **Step 4: Verify base.sh parses --strict correctly**

```bash
bash base.sh --help 2>&1 | head -5
```

Expected: Shows usage with stages listed.

- [ ] **Step 5: Commit**

```bash
git add base.sh
git commit -m "feat: add --strict flag and check_stage_status to base.sh"
```

---

### Task 3: Modify `src/utils/parallel.py` — return error info from shared runners

**Files:**
- Modify: `src/utils/parallel.py`

This task adds error tracking to `run_per_gpu_pool` and `run_per_gpu_processes` so stages 6-9 can use the returned values directly.

- [ ] **Step 1: Modify `run_per_gpu_pool` to return `(error_count, error_details)`**

In `run_per_gpu_pool`, find the `except Exception as exc:` block inside the future submission loop (approximately line 101-102). Add error tracking variables at the top and modify the exception handler:

At the top of `run_per_gpu_pool` (after the docstring), add:

```python
def run_per_gpu_pool(
    items: list,
    work_fn: Callable,
    initializer: Callable | None = None,
    initargs: tuple = (),
    num_gpus: int | None = None,
    num_workers_per_gpu: int = 1,
    desc: str = "Processing",
) -> tuple[int, list[dict]]:
```

Change the return type. At the end of the function, before the final `logger.success`, add error counting:

Before the `try` block that creates executors, add:

```python
error_count = 0
error_details: list[dict] = []
```

In the `except Exception as exc:` block (approximately line 101):

Before:
```python
                except Exception as exc:
                    logger.error(f"{desc}: task failed: {exc}")
```

After:
```python
                except Exception as exc:
                    logger.error(f"{desc}: task failed: {exc}")
                    error_count += 1
                    error_details.append({"item": str(item), "reason": str(exc)})
```

At the end of the function, replace `return` with:

```python
    return error_count, error_details
```

- [ ] **Step 2: Modify `run_per_gpu_processes` to return `(error_count, error_details)`**

In `run_per_gpu_processes`, change the return type. This function uses `mp.Process` — errors inside workers are caught by the worker's own `try/except`. We rely on the caller (transcription.py) to track errors inside its worker. But we also wrap the function to handle any uncaught exceptions:

At the top, change return type:

```python
def run_per_gpu_processes(
    run_worker: Callable,
    num_gpus: int,
    args: tuple = (),
    join: bool = True,
) -> tuple[int, list[dict]]:
```

At the end, return:

```python
    return 0, []
```

(The actual error tracking happens inside the worker function — transcription.py's `run_worker`. `run_per_gpu_processes` always returns 0 here because workers communicate errors through shared counters passed in `args`.)

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "from src.utils.parallel import run_per_gpu_pool, run_per_gpu_processes; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/utils/parallel.py
git commit -m "feat: return error counts from run_per_gpu_pool and run_per_gpu_processes"
```

---

### Task 4: Modify `src/report.py` — single-process stage

**Files:**
- Modify: `src/report.py`

- [ ] **Step 1: Add import and call `write_stage_status` at end of `main()`**

Add import at top:

```python
from src.utils.stage_status import write_stage_status
```

At the end of `main()`, after the final `logger.success` / `print` lines (around line 195), add:

```python
    write_stage_status(
        stage=12,
        stage_name="report",
        log_dir=args.log_dir or "./logs",
        processed=1,
        skipped=0,
        errors=0,
    )
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/report.py', doraise=True); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/report.py
git commit -m "feat: add stage status reporting to report stage"
```

---

### Task 5: Modify `src/download/download.py` — ThreadPoolExecutor stage

**Files:**
- Modify: `src/download/download.py`

- [ ] **Step 1: Add import and error tracking in `main()`**

Add import at top:

```python
from src.utils.stage_status import write_stage_status
```

In `main()`, after the client initialization section, track counters:

Add after `args = parser.parse_args()` (approximately line 149):

```python
    processed = 0
    errors = 0
    error_details: list[dict] = []
```

In the per-podcast download loop (the `try/except Exception` block around line 178-189), modify the exception handler:

Before:
```python
                except Exception as e:
                    logger.error(f"Error downloading podcast: {e}")
```

After:
```python
                except Exception as e:
                    logger.error(f"Error downloading podcast: {e}")
                    errors += 1
                    error_details.append({"podcast": str(url), "reason": str(e)})
```

After the `if success:` block, track processed:

```python
                if success:
                    logger.info(f"Successfully downloaded podcast: {podcast_name}")
                    processed += 1
```

At the end of `main()`, after the loop (before the implicit return), add:

```python
    write_stage_status(
        stage=0,
        stage_name="download",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
        error_details=error_details,
    )
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/download/download.py', doraise=True); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/download/download.py
git commit -m "feat: add stage status reporting to download stage"
```

---

### Task 6: Modify `src/collate.py` — ThreadPoolExecutor stage

**Files:**
- Modify: `src/collate.py`

- [ ] **Step 1: Add import and error tracking**

Add import at top:

```python
from src.utils.stage_status import write_stage_status
```

In `main()`, add after `args = parser.parse_args()`:

```python
    processed = 0
    errors = 0
    error_details: list[dict] = []
```

In the `as_completed` loop (around line 57-64), modify the try/except:

Before:
```python
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
```

After:
```python
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    errors += 1
                    error_details.append({"file": str(path), "reason": str(e)})
```

After the success path in the same try block, add:

```python
                    processed += 1
```

At the end of `main()`, after `logger.info(f"Successfully saved data to {output_path}")`, add:

```python
    write_stage_status(
        stage=10,
        stage_name="collate",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
        error_details=error_details,
    )
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/collate.py', doraise=True); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/collate.py
git commit -m "feat: add stage status reporting to collate stage"
```

---

### Task 7: Modify `src/to_webdataset.py` — ProcessPoolExecutor stage

**Files:**
- Modify: `src/to_webdataset.py`

- [ ] **Step 1: Add import and shared counters**

Add import at top:

```python
import multiprocessing as mp
from src.utils.stage_status import write_stage_status
```

In `main(config)`, after `config = ...`, add shared counters:

```python
    processed = mp.Value('i', 0)
    errors = mp.Value('i', 0)
    error_details_list: list = []
```

In `worker_fn`, add `processed_counter` and `errors_counter` parameters. The function signature becomes:

```python
def worker_fn(worker_id: int, paths: list[str], output_dir: Path, config: dict,
              processed_counter, errors_counter):
```

In `worker_fn`, after successful file processing (after the `sink.write` success path), add:

```python
                processed_counter.value += 1
```

In each `except Exception` block inside `worker_fn`, add:

```python
                errors_counter.value += 1
```

In the `as_completed` loop in `main()`, modify the exception handler to track errors:

Before:
```python
            except Exception as e:
                logger.error(f"Worker process failed: {e}")
```

After:
```python
            except Exception as e:
                logger.error(f"Worker process failed: {e}")
                errors.value += 1
```

Pass the counters when submitting workers:

```python
    futures.append(executor.submit(worker_fn, i, chunk, output_dir, config,
                                    processed, errors))
```

At the end of `main()`, after `logger.success(...)`, add:

```python
    write_stage_status(
        stage=11,
        stage_name="to_webdataset",
        log_dir=config.get("log_dir", "./logs"),
        processed=processed.value,
        skipped=0,
        errors=errors.value,
    )
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/to_webdataset.py', doraise=True); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/to_webdataset.py
git commit -m "feat: add stage status reporting to to_webdataset stage"
```

---

### Task 8–12: mp.spawn stages (crest_factor_remover, preprocess_audio, music_detect, distillmos_process, preprocess)

These stages share a pattern: `mp.spawn(run_worker, ...)` with per-file CSV output. Each needs shared `multiprocessing.Value` counters passed to workers.

#### Task 8: Modify `src/preprocess/crest_factor_remover.py`

**Files:**
- Modify: `src/preprocess/crest_factor_remover.py`

- [ ] **Step 1: Add import and shared counters in `main()`**

Add import at top:

```python
from src.utils.stage_status import write_stage_status
```

Already imports `multiprocessing as mp` at top.

In `main(args)`, after `all_file_paths = ...` (approximately line 175), add:

```python
    processed = mp.Value('i', 0)
    skipped = mp.Value('i', 0)
    errors = mp.Value('i', 0)
```

- [ ] **Step 2: Pass counters to `run_worker`**

Change `run_worker` signature to accept counters. Add parameters:

```python
def run_worker(rank: int, world_size: int, all_file_paths: list[Path],
               podcasts_path: str, config: dict,
               processed_counter, skipped_counter, errors_counter):
```

In the spawn call, pass counters:

Before:
```python
    mp.spawn(run_worker, args=(num_workers, all_file_paths, podcasts_path, config),
             nprocs=num_workers, join=True)
```

After:
```python
    mp.spawn(run_worker, args=(num_workers, all_file_paths, podcasts_path, config,
                                processed, skipped, errors),
             nprocs=num_workers, join=True)
```

Also update the direct call:

Before:
```python
    run_worker(0, 1, all_file_paths, podcasts_path, config)
```

After:
```python
    run_worker(0, 1, all_file_paths, podcasts_path, config, processed, skipped, errors)
```

- [ ] **Step 3: Increment counters in worker**

In `run_worker`, modify the error handling:

In the batch computation try/except (approximately line 120-122):

Before:
```python
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue
```

After:
```python
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                errors_counter.value += 1
                continue
```

After successful batch processing and CSV writes, add:

```python
            processed_counter.value += len(batch_results)
```

For skipped files (`writer.already_done()`), increment skipped. In the file processing loop, after the resume check (approximately line 87-88):

```python
        if writer.already_done(resolved_path):
            skipped_counter.value += 1
            continue
```

- [ ] **Step 4: Write status at end of `main()`**

At the end of `main()` in the `if num_workers > 1` path (after the spawn block, before the absorb):

```python
    write_stage_status(
        stage=2,
        stage_name="crest_factor_remover",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )

    absorb_partial_csvs(...)
```

Also add the same call at the end of the `else` (direct call) path, and at the end of the function for the unified absorb:

Actually, the cleanest approach: add the `write_stage_status` call right before the final `logger.info("Crest factor check completed.")` (line 281), after the record_stage_summary if present. This covers both paths.

- [ ] **Step 5: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/preprocess/crest_factor_remover.py', doraise=True); print('OK')"
git add src/preprocess/crest_factor_remover.py
git commit -m "feat: add stage status reporting to crest_factor_remover stage"
```

#### Task 9: Modify `src/preprocess/preprocess_audio.py`

Same pattern as Task 8. Apply the same changes:

- Add import: `from src.utils.stage_status import write_stage_status`
- Add `processed`, `skipped`, `errors` as `mp.Value('i', 0)` in `main()`
- Pass counters to `run_worker` in both `mp.spawn` and direct call paths
- In `run_worker`:
  - `writer.already_done()` → `skipped_counter.value += 1`
  - Loader error → `errors_counter.value += 1`
  - Successful normalization → `processed_counter.value += 1`
- Call `write_stage_status(stage=3, stage_name="preprocess_audio", ...)` at end of `main()` before final logger

- [ ] **Step: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/preprocess/preprocess_audio.py', doraise=True); print('OK')"
git add src/preprocess/preprocess_audio.py
git commit -m "feat: add stage status reporting to preprocess_audio stage"
```

#### Task 10: Modify `src/separation/music_detect.py`

Same mp.spawn pattern. Apply:

- Add import: `from src.utils.stage_status import write_stage_status`
- Add `processed`, `skipped`, `errors` as `mp.Value('i', 0)` in `main()`
- Pass counters to `run_worker` (this stage only uses mp.spawn, no direct call path)
- In `run_worker`:
  - `writer.already_done()` → `skipped_counter.value += 1`
  - Model load error → `errors_counter.value += 1`
  - Worker-level `except Exception` → `errors_counter.value += 1`
  - File deletion failure → `errors_counter.value += 1`
  - Successful processing → `processed_counter.value += 1`
- Call `write_stage_status(stage=4, stage_name="music_detect", ...)` at end of `main()`

- [ ] **Step: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/separation/music_detect.py', doraise=True); print('OK')"
git add src/separation/music_detect.py
git commit -m "feat: add stage status reporting to music_detect stage"
```

#### Task 11: Modify `src/separation/distillmos_process.py`

Same mp.spawn pattern. Apply:

- Add import: `from src.utils.stage_status import write_stage_status`
- Add `processed`, `skipped`, `errors` as `mp.Value('i', 0)` in `main()`
- Pass counters to `run_inference_worker` in mp.spawn call
- In `run_inference_worker`:
  - `writer.already_done()` → `skipped_counter.value += 1`
  - Model load failure → `errors_counter.value += 1`, return early
  - CUDA OOM → `errors_counter.value += 1`, re-raise
  - Batch error → `errors_counter.value += 1`
  - Successful inference → `processed_counter.value += 1`
- Call `write_stage_status(stage=5, stage_name="distillmos_process", ...)` at end of `main()`

- [ ] **Step: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/separation/distillmos_process.py', doraise=True); print('OK')"
git add src/separation/distillmos_process.py
git commit -m "feat: add stage status reporting to distillmos_process stage"
```

#### Task 12: Modify `src/preprocess/preprocess.py` — complex two-level executor

This stage has a unique architecture: `ThreadPoolExecutor` outer (one thread per GPU) with `ProcessPoolExecutor` inner (one process per GPU for model inference).

**Files:**
- Modify: `src/preprocess/preprocess.py`

- [ ] **Step 1: Add import and shared counters**

Add import at top:

```python
from src.utils.stage_status import write_stage_status
```

In `main(args)`, after `chunk_audio = ...` setup (approximately line 580), add:

```python
    processed = 0
    errors = 0
    error_details: list[dict] = []
```

These are regular integers since the error counting happens in the main thread during `as_completed` aggregation (the inner `ProcessPoolExecutor` returns results through futures, and errors are caught during `future.result()`).

- [ ] **Step 2: Track errors in future aggregation**

In the `as_completed(futures)` loop (approximately line 626-630), modify the exception handler:

Before:
```python
            except Exception as e:
                logger.error(f"Error in GPU batch: {e}")
```

After:
```python
            except Exception as e:
                logger.error(f"Error in GPU batch: {e}")
                errors += 1
                error_details.append({"reason": str(e)})
```

For successful futures, count processed. After the successful path where segments are accumulated:

```python
            processed += 1
```

- [ ] **Step 3: Write status at end of `main()`**

At the very end of `main()`, after `record_stage_summary(...)` (approximately line 666), add:

```python
    write_stage_status(
        stage=1,
        stage_name="preprocess",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
        error_details=error_details,
    )
```

- [ ] **Step 4: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/preprocess/preprocess.py', doraise=True); print('OK')"
git add src/preprocess/preprocess.py
git commit -m "feat: add stage status reporting to preprocess stage"
```

---

### Task 13: Modify `src/transcription/transcription.py` — run_per_gpu_processes stage

This stage uses `run_per_gpu_processes` which was modified in Task 3. The worker function catches all exceptions internally, so we track errors inside the worker via shared counters.

**Files:**
- Modify: `src/transcription/transcription.py`

- [ ] **Step 1: Add import and shared counters**

Add import:

```python
from src.utils.stage_status import write_stage_status
```

In `main(args)`, after config loading (approximately line 225), add:

```python
    processed = mp.Value('i', 0)
    errors = mp.Value('i', 0)
    error_details_list = mp.Manager().list()
```

- [ ] **Step 2: Pass counters to `run_worker`**

Modify the call to `run_per_gpu_processes` to include counters in args:

Before:
```python
    run_per_gpu_processes(
        run_worker,
        num_gpus=num_gpus,
        args=(model_name, all_files, config),
    )
```

After:
```python
    run_per_gpu_processes(
        run_worker,
        num_gpus=num_gpus,
        args=(model_name, all_files, config, processed, errors, error_details_list),
    )
```

- [ ] **Step 3: Modify `run_worker` to accept and use counters**

Change `run_worker` signature:

```python
def run_worker(cuda_id: int, world_size: int, model_name: str, all_files: list[str],
               config: dict, processed_counter, errors_counter, error_details):
```

In `run_worker`, modify error handling:

In the catch-all `except Exception as e:` at the end of `run_worker` (approximately line 167-168):

Before:
```python
            except Exception as e:
                logger.exception(f"Worker {cuda_id} fatal error ({model_name}): {e}")
```

After:
```python
            except Exception as e:
                logger.exception(f"Worker {cuda_id} fatal error ({model_name}): {e}")
                errors_counter.value += 1
                error_details.append({"worker": cuda_id, "model": model_name, "reason": str(e)})
```

For per-file processing errors (in the batch/single-file loop, approximately line 150-156), track errors:

```python
            errors_counter.value += 1
            error_details.append({"file": str(audio_path), "model": model_name, "reason": str(e)})
```

For successful processing, after `save_results`:

```python
            processed_counter.value += 1
```

- [ ] **Step 4: Write status at end of `main()`**

At the end of `main()`, after `logger.info("Transcription pipeline complete!")`:

```python
    write_stage_status(
        stage=6,
        stage_name="transcription",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=0,
        errors=errors.value,
        error_details=list(error_details_list),
    )
```

- [ ] **Step 5: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/transcription/transcription.py', doraise=True); print('OK')"
git add src/transcription/transcription.py
git commit -m "feat: add stage status reporting to transcription stage"
```

---

### Task 14: Modify `src/punctuation/punctuation.py`, `src/accents/accents.py`, `src/phonemizer/phonemizer.py` — run_per_gpu_pool stages

These three stages use `run_per_gpu_pool` which was modified in Task 3 to return `(error_count, error_details)`. Each just needs to capture that return value and write the status file.

#### Task 14a: Modify `src/punctuation/punctuation.py`

**Files:**
- Modify: `src/punctuation/punctuation.py`

- [ ] **Step 1: Add import and capture return value**

Add import:

```python
from src.utils.stage_status import write_stage_status
```

In `main(args)`, change the `run_per_gpu_pool` call to capture the result:

Before:
```python
    run_per_gpu_pool(
        items,
        work_fn=make_punct_txt,
        initializer=init_process,
        initargs=(config, args.log_dir),
        num_gpus=num_gpus,
        desc="Punctuation",
    )
```

After:
```python
    error_count, error_details = run_per_gpu_pool(
        items,
        work_fn=make_punct_txt,
        initializer=init_process,
        initargs=(config, args.log_dir),
        num_gpus=num_gpus,
        desc="Punctuation",
    )
    write_stage_status(
        stage=7,
        stage_name="punctuation",
        log_dir=args.log_dir or "./logs",
        processed=len(items) - error_count,
        skipped=0,
        errors=error_count,
        error_details=error_details,
    )
```

- [ ] **Step 2: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/punctuation/punctuation.py', doraise=True); print('OK')"
git add src/punctuation/punctuation.py
git commit -m "feat: add stage status reporting to punctuation stage"
```

#### Task 14b: Modify `src/accents/accents.py`

Apply the same pattern as Task 14a:

- Add import: `from src.utils.stage_status import write_stage_status`
- Capture `error_count, error_details = run_per_gpu_pool(...)`
- Add `write_stage_status(stage=8, stage_name="accents", ...)` after
- `processed=len(items) - error_count`

- [ ] **Step: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/accents/accents.py', doraise=True); print('OK')"
git add src/accents/accents.py
git commit -m "feat: add stage status reporting to accents stage"
```

#### Task 14c: Modify `src/phonemizer/phonemizer.py`

Apply the same pattern:

- Add import: `from src.utils.stage_status import write_stage_status`
- Capture `error_count, error_details = run_per_gpu_pool(...)`
- Add `write_stage_status(stage=9, stage_name="phonemizer", ...)` after
- `processed=len(items) - error_count`

- [ ] **Step: Verify syntax and commit**

```bash
python3 -c "import py_compile; py_compile.compile('src/phonemizer/phonemizer.py', doraise=True); print('OK')"
git add src/phonemizer/phonemizer.py
git commit -m "feat: add stage status reporting to phonemizer stage"
```

---

### Task 15: End-to-end syntax check

- [ ] **Step 1: Verify all modules compile**

```bash
for f in \
  src/utils/stage_status.py \
  src/utils/parallel.py \
  src/report.py \
  src/download/download.py \
  src/collate.py \
  src/to_webdataset.py \
  src/preprocess/preprocess.py \
  src/preprocess/crest_factor_remover.py \
  src/preprocess/preprocess_audio.py \
  src/separation/music_detect.py \
  src/separation/distillmos_process.py \
  src/transcription/transcription.py \
  src/punctuation/punctuation.py \
  src/accents/accents.py \
  src/phonemizer/phonemizer.py; do
  python3 -c "import py_compile; py_compile.compile('$f', doraise=True)" || echo "FAIL: $f"
done
echo "All modules compile OK"
```

Expected: `All modules compile OK`

- [ ] **Step 2: Verify base.sh syntax**

```bash
bash -n base.sh && echo "base.sh syntax OK"
```

Expected: `base.sh syntax OK`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: verify all modules compile after strict pipeline mode changes"
```

---

## Verification

After implementation, manual smoke tests:

1. `bash base.sh --strict --stage 12 --stop_stage 12` — run report stage, verify `logs/stage_12_status.json` exists with `errors: 0`.
2. Run a stage known to have failures with `--strict` — verify pipeline stops.
3. Run same stage without `--strict` — verify pipeline continues (current behavior).
