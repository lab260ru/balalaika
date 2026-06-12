"""Stage 8 — RUPunct punctuation restoration on ``*_rover.txt`` sidecars."""
import argparse
import multiprocessing
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer, pipeline

from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.node_profile import resolve_batch_size
from src.utils.parallel import run_per_gpu_pool
from src.utils.sidecars import DirNameCache, pending_audio_to_sidecar
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, process_token, read_file_content

apply_torch_perf_defaults()

# RUPunct/RUPunct_big is a BERT with max_position_embeddings=512, but its
# tokenizer ships model_max_length = the VERY_LARGE_INTEGER sentinel, so the
# pipeline's `truncation=True` is a silent no-op (verified on this node). Any
# text tokenizing past this many ids would feed out-of-range position ids into
# the embedding (a CUDA device-side assert that poisons the worker). We pin the
# tokenizer to 512 at init (engages the fast tokenizer's native chunk+aggregate
# for over-limit texts) and pre-screen oversize texts onto the per-file path so
# one long transcript can't fail — or 2x-rerun — its whole slab.
MODEL_MAX_TOKENS = 512
MODEL_STRIDE = 128

# HF pipelines pad each DataLoader batch to its in-batch max, so the length sort
# only cuts padding waste if the slab is fed in several smaller micro-batches:
# length-sorted neighbors share micro-batches padded to their LOCAL max.
PUNCT_PIPELINE_BATCH = 8

model = None
tokenizer = None


def init_process(model_name: str, device: str) -> None:
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        strip_accents=False,
        add_prefix_space=True,
    )
    # Make `truncation`/chunking actually honor the model's 512-token limit.
    tokenizer.model_max_length = MODEL_MAX_TOKENS
    model = pipeline(
        "ner",
        model=model_name,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=device,
        stride=MODEL_STRIDE,
    )


def _token_count(text: str) -> int:
    """Number of input ids (incl. special tokens) the model would see."""
    return len(tokenizer(text, truncation=False)["input_ids"])


def _punct_text_from_preds(preds) -> str:
    return " ".join(
        process_token(item["word"].strip(), item["entity_group"]) for item in preds
    ).strip()


def make_punct_txt(rover_path: Path) -> None:
    rover_path = Path(rover_path)
    punct_path = rover_path.with_name(rover_path.name.replace("_rover.txt", "_punct.txt"))
    if punct_path.exists():
        return

    src_text = read_file_content(rover_path)
    if not src_text:
        return

    preds = model(src_text)
    punct_path.write_text(_punct_text_from_preds(preds), encoding="utf-8")


def _punct_one(punct_path: Path, text: str) -> None:
    preds = model(text)
    punct_path.write_text(_punct_text_from_preds(preds), encoding="utf-8")


def make_punct_batch(rover_paths) -> None:
    """Batched RUPunct over a slab of files (one pipeline call for the slab).

    The NER pipeline is ~5x faster fed in batches than file-by-file (measured
    by benchmarking/warmup.py on this node: 48.8 -> 252 texts/s at batch 64).

    Two padding/robustness refinements:

    * Texts are sorted by token length and fed in PUNCT_PIPELINE_BATCH-sized
      micro-batches so a single long transcript doesn't pad the whole slab to
      its length (HF pads each DataLoader batch to its in-batch max, and BERT
      attention is O(L^2)). Each output goes to its own file, so write-back
      order is irrelevant and per-text outputs are unchanged.
    * Texts tokenizing past the model's 512-token limit are pre-screened onto
      the per-file path (the fast tokenizer chunks + aggregates them there)
      rather than being fed into the batched call where they would fail the
      whole slab and force a per-file re-run of every good file.

    Still falls back to per-file processing if the batched call fails, so one
    bad file cannot take down its whole slab.
    """
    pending: list[tuple[Path, str, int]] = []
    oversize: list[tuple[Path, str]] = []
    for rover_path in rover_paths:
        rover_path = Path(rover_path)
        punct_path = rover_path.with_name(
            rover_path.name.replace("_rover.txt", "_punct.txt")
        )
        if punct_path.exists():
            continue
        src_text = read_file_content(rover_path)
        if not src_text:
            continue
        n_tokens = _token_count(src_text)
        if n_tokens > MODEL_MAX_TOKENS:
            oversize.append((punct_path, src_text))
        else:
            pending.append((punct_path, src_text, n_tokens))

    failed: list[str] = []

    # Oversize texts: the doomed-in-a-batch ones. Process each on its own via
    # the pipeline's native 512-token chunk+aggregate (model_max_length is
    # pinned in init_process).
    for punct_path, text in oversize:
        try:
            _punct_one(punct_path, text)
        except Exception as file_exc:
            logger.error(f"Punctuation failed for {punct_path}: {file_exc}")
            failed.append(punct_path.name)

    if pending:
        # Group similar lengths together to cut padding waste in the one
        # padded forward per slab.
        pending.sort(key=lambda item: item[2])
        try:
            all_preds = model(
                [text for _, text, _ in pending],
                batch_size=min(PUNCT_PIPELINE_BATCH, len(pending)),
            )
            for (punct_path, _, _), preds in zip(pending, all_preds):
                punct_path.write_text(
                    _punct_text_from_preds(preds), encoding="utf-8"
                )
        except Exception as exc:
            logger.warning(f"Batched punctuation failed ({exc}); retrying per file.")
            for punct_path, text, _ in pending:
                try:
                    _punct_one(punct_path, text)
                except Exception as file_exc:
                    logger.error(f"Punctuation failed for {punct_path}: {file_exc}")
                    failed.append(punct_path.name)

    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed: {failed[:3]}...")


def main(args):
    setup_logging("punctuation", log_dir=args.log_dir)
    config = load_config(args.config_path, "punctuation")

    num_workers_per_gpu = config.get("num_workers", 4)
    model_name = config.get("model_name", "RUPunct/RUPunct_big")
    podcasts_path = config.get("podcasts_path", "../../../balalaika")

    pending_files = pending_audio_to_sidecar(
        podcasts_path,
        in_suffix="_rover.txt",
        out_suffix="_punct.txt",
        config_path=args.config_path,
    )
    if not pending_files:
        logger.success("No pending _rover.txt files; punctuation already up to date.")
        return
    # Path order keeps sidecar reads directory-clustered on HDD datasets
    # (pending order otherwise follows arbitrary CSV-row order).
    pending_files.sort()

    logger.info(f"Found {len(pending_files)} _rover.txt files needing punctuation.")

    batch_size = resolve_batch_size("punctuation", config.get("batch_size"), 16)
    slabs = [
        pending_files[i : i + batch_size]
        for i in range(0, len(pending_files), batch_size)
    ]

    error_count, error_details = run_per_gpu_pool(
        slabs,
        work_fn=make_punct_batch,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (model_name, f"cuda:{gpu_id}"),
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Punctuation",
    )
    # Exact accounting: errors are per-slab now, so count produced sidecars.
    # Use DirNameCache so this is O(#dirs) scandirs instead of O(#files)
    # cold-cache stat() calls on the production HDD (same fix as §4.7's
    # pending scans; identical counting semantics).
    out_cache = DirNameCache()
    produced = sum(
        1
        for rover_path in map(Path, pending_files)
        if out_cache.exists(
            rover_path.with_name(
                rover_path.name.replace("_rover.txt", "_punct.txt")
            )
        )
    )
    write_stage_status(
        stage=8,
        stage_name="punctuation",
        log_dir=args.log_dir or "./logs",
        processed=produced,
        skipped=0,
        errors=error_count,
        error_details=error_details,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Multi-GPU punctuation restoration via RUPunct.")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
