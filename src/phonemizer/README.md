## Phonemizer (TryIParu)

Grapheme → IPA from **`{stem}_rover.txt`** using **`src.phonemizer.fast_g2p.FastG2P`**
— a drop-in, token-identical replacement for `tryiparu.G2PModel` (same weights,
tokenizer and rules) that greedy-decodes the unique OOV words of a file as one
padded batch (~21-37× per word), loads the 398k-word dictionary from a pickle
cache (`cache/g2p_dict.pkl`), memoizes rule splits per unique word, and
persists OOV decodes across runs/workers (`oov_cache_path` config knob,
weights-fingerprint keyed). Equivalence to stock is pinned by
`tests/test_phonemizer_fast_g2p.py` and report.md §4.9.

Multi-GPU pool orchestration, sidecar discovery, and skip-already-done
filtering come from `src/utils`; the stage script is just `init_process` +
`process_text` + a `main()` glue that calls
`src.utils.parallel.run_per_gpu_pool` once.

## Run

```bash
bash src/phonemizer/phonemizer_yaml.sh configs/config.yaml
```

## Parameters

See **`phonemizer`** in `configs/config.yaml` (`podcasts_path`, `num_workers`,
`device` — set `cpu` to run the tiny d=128 model on CPU when GPUs are busy,
`g2p_batch_size`, `oov_cache_path` — `""` disables the persistent OOV cache).

## Output

For each `{stem}_rover.txt`:

- **`{stem}_rover_phonemes.txt`** — space-separated IPA symbols.

WebDataset key: **`rover_phonemes.txt`**.

## Resume / interrupt safety

Pending inputs are rediscovered via
`src.utils.sidecars.pending_sidecar_chain(in_suffix="_rover.txt",
out_derive=lambda p: p.with_name(f"{p.stem}_phonemes.txt"))`, which only
returns chunks whose `_rover_phonemes.txt` is missing. A forced stop is safe:
on the next run, only un-phonemized chunks are scheduled.
