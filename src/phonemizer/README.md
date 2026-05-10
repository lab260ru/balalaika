## Phonemizer (TryIParu)

Grapheme → IPA from **`{stem}_rover.txt`** using **`tryiparu.G2PModel`**
(workers call `load_dataset=True` at init).

Multi-GPU pool orchestration, sidecar discovery, and skip-already-done
filtering come from `src/utils`; the stage script is just `init_process` +
`process_text` + a `main()` glue that calls
`src.utils.parallel.run_per_gpu_pool` once.

## Run

```bash
bash src/phonemizer/phonemizer_yaml.sh configs/config.yaml
```

## Parameters

See **`phonemizer`** in `configs/config.yaml` (`podcasts_path`, `num_workers`).

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
