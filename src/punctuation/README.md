## Punctuation (RUPunct)

Restores punctuation and capitalization from **`{stem}_rover.txt`**.

Multi-GPU pool orchestration, sidecar discovery, and skip-already-done
filtering all come from `src/utils` — the stage script is just `init_process`
+ `make_punct_txt` + a `main()` glue that calls
`src.utils.parallel.run_per_gpu_pool` once.

## Run

```bash
bash src/punctuation/punctuation_yaml.sh configs/config.yaml
```

## Parameters

See **`punctuation`** in `configs/config.yaml` (`podcasts_path`, `model_name`, `num_workers`).

## Output

For each `{stem}_rover.txt`, writes **`{stem}_punct.txt`**.

```text
{podcasts_path}/
└── {playlist_id}/
    └── {podcast_id}/
        ├── {stem}.mp3
        ├── {stem}_rover.txt   # input
        └── {stem}_punct.txt   # output
```

WebDataset packs this as `punct.txt` inside `json` (`src/to_webdataset.py`).

## Resume / interrupt safety

Pending work is rediscovered each run via
`src.utils.sidecars.pending_audio_to_sidecar(in_suffix="_rover.txt",
out_suffix="_punct.txt")`, which only returns chunks whose `_punct.txt` is
missing. Because the output file is written atomically per chunk, a forced
stop never leaves a half-written sidecar that would be mistakenly skipped on
the next run.
