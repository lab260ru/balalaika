## Accents (ruAccent)

Lexical stress and text normalization from the **`punct`** text in each chunk's
**`{stem}.json`**, writing the **`accent`** key back.

Multi-GPU pool orchestration, chunk-JSON discovery, and skip-already-done
filtering come from `src/utils`; the stage script is just `init_process` +
`process_file` + a `main()` glue that calls
`src.utils.parallel.run_per_gpu_pool` once.

## Run

```bash
bash src/accents/accents_yaml.sh configs/config.yaml
```

## Parameters

See **`accent`** in `configs/config.yaml` (`podcasts_path`, `model_name`, `num_workers`, `use_tensorrt`). The YAML section name is **`accent`**, not `accents`.

When `use_tensorrt: True`, providers are built by
`src.utils.gpu.get_onnx_providers`, which reads the engine-cache root and
workspace from the global `runtime:` block.

## Output

```text
{stem}_punct.txt  →  {stem}_accent.txt
```

WebDataset key: **`accent.txt`**.

## Resume / interrupt safety

Pending inputs are rediscovered via
`src.utils.sidecars.pending_sidecar_chain(in_suffix="_punct.txt",
out_derive=replace_in_stem(_, "_punct", "_accent"))`, which only returns
files whose `_accent.txt` is missing. Output files are written via
`Path.write_text`, so a forced stop never leaves a partial sidecar.
