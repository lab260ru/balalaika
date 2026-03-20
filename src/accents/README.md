## Accents (ruAccent)

Lexical stress and text normalization from **`{stem}_punct.txt`**.

## Run

```bash
bash src/accents/accents_yaml.sh configs/config.yaml
```

## Parameters

See **`accent`** in `configs/config.yaml` (`podcasts_path`, `model_name`, `num_workers`, `use_tensorrt`). The YAML section name is **`accent`**, not `accents`.

## Output

```text
{stem}_punct.txt  →  {stem}_accent.txt
```

WebDataset key: **`accent.txt`**.
