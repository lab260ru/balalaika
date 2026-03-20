## Punctuation (RUPunct)

Restores punctuation and capitalization from **`{stem}_rover.txt`**.

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
