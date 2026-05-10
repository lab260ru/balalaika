## Download (Yandex Music)

Downloads episodes from URLs / playlists configured for the downloader.

## Run

```bash
bash src/download/download_yaml.sh configs/config.yaml
```

## Parameters

See **`download`** in `configs/config.yaml` (`podcasts_path`, `episodes_limit`, `num_workers`, `podcasts_urls_file`).

**Note:** `src/collate.py` also reads the **`download`** section for `podcasts_path` and `num_workers` when building `balalaika.parquet`. Keep that path aligned with the rest of the pipeline.

## Output

```text
{podcasts_path}/
└── {podcast_id}/
    └── {episode_id}/
        └── *.mp3
```

Next step: **preprocess** (Sortformer, Smart VAD, chunking).

## Resume

The downloader skips episodes whose target file already exists on disk, so a
forced stop simply resumes on the next run. There is no CSV state at this
stage; `balalaika.csv` is bootstrapped by **preprocess** (or any later
CSV-touching stage) when needed.
