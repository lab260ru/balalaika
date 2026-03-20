## Phonemizer (TryIParu)

Grapheme → IPA from **`{stem}_rover.txt`** using **`tryiparu.G2PModel`** (workers call `load_dataset=True` at init).

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
