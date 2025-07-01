# BalalaikaDataset Example

This example demonstrates how to load and access a sample from the `BalalaikaDataset`.

## Usage

```python
from dataset import BalalaikaDataset

if __name__ == "__main__":
    dataset = BalalaikaDataset(
        podcasts_path='../Balalaika100H',
        parquet_path='../balalaika/balalaika.parquet'
    )
    print(dataset[0])
```

## Output Example

```python
(
  '/home/nikita/podcasts_1/21851634/102739417/469.72_483.93_21851634_102739417.mp3',
  {
    'audio_path': '21851634/102739417/469.72_483.93_21851634_102739417.mp3',
    'is_mono': True,
    'NOI': 3.0010192,
    'COL': 4.100467,
    'DISC': 2.4562664,
    'LOUD': 3.4343345,
    'MOS': 3.9075212,
    'playlist_id': 21851634,
    'podcast_id': 102739417,
    'start': 469.72,
    'end': 483.93,
    'speaker': 0.0,
    'fullness': 0.8249,
    'accent': 'П+апа +едет дом+ой. ...',
    'phonemes': 'p a p ə   j e dʲ ɪ t   d ɐ m o j ...',
    'giga': 'папа едет домой и оля сейчас ...',
    'punct': 'Папа едет домой. И Оля сейчас поедет домой. ...',
    'whisper': 'Папа едет домой. И Оля сейчас поедет домой. ...',
    'e': 'папа едет домой и оля сейчас ...'
  }
)
```

## Field Descriptions

- **`audio_path`** — relative path to the audio segment.
- **`is_mono`** — whether the audio is mono.
- **`NOI`, `COL`, `DISC`, `LOUD`, `MOS`** — NISQA metrics (noise, coloration, discontinuity, loudness, MOS).
- **`playlist_id`, `podcast_id`, `start`, `end`** — source identifiers and time boundaries of the segment.
- **`speaker`** — predicted speaker ID.
- **`fullness`** — ratio of speech to silence in the segment.
- **`accent`** — text with stress markers.
- **`phonemes`** — phoneme-level representation of the utterance.
- **`giga`** — raw ASR output from GigaAM.
- **`punct`** — GigaAM output with punctuation.
- **`whisper`** — transcription from Whisper model.
- **`e`** — GigaAM output with correct usage of the letter `ё`.

---

Ensure that the dataset path and metadata are correctly specified before running the script.
