# Balalaika WebDataset (Hugging Face Compatible)

This repository contains the **Balalaika Dataset** packed into the [WebDataset](https://github.com/webdataset/webdataset) format. It is specifically optimized for use with the Hugging Face `datasets` library, allowing for high-performance streaming directly from disk or remote storage.

## Features

- **Efficient I/O**: Data is stored in `.tar` shards, reducing metadata overhead on the file system.
- **Streaming Support**: Load massive datasets without filling up your RAM.
- **Auto-Decoding**: Hugging Face automatically decodes audio into NumPy arrays and parses JSON metadata.

## Installation

You will need the `datasets` and `webdataset` libraries:

```bash
pip install datasets webdataset torchaudio

Usage Example

The following script demonstrates how to load the dataset using the Hugging Face load_dataset interface with streaming enabled.
code Python

from datasets import load_dataset
import time

if __name__ == "__main__":
    # Hugging Face scans the folder for .tar archives and collects them
    dataset = load_dataset(
        "webdataset", 
        data_dir="/path/to/balalaika_data_webdataset/train", 
        split="train",
        streaming=True  # Enables streaming for low memory usage
    )

    for item in dataset:
        print(f"=== Sample ID: {item['__key__']} ===")
        
        # Audio is automatically loaded as a dictionary (array and sampling_rate)
        # The key matches the file extension (mp3, wav, flac, etc.)
        audio_key = next((k for k in item.keys() if k in ['mp3', 'wav', 'flac', 'ogg']), None)
        
        if audio_key:
            audio_data = item[audio_key]
            print(f"Audio Shape: {audio_data['array'].shape}")
            print(f"Sampling Rate: {audio_data['sampling_rate']} Hz")
        
        # JSON metadata and transcriptions are automatically parsed into a dictionary
        metadata = item['json']
        print(f"Transcription: {metadata.get('whisper')}")
        print(f"Quality Score (MOS): {metadata.get('DistillMOS')}")
        
        print("-" * 50)
        time.sleep(1) # Just for demonstration purposes

Dataset Structure

When using load_dataset, each sample is a dictionary with columns mapped from the file extensions inside the .tar shards:
Column	Type	Description
__key__	string	Unique sample identifier (dots replaced with underscores).
mp3 / wav	dict	Audio data containing array and sampling_rate.
json	dict	Metadata including metrics and transcriptions.
Metadata (JSON) Fields

The json column consolidates all textual information:

    DistillMOS: Predicted Mean Opinion Score for speech quality.

    whisper / giga / punct: Transcriptions from various ASR models.

    accent: Text with stress markers.

    phonemes: Phonetic representation of the audio.

    speaker: Speaker identification ID.

    fullness: Ratio of active speech in the segment.

Advantages of this Format

    Zero Bottleneck: Optimized for training on fast GPUs where reading individual small files is usually the bottleneck.

    Easy Integration: Works out-of-the-box with DataCollator and standard Hugging Face training pipelines.

    Flexible Storage: The dataset can be stored locally, on an S3 bucket, or on the Hugging Face Hub without changing the loading logic.
    """

print(readme_content)