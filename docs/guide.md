# build vosk
```cmd
python -m pip install git+https://github.com/lhotse-speech/lhotse
```
```cmd
python -m pip install pip install https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250807+cuda12.9.torch2.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```
```cmd
python -m pip install kaldifeat==1.25.5.dev20250203+cuda12.4.torch2.5.1 -f https://csukuangfj.github.io
/kaldifeat/cuda.html
```
```
python3 -m pip install git+https://github.com/k2-fsa/icefall
```

# balalaika on your data
if the raw data
```
dataset/
└── {album_id}/
    └── {episode_id}/
        ├── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_giga.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_punct.txt
        ├── {start_time}_{end_time}_{album_id}_{episode_id}_accent.txt
        └── {start_time}_{end_time}_{album_id}_{episode_id}_giga_phonemes.txt
```

if the raw data
```
dataset/
└── {album_id}/
    └── {episode_id}/
        └── {start_time}_{end_time}_{album_id}_{episode_id}.mp3
        ...
```

if the audio is already cropped

```
dataset/
└── {album_id}/
    └── {episode_id}/
        ├─ audio_1.mp3
        ├─ audio_2.wav
        ├─ audio_3.opus
        ...
```

# if only transcriptions are needed
`base.sh 

```
SCRIPTS=(
    # "./src/download/download_yaml.sh"
    # "./src/preprocess/preprocess_yaml.sh"
    # "./src/separation/separation_yaml.sh"
    "./src/transcription/transcription_yaml.sh"
    # "./src/punctuation/punctuation_yaml.sh"
    # "./src/accents/accents_yaml.sh"
    # "./src/phonemizer/phonemizer_yaml.sh"
    # "./src/collate_yamls.sh"
)
```