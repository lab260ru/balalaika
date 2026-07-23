from pathlib import Path
from typing import List, Tuple

import re
import yaml
from loguru import logger


def model_key(model_name: str) -> str:
    """Deterministic short key for an onnx-asr model name (no lookup table).

    Model names are passed straight to ``onnx_asr.load_model`` (e.g.
    ``gigaam-v3-ctc``, ``t-tech/t-one``, ``alphacep/vosk-model-ru``); this is the
    name used as the JSON/parquet key for that model's outputs. We take the last
    ``/``-segment so HF-style ``org/model`` names yield a clean column
    (``t-tech/t-one`` -> ``t-one``, ``alphacep/vosk-model-ru`` -> ``vosk-model-ru``).
    """
    return str(model_name).split("/")[-1]


def load_config(config_path: str, process_name: str):
    config = {}
    if config_path is None or process_name is None:
        logger.info("Configuration not provided. Parameters will be taken from argparse.")
        return config
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            config = config.get(process_name, {})
            logger.info('Loaded parameters from config')
    except Exception as e:
        logger.error(f"Configuration loading error: {e}")

    finally:
        return config

def get_txt_paths(podcast_path: str, postfix: str) -> List[Path]:
    return list(Path(podcast_path).rglob(f"*{postfix}"))

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return ''

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".opus")


def get_audio_paths(podcast_path: str):
    """Collect audio files in one os.walk pass (was: five full rglob scans).

    Matching stays case-sensitive for parity with the original
    ``rglob('*.mp3')`` behavior. Directory symlinks are not followed.
    """
    import os

    out = []
    append = out.append
    for root, _dirs, files in os.walk(podcast_path):
        for name in files:
            if name.endswith(AUDIO_SUFFIXES):
                append(Path(os.path.join(root, name)))
    return out


def process_token(token, label):
    if label == "LOWER_O":
        return token
    if label == "LOWER_PERIOD":
        return token + "."
    if label == "LOWER_COMMA":
        return token + ","
    if label == "LOWER_QUESTION":
        return token + "?"
    if label == "LOWER_TIRE":
        return token + "—"
    if label == "LOWER_DVOETOCHIE":
        return token + ":"
    if label == "LOWER_VOSKL":
        return token + "!"
    if label == "LOWER_PERIODCOMMA":
        return token + ";"
    if label == "LOWER_DEFIS":
        return token + "-"
    if label == "LOWER_MNOGOTOCHIE":
        return token + "..."
    if label == "LOWER_QUESTIONVOSKL":
        return token + "?!"
    if label == "UPPER_O":
        return token.capitalize()
    if label == "UPPER_PERIOD":
        return token.capitalize() + "."
    if label == "UPPER_COMMA":
        return token.capitalize() + ","
    if label == "UPPER_QUESTION":
        return token.capitalize() + "?"
    if label == "UPPER_TIRE":
        return token.capitalize() + " —"
    if label == "UPPER_DVOETOCHIE":
        return token.capitalize() + ":"
    if label == "UPPER_VOSKL":
        return token.capitalize() + "!"
    if label == "UPPER_PERIODCOMMA":
        return token.capitalize() + ";"
    if label == "UPPER_DEFIS":
        return token.capitalize() + "-"
    if label == "UPPER_MNOGOTOCHIE":
        return token.capitalize() + "..."
    if label == "UPPER_QUESTIONVOSKL":
        return token.capitalize() + "?!"
    if label == "UPPER_TOTAL_O":
        return token.upper()
    if label == "UPPER_TOTAL_PERIOD":
        return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":
        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":
        return token.upper() + "?"
    if label == "UPPER_TOTAL_TIRE":
        return token.upper() + " —"
    if label == "UPPER_TOTAL_DVOETOCHIE":
        return token.upper() + ":"
    if label == "UPPER_TOTAL_VOSKL":
        return token.upper() + "!"
    if label == "UPPER_TOTAL_PERIODCOMMA":
        return token.upper() + ";"
    if label == "UPPER_TOTAL_DEFIS":
        return token.upper() + "-"
    if label == "UPPER_TOTAL_MNOGOTOCHIE":
        return token.upper() + "..."
    if label == "UPPER_TOTAL_QUESTIONVOSKL":
        return token.upper() + "?!"
    logger.debug(f"process_token: unrecognized label {label!r}; returning token unchanged.")
    return token

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text
