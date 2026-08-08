from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).parents[1] / "docker" / "prepare_config.py"
SPEC = importlib.util.spec_from_file_location("balalaika_prepare_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_config)


def test_render_config_rewrites_container_paths_without_touching_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.yaml"
    destination = tmp_path / "rendered.yaml"
    original = {
        "cache_path": "./cache",
        "runtime": {
            "venv_path": "/host/venv",
            "log_dir": "/host/logs",
            "trt_cache_path": "/host/trt",
        },
        "download": {"podcasts_path": "/host/data"},
        "preprocess": {
            "podcasts_path": "/host/data",
            "sortformer_model": "./models/sortformer.onnx",
            "vad_args": {"smart_vad_model": "/host/models/smart-turn.onnx"},
        },
        "separation": {
            "podcasts_path": "/host/data",
            "music_detect": {"onnx_path": "./models/music.onnx"},
            "antispoofing": {"onnx_path": "./models/spoof.onnx"},
            "tts_suitability": {"onnx_path": "./models/tts.onnx"},
        },
        "transcription": {
            "podcasts_path": "/host/data",
            "model_path": "/host/models/gigaam",
            "vosk_path": "/host/models/vosk",
        },
        "phonemizer": {
            "podcasts_path": "/host/data",
            "oov_cache_path": "cache/oov.pkl",
        },
        "denoising": {
            "podcasts_path": "/host/data",
            "onnx_path": "./models/denoise.onnx",
        },
        "export": {
            "podcasts_path": "/host/data",
            "output_path": "/host/output",
        },
    }
    source.write_text(yaml.safe_dump(original), encoding="utf-8")

    monkeypatch.setenv("VIRTUAL_ENV", "/opt/balalaika/.venv")
    monkeypatch.setenv("BALALAIKA_DATA_ROOT", "/data")
    monkeypatch.setenv("BALALAIKA_MODELS_ROOT", "/models")
    monkeypatch.setenv("BALALAIKA_OUTPUT_ROOT", "/output")
    monkeypatch.setenv("BALALAIKA_LOG_DIR", "/logs")
    monkeypatch.setenv("BALALAIKA_TRT_CACHE_PATH", "/cache/trt")
    monkeypatch.setenv("BALALAIKA_CACHE_ROOT", "/node-cache/balalaika")

    prepare_config.render_config(source, destination)

    rendered = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == original
    assert rendered["runtime"]["venv_path"] == "/opt/balalaika/.venv"
    assert rendered["runtime"]["log_dir"] == "/logs"
    assert rendered["runtime"]["trt_cache_path"] == "/cache/trt"
    assert rendered["download"]["podcasts_path"] == "/data"
    assert rendered["preprocess"]["podcasts_path"] == "/data"
    assert rendered["separation"]["podcasts_path"] == "/data"
    assert rendered["phonemizer"]["podcasts_path"] == "/data"
    assert rendered["denoising"]["podcasts_path"] == "/data"
    assert rendered["export"]["podcasts_path"] == "/data"
    assert rendered["preprocess"]["sortformer_model"] == "/models/sortformer.onnx"
    assert (
        rendered["preprocess"]["vad_args"]["smart_vad_model"]
        == "/models/smart-turn.onnx"
    )
    assert rendered["separation"]["music_detect"]["onnx_path"] == "/models/music.onnx"
    assert rendered["denoising"]["onnx_path"] == "/models/denoise.onnx"
    assert rendered["transcription"]["model_path"] == "/models/gigaam"
    assert rendered["transcription"]["vosk_path"] == "/models/vosk"
    assert rendered["phonemizer"]["oov_cache_path"] == (
        "/node-cache/balalaika/g2p_oov_cache.pkl"
    )
    assert rendered["cache_path"] == "/node-cache/balalaika"
    assert rendered["export"]["output_path"] == "/output"


def test_render_config_rejects_non_mapping_yaml(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("- invalid\n- top-level\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Top-level config"):
        prepare_config.render_config(source, tmp_path / "rendered.yaml")
