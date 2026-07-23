"""Regression tests for safe_audio_duration probe order.

Pins the invariant that:
- .ogg paths use torchaudio-first (libsndfile excludes Vorbis padding frames
  that torchaudio/ffmpeg includes, causing ~30 ms under-count per file).
- .wav and .flac paths still use soundfile-first (fast header read, agrees
  with torchaudio exactly for PCM/FLAC containers).

The monkeypatch records which probe was consulted first and returns distinct
sentinel durations so we can verify both call order and return value.
"""
from __future__ import annotations

import pytest

import src.utils.audit as audit_mod


_SOUNDFILE_SENTINEL = 1.500  # what the (buggy) soundfile-first path would return
_TORCHAUDIO_SENTINEL = 1.528  # torchaudio/ffmpeg value that includes Vorbis padding


@pytest.fixture()
def patched_probes(monkeypatch):
    """Replace internal probe helpers with sentinels that record call order."""
    calls: list[str] = []

    def _fake_soundfile(p: str) -> float:
        calls.append("soundfile")
        return _SOUNDFILE_SENTINEL

    def _fake_torchaudio(p: str) -> float:
        calls.append("torchaudio")
        return _TORCHAUDIO_SENTINEL

    monkeypatch.setattr(audit_mod, "_soundfile_duration", _fake_soundfile)
    monkeypatch.setattr(audit_mod, "_torchaudio_duration", _fake_torchaudio)
    return calls


class TestSafeAudioDurationProbeOrder:
    def test_ogg_uses_torchaudio_first(self, patched_probes):
        """For .ogg, torchaudio must be consulted first and its value returned.

        This test FAILS before the fix because .ogg is in
        _SOUNDFILE_EXACT_SUFFIXES, making soundfile go first.
        """
        calls = patched_probes
        result = audit_mod.safe_audio_duration("/data/episode/chunk_0001.ogg")

        assert calls[0] == "torchaudio", (
            f"expected torchaudio as first probe for .ogg, got {calls[0]!r}; "
            "libsndfile excludes Vorbis padding frames (~30 ms per file)"
        )
        assert result == pytest.approx(_TORCHAUDIO_SENTINEL), (
            f"expected torchaudio sentinel {_TORCHAUDIO_SENTINEL}, got {result}"
        )

    def test_wav_uses_soundfile_first(self, patched_probes):
        """For .wav, soundfile must remain the first probe (fast header read)."""
        calls = patched_probes
        result = audit_mod.safe_audio_duration("/data/episode/chunk_0001.wav")

        assert calls[0] == "soundfile", (
            f"expected soundfile as first probe for .wav, got {calls[0]!r}"
        )
        assert result == pytest.approx(_SOUNDFILE_SENTINEL)

    def test_flac_uses_soundfile_first(self, patched_probes):
        """For .flac, soundfile must remain the first probe."""
        calls = patched_probes
        result = audit_mod.safe_audio_duration("/data/episode/chunk_0001.flac")

        assert calls[0] == "soundfile", (
            f"expected soundfile as first probe for .flac, got {calls[0]!r}"
        )
        assert result == pytest.approx(_SOUNDFILE_SENTINEL)

    def test_ogg_uppercase_extension_uses_torchaudio_first(self, patched_probes):
        """Extension matching is case-insensitive; .OGG must also go torchaudio-first."""
        calls = patched_probes
        result = audit_mod.safe_audio_duration("/data/ep/chunk.OGG")

        assert calls[0] == "torchaudio", (
            f"expected torchaudio-first for .OGG, got {calls[0]!r}"
        )
        assert result == pytest.approx(_TORCHAUDIO_SENTINEL)
