"""Fix A (atomic sidecar write) + Fix B (fast-path fallback observability).

FIX A — ``transcription._write_text_atomic`` must stage through a *unique*
tempfile in the destination directory (via ``tempfile.mkstemp``), clean it up
if ``os.replace`` fails, and never use a FIXED ``<name>.tmp`` staging path (an
orphaned worker from a crashed re-run could otherwise race that fixed path and
publish interleaved bytes). This mirrors the peer writers in
``phonemizer._process_one`` and ``accents._write_accent``.

FIX B — the three silent fast-path fallbacks must leave a greppable signal:
  * ``accents.process_chunk``  : ``process_batch`` -> per-file stock path.
  * ``rover.ROVERWrapper._make_aggregator`` : FastROVER -> crowd-kit ROVER.
  * ``transcription.maybe_patch_fast_rnnt`` : fast_rnnt patch skip.
Each increments a fallback counter that surfaces in a clearly greppable summary
log line ('fast-path fallbacks: N'); behavior is otherwise identical.

Run: .dev_venv/bin/python -m pytest tests/test_atomic_write_and_fallback_obs.py -x -q
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from loguru import logger


# --------------------------------------------------------------------------- #
# loguru capture helper (loguru does not feed pytest's caplog).               #
# --------------------------------------------------------------------------- #
class _LogCapture:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message) -> None:  # loguru sink
        self.messages.append(str(message))

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


@pytest.fixture()
def logcap():
    cap = _LogCapture()
    sink_id = logger.add(cap, level="DEBUG", format="{message}")
    try:
        yield cap
    finally:
        logger.remove(sink_id)


# --------------------------------------------------------------------------- #
# FIX A — _write_text_atomic                                                   #
# --------------------------------------------------------------------------- #
class TestWriteTextAtomic:
    def test_content_written_correctly(self, tmp_path):
        import src.transcription.transcription as tr

        dst = tmp_path / "file_giga_rnnt.txt"
        tr._write_text_atomic(dst, "привет мир")
        assert dst.read_text(encoding="utf-8") == "привет мир"

    def test_uses_unique_tempfile_not_fixed_name(self, tmp_path, monkeypatch):
        """The staging file must come from ``tempfile.mkstemp`` (unique), NOT a
        fixed ``<name>.tmp`` path. This assertion FAILS pre-fix because the old
        code uses ``path.with_name(path.name + '.tmp')``."""
        import src.transcription.transcription as tr

        seen = {}
        real_mkstemp = tempfile.mkstemp

        def recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            seen["dir"] = kwargs.get("dir")
            seen["suffix"] = kwargs.get("suffix")
            seen["name"] = name
            return fd, name

        monkeypatch.setattr(tr.tempfile, "mkstemp", recording_mkstemp)

        dst = tmp_path / "file_giga_rnnt.txt"
        # Guard: the fixed legacy staging name must NEVER appear on disk.
        legacy_tmp = dst.with_name(dst.name + ".tmp")

        real_replace = tr.os.replace

        def guarded_replace(src, target):
            assert not legacy_tmp.exists(), "fixed-name staging file was used"
            assert Path(src) != legacy_tmp, "os.replace staged from fixed-name tmp"
            return real_replace(src, target)

        monkeypatch.setattr(tr.os, "replace", guarded_replace)

        tr._write_text_atomic(dst, "data")

        assert seen, "tempfile.mkstemp was not used (fixed-name staging path)"
        assert seen["suffix"] == ".tmp"
        assert Path(seen["dir"]) == dst.parent
        assert dst.read_text(encoding="utf-8") == "data"
        assert not legacy_tmp.exists()

    def test_error_path_cleans_up_tmp_and_propagates(self, tmp_path, monkeypatch):
        import src.transcription.transcription as tr

        created = []
        real_mkstemp = tempfile.mkstemp

        def recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            created.append(name)
            return fd, name

        monkeypatch.setattr(tr.tempfile, "mkstemp", recording_mkstemp)

        def boom(src, target):
            raise OSError("replace failed")

        monkeypatch.setattr(tr.os, "replace", boom)

        dst = tmp_path / "file_giga_rnnt.txt"
        with pytest.raises(OSError, match="replace failed"):
            tr._write_text_atomic(dst, "data")

        assert created, "tempfile.mkstemp not used"
        # The staging file must be removed on the error path.
        assert not os.path.exists(created[0]), "tmp file leaked on error"
        assert not dst.exists()


# --------------------------------------------------------------------------- #
# FIX B.1 — accents.process_chunk fallback counter + summary line             #
# --------------------------------------------------------------------------- #
class TestAccentsFallbackObservability:
    def test_batch_failure_falls_back_and_counts(self, tmp_path, logcap, monkeypatch):
        import src.accents.accents as acc

        # Stub accentizer: process_batch raises (force fallback), process_all
        # succeeds (stock per-file path produces real output).
        class _StubAcc:
            def process_batch(self, texts):
                raise RuntimeError("batch boom")

            def process_all(self, text):
                return f"<{text}>"

        monkeypatch.setattr(acc, "accentizer", _StubAcc())

        # _punct.txt -> _accent.txt sidecars
        p1 = tmp_path / "a_punct.txt"
        p2 = tmp_path / "b_punct.txt"
        p1.write_text("один", encoding="utf-8")
        p2.write_text("два", encoding="utf-8")

        failures = acc.process_chunk([p1, p2])

        # Stock path was used: outputs written, no failures.
        assert failures == []
        assert (tmp_path / "a_accent.txt").read_text(encoding="utf-8") == "<один>"
        assert (tmp_path / "b_accent.txt").read_text(encoding="utf-8") == "<два>"
        # Counter surfaced in a greppable summary line.
        assert "fast-path fallbacks:" in logcap.text


# --------------------------------------------------------------------------- #
# FIX B.2 — rover._make_aggregator fallback counter + summary line            #
# --------------------------------------------------------------------------- #
class TestRoverFallbackObservability:
    def test_fast_rover_unavailable_falls_back_and_counts(self, logcap, monkeypatch):
        import src.transcription.rover as rover_mod
        from crowdkit.aggregation import ROVER

        wrapper = rover_mod.ROVERWrapper(podcasts_path=".", model_names=["m0"])

        # Force the FastROVER import/construction to raise inside _make_aggregator.
        import src.transcription.fast_rover as fr

        def boom(*a, **k):
            raise RuntimeError("no fast rover")

        monkeypatch.setattr(fr, "FastROVER", boom)

        agg = wrapper._make_aggregator()
        assert isinstance(agg, ROVER)

        # A second fallback to prove the counter increments.
        wrapper._make_aggregator()

        assert wrapper.fast_path_fallbacks == 2
        wrapper.log_fallback_summary()
        assert "fast-path fallbacks: 2" in logcap.text


# --------------------------------------------------------------------------- #
# FIX B.3 — transcription.maybe_patch_fast_rnnt fallback counter + summary    #
# --------------------------------------------------------------------------- #
class TestFastRnntFallbackObservability:
    def test_patch_skip_counts_and_summary(self, logcap, monkeypatch):
        import src.transcription.transcription as tr

        def boom(model):
            raise RuntimeError("patch boom")

        monkeypatch.setattr(tr, "_patch_fast_rnnt", boom)
        tr.reset_fast_path_fallbacks()

        sentinel = object()
        out = tr.maybe_patch_fast_rnnt(sentinel, {"use_fast_rnnt": True})
        assert out is sentinel  # unpatched stock model returned (behavior intact)
        assert tr.fast_path_fallback_count() == 1

        tr.log_fast_path_fallbacks(cuda_id=0)
        assert "fast-path fallbacks: 1" in logcap.text
