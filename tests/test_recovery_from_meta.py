"""Logic tests for recovery_from_meta.process_audio_file with a fake decoder.

Pins: (a) the existence check short-circuits BEFORE decoding when all output
segments already exist (no decode, no source removal); (b) when segments are
missing, only those are exported, the decode happens once, and the source is
removed exactly as before; (c) the set of files written/skipped matches the old
(decode-first) implementation.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

import src.recovery_from_meta as rfm


class FakeSegment:
    def __init__(self, decoder, span=None):
        self.decoder = decoder
        self.span = span

    def __getitem__(self, sl):
        return FakeSegment(self.decoder, (sl.start, sl.stop))

    def export(self, dest_path, format="mp3"):
        self.decoder.exported.append((self.span, dest_path))
        with open(dest_path, "wb") as f:
            f.write(b"FAKEMP3")


class FakeDecoder:
    def __init__(self):
        self.calls = 0
        self.exported = []

    def from_mp3(self, audio_path):
        self.calls += 1
        return FakeSegment(self)


def _meta(playlist_id, podcast_id, spans):
    return pd.DataFrame(
        {
            "playlist_id": [playlist_id] * len(spans),
            "podcast_id": [podcast_id] * len(spans),
            "start": [s for s, _ in spans],
            "end": [e for _, e in spans],
        }
    )


@pytest.fixture
def patched(monkeypatch):
    decoder = FakeDecoder()
    monkeypatch.setattr(rfm, "AudioSegment", decoder)
    return decoder


def _setup(tmp_path):
    src_dir = tmp_path / "src" / "11"
    src_dir.mkdir(parents=True)
    audio_path = src_dir / "22.mp3"
    audio_path.write_bytes(b"ORIGINAL")
    out_root = tmp_path / "out"
    out_root.mkdir()
    spans = [(0.0, 1.5), (1.5, 3.0)]
    df = _meta(11, 22, spans)
    grouped = df.groupby(["playlist_id", "podcast_id"])
    return str(audio_path), str(out_root), grouped, spans


def test_all_segments_exist_short_circuits(tmp_path, patched):
    audio_path, out_root, grouped, spans = _setup(tmp_path)
    dest_dir = os.path.join(out_root, "11", "22")
    os.makedirs(dest_dir)
    # Pre-create both output segments
    for s, e in spans:
        name = f"{s:.2f}_{e:.2f}_11_22.mp3"
        open(os.path.join(dest_dir, name), "wb").write(b"DONE")

    rfm.process_audio_file(audio_path, out_root, grouped)

    assert patched.calls == 0, "decoder must NOT be invoked when all exist"
    assert patched.exported == []
    assert os.path.exists(audio_path), "source kept when nothing exported"


def test_missing_segments_export_then_remove(tmp_path, patched):
    audio_path, out_root, grouped, spans = _setup(tmp_path)

    rfm.process_audio_file(audio_path, out_root, grouped)

    assert patched.calls == 1, "decode happens once when there is work"
    # Both segments exported
    assert len(patched.exported) == 2
    dest_dir = os.path.join(out_root, "11", "22")
    for s, e in spans:
        name = f"{s:.2f}_{e:.2f}_11_22.mp3"
        assert os.path.exists(os.path.join(dest_dir, name))
    assert not os.path.exists(audio_path), "source removed after export"


def test_partial_existing_only_missing_exported(tmp_path, patched):
    audio_path, out_root, grouped, spans = _setup(tmp_path)
    dest_dir = os.path.join(out_root, "11", "22")
    os.makedirs(dest_dir)
    # Pre-create the FIRST segment only
    s0, e0 = spans[0]
    open(os.path.join(dest_dir, f"{s0:.2f}_{e0:.2f}_11_22.mp3"), "wb").write(b"DONE")

    rfm.process_audio_file(audio_path, out_root, grouped)

    assert patched.calls == 1
    # Only the second span exported (ms-converted bounds)
    exported_paths = [p for _, p in patched.exported]
    s1, e1 = spans[1]
    expected = os.path.join(dest_dir, f"{s1:.2f}_{e1:.2f}_11_22.mp3")
    assert exported_paths == [expected]
    # Span passed to the slice is ms-converted
    assert patched.exported[0][0] == (int(s1 * 1000), int(e1 * 1000))
    assert not os.path.exists(audio_path)


def test_meta_not_found_no_decode(tmp_path, patched):
    audio_path, out_root, _, _ = _setup(tmp_path)
    # group key (11,22) absent
    empty = _meta(99, 99, [(0.0, 1.0)]).groupby(["playlist_id", "podcast_id"])
    rfm.process_audio_file(audio_path, out_root, empty)
    assert patched.calls == 0
    assert os.path.exists(audio_path)
