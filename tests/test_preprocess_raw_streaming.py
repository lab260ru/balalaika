"""Raw-mode streaming: rows go out incrementally with bounded worker RAM.

The Sortformer/SmartTurn ONNX models are absent on this node, so the model layer
(``process_audio_file``) and the source dataloader are faked deterministically.
The test pins that ``_run_diarization_shard``:

* writes every produced chunk row to the partial CSV as it goes (never holds the
  full row list — it returns only aggregates), and
* returns row count + duration sum that exactly match what was streamed, and
* yields a final balalaika.csv (via the same ``absorb_partial_csvs`` machinery
  the stage uses) identical to one built from the same rows in one shot.
"""

import pandas as pd
import torch

import src.preprocess.preprocess as P
from src.utils.csv_manager import (
    absorb_partial_csvs,
    ensure_main_csv,
    read_partial_csvs,
)


def _fake_segments(path, n_chunks):
    rows = []
    for i in range(n_chunks):
        rows.append(
            {
                "filepath": f"{path}__chunk{i}.flac",
                "speaker_id": i % 4,
                "start": round(i * 1.5, 2),
                "end": round(i * 1.5 + 1.0, 2),
                "total_duration": round(1.0 + 0.1 * i, 2),
                "playlist_id": "album",
                "podcast_id": "ep",
                "silence_percent": 12.5,
                "max_silence_duration": 0.4,
                "is_single_speaker": (i % 2 == 0),
            }
        )
    return rows


def _install_fakes(monkeypatch, files_to_chunks):
    """Fake the dataloader (one batch of synthetic items) and the model layer."""

    def fake_loader(file_paths, **kwargs):
        # one batch containing every file as (path, audio, sr, error, raw_bytes)
        batch = [
            (str(p), torch.zeros((1, 16000), dtype=torch.float32), 16000, "", None)
            for p in file_paths
        ]
        return [batch]

    def fake_process(path_audio, audio, sr, config, raw_bytes=None):
        n = files_to_chunks[str(path_audio)]
        return {
            "segments": _fake_segments(str(path_audio), n),
            "source_duration_s": 10.0,
            "crest_audit": P._new_crest_audit(),
        }

    monkeypatch.setattr(P, "create_diarization_dataloader", fake_loader)
    monkeypatch.setattr(P, "process_audio_file", fake_process)


def test_shard_streams_rows_and_returns_aggregates(tmp_path, monkeypatch):
    files = [str(tmp_path / f"src{i}.flac") for i in range(5)]
    files_to_chunks = {f: (i + 1) for i, f in enumerate(files)}  # 1..5 chunks
    _install_fakes(monkeypatch, files_to_chunks)

    config = {"fuse_audio_preprocessing": False, "diarization_batch_size": 1}
    result = P._run_diarization_shard(
        gpu_id=0,
        gpu_files=files,
        config=config,
        num_loader_workers=0,
        podcasts_path=str(tmp_path),
    )

    # Returns ONLY aggregates — no row list retained.
    assert set(result.keys()) == {"rows", "duration_sum_s", "crest_audit"}
    expected_rows = sum(files_to_chunks.values())  # 1+2+3+4+5 = 15
    assert result["rows"] == expected_rows

    # The partial CSV holds every streamed row.
    partial_df = read_partial_csvs(str(tmp_path), P.PARTIAL_PREFIX)
    assert len(partial_df) == expected_rows

    # Duration sum matches the rows that were written.
    expected_duration = sum(
        row["total_duration"]
        for f in files
        for row in _fake_segments(f, files_to_chunks[f])
    )
    assert abs(result["duration_sum_s"] - expected_duration) < 1e-6
    csv_duration = partial_df["total_duration"].astype(float).sum()
    assert abs(csv_duration - expected_duration) < 1e-6


def test_final_csv_identical_to_one_shot(tmp_path, monkeypatch):
    files = [str(tmp_path / f"src{i}.flac") for i in range(4)]
    files_to_chunks = {f: 3 for f in files}
    _install_fakes(monkeypatch, files_to_chunks)

    ensure_main_csv(str(tmp_path))
    config = {"fuse_audio_preprocessing": False}
    P._run_diarization_shard(
        gpu_id=0, gpu_files=files, config=config,
        num_loader_workers=0, podcasts_path=str(tmp_path),
    )
    value_columns = [c for c in P.PARTIAL_FIELDS if c != "filepath"]
    main_df, absorbed = absorb_partial_csvs(
        str(tmp_path), P.PARTIAL_PREFIX, value_columns=value_columns, preserve_existing=True
    )

    expected_rows = sum(files_to_chunks.values())
    assert absorbed == expected_rows

    # Build the reference set of (filepath, total_duration) pairs and compare.
    expected = {
        row["filepath"]: row["total_duration"]
        for f in files
        for row in _fake_segments(f, files_to_chunks[f])
    }
    got_df = pd.read_parquet(tmp_path / "balalaika.parquet")
    got = {
        row.filepath: float(row.total_duration)
        for row in got_df.itertuples()
    }
    assert got == expected


def test_empty_shard_returns_zero(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, {})
    result = P._run_diarization_shard(
        gpu_id=0, gpu_files=[], config={"fuse_audio_preprocessing": False},
        num_loader_workers=0, podcasts_path=str(tmp_path),
    )
    assert result["rows"] == 0
    assert result["duration_sum_s"] == 0.0

def test_single_speaker_only_drops_short_multispeaker_source(tmp_path, monkeypatch):
    source = tmp_path / "short.flac"
    source.write_bytes(b"x")
    monkeypatch.setattr(
        P,
        "diarize_audio",
        lambda audio, sr, chunk_duration: [(0.0, 0.5, 0), (0.5, 1.0, 1)],
    )

    result = P.process_audio_file(
        str(source),
        torch.zeros((1, 16000), dtype=torch.float32),
        16000,
        {"duration": 15, "single_speaker_only": True, "fuse_audio_preprocessing": False},
    )

    assert result["segments"] == []
    assert not source.exists()
    assert result["crest_audit"]["single_speaker_rejections"] == 1


def test_cut_audio_single_speaker_only_skips_multispeaker_window_before_decode(tmp_path, monkeypatch):
    class Metadata:
        sample_rate = 16000
        duration_seconds = 2.0

    class FakeDecoder:
        metadata = Metadata()

        def __init__(self, path):
            self.path = path

        def get_samples_played_in_range(self, **kwargs):
            raise AssertionError("multi-speaker window should be skipped before decode")

    monkeypatch.setattr(P, "AudioDecoder", FakeDecoder)
    audit = P._new_crest_audit()

    rows = P.cut_audio(
        str(tmp_path / "src.flac"),
        [(0.0, 2.0, 0)],
        [(0.0, 1.0, 0), (1.0, 2.0, 1)],
        str(tmp_path / "out"),
        "album",
        "episode",
        config={"single_speaker_only": True},
        crest_audit=audit,
    )

    assert rows == []
    assert audit["single_speaker_rejections"] == 1

