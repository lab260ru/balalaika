"""Standalone probe: run one existing-chunks stage iteration on a fixture.

Used by ``test_existing_chunks_single_read.py`` under strace to count real
``openat`` syscalls against the fixture path. ``reuse`` selects byte-reuse
(loader bytes feed the native decode) vs the legacy double-decode path.

Usage: python tests/_open_count_probe.py <audio_path> <reuse:0|1>
"""
import sys

import torch  # noqa: F401  (import side effects happen before MARKER)


def main() -> None:
    audio_path = sys.argv[1]
    reuse = sys.argv[2] == "1"

    from src.preprocess import preprocess_existing_chunks as pec
    from src.utils.datasets.preprocess import DiarizationDataset

    config = {
        "crest_threshold": 1000.0,
        "peak": -1.0,
        "loudness": -23.0,
        "block_size": 0.400,
        "fuse_audio_preprocessing": True,
    }

    cap = 60.0 if reuse else None
    # MARKER below fences the measured region from interpreter/library startup
    # opens (imports, .so loading, font caches, ...). strace counts only opens
    # of the fixture path emitted after this line is flushed to stderr.
    sys.stderr.write("PROBE_MARKER_BEGIN\n")
    sys.stderr.flush()

    ds = DiarizationDataset([audio_path], raw_bytes_max_duration_s=cap)
    _, wav, sr, err, raw_bytes = ds[0]
    assert err == "" and wav.numel() > 0
    pec._postprocess_existing_chunk(audio_path, config, raw_bytes)

    sys.stderr.write("PROBE_MARKER_END\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
