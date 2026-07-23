"""Pin DistillMOS segment-batched scoring to the stock single-file path.

The production dataloader segments each file itself (``distillmos_segments``,
computed from the file's true length) and the stage runs the model with
``segmenting_in_forward=False`` plus a per-file mean over segment scores.
These tests pin that path to the stock upstream path — one file at a time
through ``segmenting_in_forward=True`` — which is by construction independent
of batch composition. The old ``pad_sequence`` collate was not: the model
derived its crop grid from the longest file in the batch, shifting every
other file's windows and scoring windows of pure padding (measured mean
|ΔMOS| ≈ 0.03, max 0.35 at batch_size=8).

CPU-only and deterministic; skipped when the ``distillmos`` package is not
installed.
"""

import pytest

torch = pytest.importorskip("torch")
distillmos = pytest.importorskip("distillmos")

from src.utils.datasets.separation import (  # noqa: E402
    DISTILLMOS_SEQ_LEN,
    distillmos_collate,
    distillmos_segments,
)

# Lengths covering every segmentation regime: sub-segment (zero-padded to one
# crop), exactly one segment, barely over (2 crops), and multi-hop files.
LENGTHS = [
    DISTILLMOS_SEQ_LEN // 3,
    DISTILLMOS_SEQ_LEN,
    DISTILLMOS_SEQ_LEN + 1,
    int(12.5 * 16_000),
    15 * 16_000,
]


@pytest.fixture(scope="module")
def model():
    m = distillmos.ConvTransformerSQAModel()
    m.eval()
    return m


@pytest.fixture(scope="module")
def waveforms():
    gen = torch.Generator().manual_seed(0)
    return [torch.rand(n, generator=gen) * 2 - 1 for n in LENGTHS]


def _stock_scores(model, waveforms):
    """Upstream reference: one file at a time, model segments internally."""
    model.segmenting_in_forward = True
    scores = []
    with torch.inference_mode():
        for wav in waveforms:
            scores.append(model(wav.unsqueeze(0)).item())
    return scores


def _batched_scores(model, waveforms):
    """Production path: dataset segments, collate concatenates, stage means."""
    batch = distillmos_collate(
        [(f"file_{i}", distillmos_segments(w)) for i, w in enumerate(waveforms)]
    )
    _, segments, counts = batch
    assert segments.shape[-1] == DISTILLMOS_SEQ_LEN
    model.segmenting_in_forward = False
    with torch.inference_mode():
        seg_mos = model(segments).detach().flatten()
    scores, offset = [], 0
    for count in counts:
        scores.append(seg_mos[offset : offset + count].mean().item())
        offset += count
    return scores


def test_segment_counts_match_model_grid():
    # num_hops = ceil(max(len - SEQ_LEN, 0) / 16000) + 1
    expected = [1, 1, 2, 6, 9]
    got = [distillmos_segments(torch.zeros(n)).shape[0] for n in LENGTHS]
    assert got == expected


def test_batched_path_matches_stock_single_file(model, waveforms):
    stock = _stock_scores(model, waveforms)
    batched = _batched_scores(model, waveforms)
    for s, b in zip(stock, batched):
        assert b == pytest.approx(s, abs=1e-5)


def test_score_independent_of_batch_composition(model, waveforms):
    # Scoring a file alone must equal scoring it co-batched with any others.
    together = _batched_scores(model, waveforms)
    alone = [_batched_scores(model, [w])[0] for w in waveforms]
    for a, t in zip(alone, together):
        assert t == pytest.approx(a, abs=1e-5)


def test_perf_defaults_respect_disable_math_sdp_flag():
    from src.utils.gpu import apply_torch_perf_defaults

    before = torch.backends.cuda.math_sdp_enabled()
    try:
        apply_torch_perf_defaults(disable_math_sdp=False)
        assert torch.backends.cuda.math_sdp_enabled()
        apply_torch_perf_defaults()
        assert not torch.backends.cuda.math_sdp_enabled()
    finally:
        torch.backends.cuda.enable_math_sdp(before)
