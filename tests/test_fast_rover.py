"""FastROVER must agree with crowd-kit's ROVER character-for-character.

These tests pin the fast numba implementation to the stock crowd-kit
behavior on randomized ASR-like corpora and on hand-built edge cases
that exercise every tie-break the algorithm has (option ordering in the
DP, zero-cost deletions against empty-token edge sets, and the
(count, len, word) voting rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from crowdkit.aggregation import ROVER

from src.transcription.fast_rover import FastROVER

TOKENIZER = staticmethod(lambda s: s.lower().split())
DETOKENIZER = staticmethod(lambda tokens: " ".join(tokens))


def aggregate_both(records: list[dict]) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame.from_records(records, columns=["task", "worker", "text"])
    tokenizer = lambda s: s.lower().split()  # noqa: E731
    detokenizer = lambda tokens: " ".join(tokens)  # noqa: E731
    stock = ROVER(tokenizer, detokenizer).fit_predict(df.copy())
    fast = FastROVER(tokenizer, detokenizer).fit_predict(df.copy())
    return stock, fast


def assert_equivalent(records: list[dict]) -> None:
    stock, fast = aggregate_both(records)
    assert stock.index.equals(fast.index)
    mismatched = stock[stock.values != fast.values]
    assert mismatched.empty, f"outputs differ for: {dict(mismatched.head())}"


def records_for(task: str, texts: list[str]) -> list[dict]:
    return [
        {"task": task, "worker": f"m{i}", "text": text}
        for i, text in enumerate(texts)
    ]


def test_single_hypothesis() -> None:
    assert_equivalent(records_for("a", ["привет как дела"]))


def test_identical_hypotheses() -> None:
    assert_equivalent(records_for("a", ["раз два три"] * 5))


def test_empty_and_whitespace_texts() -> None:
    records = (
        records_for("a", ["", "привет мир", "привет мир"])
        + records_for("b", ["   ", "", " "])
        + records_for("c", ["слово", ""])
        + records_for("d", ["", "слово"])
    )
    assert_equivalent(records)


def test_repeated_words_within_hypothesis() -> None:
    assert_equivalent(
        records_for("a", ["да да нет да", "да нет нет", "да да да да да"])
    )


def test_voting_tiebreak_by_length_then_lexicographic() -> None:
    # Substitutions at the same position with equal counts: the vote must
    # fall back to word length, then to lexicographic order.
    records = records_for("a", ["аб", "вг", "где"]) + records_for(
        "b", ["аа", "бб", "вв", "гг"]
    )
    assert_equivalent(records)


def test_insertion_deletion_heavy() -> None:
    assert_equivalent(
        records_for(
            "a",
            [
                "один два три четыре пять",
                "два три",
                "ноль один два три четыре пять шесть семь",
                "пять",
            ],
        )
    )


def test_disjoint_hypotheses() -> None:
    assert_equivalent(records_for("a", ["а б в", "г д е", "ж з и"]))


def test_long_texts() -> None:
    rng = np.random.default_rng(7)
    vocab = [f"слово{i}" for i in range(50)]
    texts = [
        " ".join(vocab[i] for i in rng.integers(0, len(vocab), size=200))
        for _ in range(5)
    ]
    assert_equivalent(records_for("a", texts))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_randomized_corpus(seed: int) -> None:
    from benchmarking.micro.bench_rover import synth_tasks

    df = synth_tasks(150, seed=seed)
    assert_equivalent(df.to_dict("records"))


def test_wrapper_uses_fast_path_and_falls_back(monkeypatch) -> None:
    from src.transcription.rover import ROVERWrapper

    wrapper = ROVERWrapper(podcasts_path=".", model_names=["m0"])
    from src.transcription.fast_rover import FastROVER as FR

    assert isinstance(wrapper._make_aggregator(), FR)

    wrapper_slow = ROVERWrapper(
        podcasts_path=".", model_names=["m0"], use_fast_rover=False
    )
    assert isinstance(wrapper_slow._make_aggregator(), ROVER)

def test_asr_consistency_word_wer_formula() -> None:
    from src.transcription.rover import asr_consistency_from_transcripts

    score = asr_consistency_from_transcripts(
        ["a b c", "a b x", "a b c"],
        "a b c",
    )

    assert score == pytest.approx(((1.0 + (2.0 / 3.0) + 1.0) / 3.0) * 100.0)
    assert asr_consistency_from_transcripts(["a b", ""], "a b") == pytest.approx(50.0)
    assert asr_consistency_from_transcripts(["Елка, привет!", "ёлка привет"], "ёлка привет") == pytest.approx(100.0)
    assert asr_consistency_from_transcripts(["a b c"], "a b c") is None
    assert asr_consistency_from_transcripts(["a", "b"], "") is None


def test_wrapper_saves_asr_consistency(tmp_path, monkeypatch) -> None:
    from src.transcription.rover import ROVERWrapper
    from src.utils.chunk_json import chunk_json_path, read_chunk_json, update_chunk_json

    audio = tmp_path / "a.flac"
    audio.write_bytes(b"x")
    update_chunk_json(
        audio,
        {"asr": {"m0": "a b c", "m1": "a b x", "m2": "a b c"}},
    )

    class FakeAggregator:
        def fit_predict(self, df):
            return pd.Series({str(audio): "a b c"})

    wrapper = ROVERWrapper(str(tmp_path), ["m0", "m1", "m2"])
    monkeypatch.setattr(wrapper, "_make_aggregator", lambda: FakeAggregator())

    seen, tasks, saved = wrapper._aggregate_audio_paths([str(audio)], "unit")

    assert (seen, tasks, saved) == (1, 1, 1)
    data = read_chunk_json(chunk_json_path(audio))
    assert data["rover"] == "a b c"
    assert data["asr_consistency"] == pytest.approx(88.8888888889)


def test_rover_pending_requires_asr_consistency_key(tmp_path) -> None:
    from src.transcription.rover import ROVERWrapper
    from src.utils.chunk_json import update_chunk_json

    audio = tmp_path / "pending.flac"
    audio.write_bytes(b"x")
    wrapper = ROVERWrapper(str(tmp_path), ["m0", "m1"])

    update_chunk_json(audio, {"rover": "a b"})
    assert wrapper._pending_audio_paths([str(audio)]) == [str(audio)]

    update_chunk_json(audio, {"asr_consistency": ""})
    assert wrapper._pending_audio_paths([str(audio)]) == []

