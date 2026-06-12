"""Behavior tests pinning src.phonemizer.fast_g2p.FastG2P to stock tryiparu.

FastG2P must be token-for-token identical to ``tryiparu.G2PModel`` — same
text output, same OOV greedy decode, same ValueError on oversize words, same
"words before the bad word are still cached" semantics. CPU only; large-scale
GPU equivalence (1200 OOV words, fp32 + TF32 regimes) is covered by
``benchmarking/micro/bench_g2p.py`` runs recorded in report.md.

Run: .dev_venv/bin/python -m pytest tests/test_phonemizer_fast_g2p.py -q
"""
from __future__ import annotations

import pickle

import pytest

from src.phonemizer.fast_g2p import FastG2P, _load_dictionary

VARIED_TEXTS = [
    "привет мир",
    "съешь ещё этих мягких французских булок, да выпей же чаю",
    "Привет ПРИВЕТ привет",  # case folding collapses to one dict hit
    "ёжик ёлка объём",
    "мп3 плеер и 123 числа",
    "",
    "   ",
    "...",
    "слово",
    "а",
    "по-русски кто-нибудь говорит?",
    "котозаврище мяукнулось абвгдейка",  # OOV pseudo-words
    "точка . запятая , вопрос ?",
    "слово котозаврище слово котозаврище",  # repeated OOV
]

OOV_WORDS = [
    "котозаврище", "мяукнулось", "абвгдейка", "программистозавр",
    "балалайкинг", "фонемизаторный", "подкастёрша", "нейросетюга",
    "транскрибуляция", "ударениевед", "шумодавилка", "спектрограммка",
]


@pytest.fixture(scope="module")
def stock():
    from tryiparu import G2PModel

    return G2PModel(load_dataset=True, device="cpu")


@pytest.fixture(scope="module")
def fast():
    return FastG2P(device="cpu")


def test_matches_stock_on_varied_texts(stock, fast):
    for text in VARIED_TEXTS:
        assert stock(text) == fast(text), f"diverged on {text!r}"


def test_decode_batch_matches_stock_greedy(stock, fast):
    fast.data_dict = dict(fast.data_dict)  # don't pollute the module fixture
    for word in OOV_WORDS:
        fast.data_dict.pop(word, None)
    batched = fast.decode_batch(OOV_WORDS)
    for word in OOV_WORDS:
        assert batched[word] == stock.greedy_decode(src=word, max_length=64), word


def test_decode_batch_respects_batch_size(stock):
    small = FastG2P(device="cpu", batch_size=3)
    out = small.decode_batch(OOV_WORDS)
    for word in OOV_WORDS:
        assert out[word] == stock.greedy_decode(src=word, max_length=64), word


def test_too_long_word_value_error_parity(stock, fast):
    word = "ъыъ" * 30  # no BPE merges -> 90 tokens, far over MAX_LEN
    with pytest.raises(ValueError) as stock_err:
        stock.greedy_decode(src=word, max_length=stock.max_length)
    with pytest.raises(ValueError) as fast_err:
        fast.decode_batch([word])
    assert str(stock_err.value) == str(fast_err.value)


def test_words_before_oversize_word_are_cached(fast):
    fast.data_dict.pop("балалаечник", None)
    with pytest.raises(ValueError):
        fast.decode_batch(["балалаечник", "ъыъ" * 30, "неуспевайка"])
    assert "балалаечник" in fast.data_dict


def test_oov_cache_roundtrip(tmp_path):
    cache = tmp_path / "oov.pkl"
    one = FastG2P(device="cpu", oov_cache_path=str(cache))
    one.data_dict.pop("котозаврище", None)
    decoded = one.decode_batch(["котозаврище"])["котозаврище"]
    one.flush_oov_cache()
    assert cache.exists()

    two = FastG2P(device="cpu", oov_cache_path=str(cache))
    assert two.data_dict.get("котозаврище") == decoded


def test_boundary_length_word_decodes_where_stock_crashes(stock, fast):
    """Deliberate divergence: a word encoding to exactly MAX_LEN-2 BPE ids
    crashes stock (its empty pad tensor is float32 and torch.cat promotes the
    whole encoder input, so nn.Embedding rejects it); FastG2P decodes it."""
    word = "ы" * 62
    assert len(stock.tokenizer.encode(word).ids) == 62
    with pytest.raises(RuntimeError):
        stock(f"слово {word}")
    out = fast(f"слово {word}")
    assert out and isinstance(out, list)


def test_oov_cache_wrong_typed_payload_ignored(tmp_path):
    """A payload whose 'data' is not a dict must be treated as unreadable —
    not crash worker init forever (data_dict.update('abc') would raise)."""
    cache = tmp_path / "oov.pkl"
    probe = FastG2P(device="cpu", oov_cache_path=str(tmp_path / "other.pkl"))
    with open(cache, "wb") as f:
        pickle.dump({"key": probe._weights_key, "data": "abc"}, f)
    model = FastG2P(device="cpu", oov_cache_path=str(cache))
    assert isinstance(model.data_dict, dict) and "собака" in model.data_dict


def test_oov_cache_rejected_for_other_weights(tmp_path):
    cache = tmp_path / "oov.pkl"
    with open(cache, "wb") as f:
        pickle.dump({"key": ("not", "this", "model"), "data": {"x": ["y"]}}, f)
    model = FastG2P(device="cpu", oov_cache_path=str(cache))
    assert "x" not in model.data_dict


def test_dict_cache_built_and_reused(tmp_path):
    import tryiparu
    from pathlib import Path

    csv_path = Path(tryiparu.__file__).parent / "data" / "cleaned_dataset.csv"
    cache = tmp_path / "dict.pkl"
    first = _load_dictionary(csv_path, cache)
    assert cache.exists()
    second = _load_dictionary(csv_path, cache)
    assert first == second
    assert first["собака"] == "sɐbakə"

    cache.write_bytes(b"garbage")  # corrupted cache must rebuild, not crash
    third = _load_dictionary(csv_path, cache)
    assert third["собака"] == "sɐbakə"


def test_process_tokens_matches_rules(fast):
    from tryiparu.rules import process_text

    samples = [
        [],
        [" "],
        ["sɐbakə"],
        ["sɐbakə", " ", "zont", ",", " ", "ʊkas"],
        ["ˈabc", "ː", "(ː)", "⁽ʲ⁾", "t͡ɕa"],
        ["...", "!", "sɐbakə"],
    ]
    for tokens in samples:
        assert fast._process_tokens(tokens) == process_text(tokens), tokens
        # memoized second pass must not mutate cached entries
        assert fast._process_tokens(tokens) == process_text(tokens), tokens
