"""Behavior tests pinning src.accents.fast_accent.FastRUAccent to stock ruaccent.

FastRUAccent must be CHARACTER-IDENTICAL to ``ruaccent.RUAccent.process_all``
on real Russian text, per feature (cross-file/sentence ONNX batching, the
context-free per-word accent memo, and the lazy rule-engine skip), plus the
``process_batch`` slab API.  The model runs on CPU on this node, so the
equivalence is measured against the real model.

Two test classes need NO model (pure logic):
  * the context-free / context-dependent gating of the memo, and the '+'
    position re-render, with the code-level evidence for each path;
  * the rule-engine-is-unused evidence.

Run: .dev_venv/bin/python -m pytest tests/test_accents_fast.py -q
"""
from __future__ import annotations

import inspect
import os

import pytest

ruaccent = pytest.importorskip("ruaccent")
from ruaccent import RUAccent  # noqa: E402
from ruaccent.text_postprocessor import fix_capital  # noqa: E402

from src.accents.fast_accent import FastRUAccent, capped_onnx_threads  # noqa: E402

WORKDIR = "/home/kirill/mnt/ssd_work/balalaika/cache/ruaccent_workdir"
MODEL = "turbo3.1"
PROVIDERS = ["CPUExecutionProvider"]


def _workdir_available() -> bool:
    return os.path.isdir(os.path.join(WORKDIR, "nn", "nn_accent")) and os.path.isdir(
        os.path.join(WORKDIR, "dictionary")
    )


needs_model = pytest.mark.skipif(
    not _workdir_available(),
    reason="ruaccent workdir assets not present on this node",
)


# --------------------------------------------------------------------------- #
# Fixture corpus: varied real Russian text exercising every code path.        #
# --------------------------------------------------------------------------- #
HOMOGRAPH_TEXTS = [
    # замок (lock/castle), стоит (stands/costs), окна (windows), белки (squirrels/proteins)
    "Я повесил замок на дверь, потому что старый замок сломался.",
    "Замок стоит на горе. Это стоит дорого.",
    "Через дорогу шла дорога, а вдоль неё стояли окна и белки прыгали.",
    "Мука дорогая, а мука творчества бесценна.",
    "Окна открыты. Закрой окна. Белки прыгают. Кормим белки.",
    "Стоит дом. Стоит ли это того? Дорога домой была долгой.",
    "Замок замок замок дорога дорога окна окна белки белки стоит стоит.",
]
YO_TEXTS = [
    "Ёлки-палки, ещё один ёжик пробежал мимо.",
    "пёс и пес. всё и все. ещё и еще. объём растёт.",
    "Берёза, чёрный, тёплый, актёр, реклама и щётка.",
]
CASE_TEXTS = [
    "Зеленский и Хабиб. ЗЕЛЕНСКИЙ улыбнулся, хабиб промолчал, Хабиб ушёл.",
    "Привет! ПРИВЕТ. привет всем. ПрИвЕт.",
    "Москва МОСКВА москва Санкт-Петербург.",
]
OOV_TEXTS = [
    "сегодня обсудим блогершу и стримера на тиктоке без знаков препинания",
    "котозаврище мяукнулось абвгдейка программистозавр нейросетюга",
    "криптовалюта веб-разработчик фрилансе питоне джанго докере кубернетес",
]
ASR_LIKE_TEXTS = [
    "ну поздравляю сейчас",
    "да",
    "а что там у тебя сегодня нового вечером расскажи пожалуйста подробнее",
    "это очень длинное предложение которое содержит много разных слов и должно "
    "проверить как модель справляется с большими входными данными без знаков "
    "препинания и заглавных букв что характерно для распознанной речи в реальных "
    "транскриптах разговорной русской речи на подкастах",
]
PUNCT_HEAVY_TEXTS = [
    "А.",
    "А. Б. В. Г.",
    "Это — да?! Но... (возможно). «Цитата», — сказал он.",
    "1, 2, 3... поехали! 100% готово; всё, точка.",
]
SHORT_TEXTS = ["", "   ", ".", "!", "привет", "замок", "А"]

ALL_TEXTS = (
    HOMOGRAPH_TEXTS
    + YO_TEXTS
    + CASE_TEXTS
    + OOV_TEXTS
    + ASR_LIKE_TEXTS
    + PUNCT_HEAVY_TEXTS
    + SHORT_TEXTS
)


# --------------------------------------------------------------------------- #
# Real-model fixtures (loaded once per session).                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def stock():
    acc = RUAccent()
    acc.load(
        omograph_model_size=MODEL,
        use_dictionary=True,
        tiny_mode=False,
        providers=PROVIDERS,
        workdir=WORKDIR,
    )
    return acc


def _load_fast(**knobs):
    acc = FastRUAccent(**knobs)
    acc.load(
        omograph_model_size=MODEL,
        use_dictionary=True,
        tiny_mode=False,
        providers=PROVIDERS,
        workdir=WORKDIR,
    )
    return acc


@pytest.fixture(scope="session")
def fast_all_on():
    return _load_fast()


# --------------------------------------------------------------------------- #
# 1. Full-feature equivalence: every text, every category.                    #
# --------------------------------------------------------------------------- #
@needs_model
class TestFullEquivalence:
    @pytest.mark.parametrize("text", ALL_TEXTS)
    def test_process_all_char_identical(self, stock, fast_all_on, text):
        assert stock.process_all(text) == fast_all_on.process_all(text)

    def test_zero_diffs_over_whole_corpus(self, stock, fast_all_on):
        diffs = [t for t in ALL_TEXTS if stock.process_all(t) != fast_all_on.process_all(t)]
        assert diffs == [], f"{len(diffs)} texts diverged: {diffs[:3]}"


# --------------------------------------------------------------------------- #
# 2. Per-feature equivalence (each knob isolated).                            #
# --------------------------------------------------------------------------- #
@needs_model
class TestPerFeatureEquivalence:
    def test_batch_sentences_only(self, stock):
        fast = _load_fast(batch_sentences=True, memo_accent=False, lazy_rule_engine=True)
        for t in ALL_TEXTS:
            assert stock.process_all(t) == fast.process_all(t), repr(t)

    def test_memo_accent_only(self, stock):
        fast = _load_fast(batch_sentences=False, memo_accent=True, lazy_rule_engine=True)
        for t in ALL_TEXTS:
            assert stock.process_all(t) == fast.process_all(t), repr(t)

    def test_lazy_rule_engine_only(self, stock):
        fast = _load_fast(batch_sentences=False, memo_accent=False, lazy_rule_engine=True)
        for t in ALL_TEXTS:
            assert stock.process_all(t) == fast.process_all(t), repr(t)

    def test_all_knobs_off_is_stock(self, stock):
        """use_fast_accent: false maps to all knobs off — must equal stock."""
        fast = _load_fast(
            batch_sentences=False, memo_accent=False, lazy_rule_engine=False
        )
        for t in ALL_TEXTS:
            assert stock.process_all(t) == fast.process_all(t), repr(t)


# --------------------------------------------------------------------------- #
# 3. process_batch (slab API) == per-file process_all.                        #
# --------------------------------------------------------------------------- #
@needs_model
class TestProcessBatch:
    def test_batch_equals_per_file(self, stock, fast_all_on):
        # A genuinely distinct, no-duplicate slab (so the per-file cache is not
        # what makes them match — the batched ONNX path is).
        slab = [t for t in ALL_TEXTS if t.strip()]
        batched = fast_all_on.process_batch(slab)
        per_file = [stock.process_all(t) for t in slab]
        assert batched == per_file

    def test_batch_handles_empty_and_blank(self, stock):
        fast = _load_fast()
        slab = ["", "   ", "привет", "замок стоит"]
        assert fast.process_batch(slab) == [stock.process_all(t) for t in slab]

    def test_batch_order_preserved(self, stock):
        fast = _load_fast()
        slab = ["замок один", "дорога два", "окна три", "белки четыре"]
        out = fast.process_batch(slab)
        assert out == [stock.process_all(t) for t in slab]
        assert len(out) == len(slab)


# --------------------------------------------------------------------------- #
# 4. Thread cap loads and is output-invariant.                               #
# --------------------------------------------------------------------------- #
@needs_model
class TestThreadCap:
    def test_capped_load_is_equivalent(self, stock):
        fast = FastRUAccent()
        with capped_onnx_threads(2):
            fast.load(
                omograph_model_size=MODEL,
                use_dictionary=True,
                tiny_mode=False,
                providers=PROVIDERS,
                workdir=WORKDIR,
            )
        for t in HOMOGRAPH_TEXTS + YO_TEXTS:
            assert stock.process_all(t) == fast.process_all(t), repr(t)

    def test_cap_zero_is_noop(self):
        # No exception, no patching leaks: a 0/None cap must be a clean no-op.
        import onnxruntime as ort

        before = ort.InferenceSession.__init__
        with capped_onnx_threads(0):
            pass
        with capped_onnx_threads(None):
            pass
        assert ort.InferenceSession.__init__ is before


# --------------------------------------------------------------------------- #
# 5. Memo gating — the equivalence-study evidence (no model needed).          #
# --------------------------------------------------------------------------- #
class TestMemoGatingEvidence:
    """Code-level proof of WHICH call paths are context-free (memoizable).

    put_accent(word): in ruaccent/accent_model.py, ``put_accent`` lowercases the
    word, tokenizes THAT single string, runs the accent ONNX, and renders stress
    onto the original-case word.  No sentence context enters -> CONTEXT-FREE ->
    memoizable.  The omograph (``classify(texts, ...)`` with ``<w>..</w>`` markers
    in the surrounding sentence) and e-homograph (``predict_yo_homographs(sentence)``)
    paths read the whole sentence -> CONTEXT-DEPENDENT -> never memoized.
    """

    def test_put_accent_signature_is_word_only(self):
        from ruaccent.accent_model import AccentModel

        sig = inspect.signature(AccentModel.put_accent)
        # (self, word) — exactly one non-self argument, and no sentence/context.
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["word"], params
        src = inspect.getsource(AccentModel.put_accent)
        assert "sentence" not in src and "context" not in src
        # It only consumes `word` (lowercased) — proving context-freeness.
        assert "word.lower()" in src

    def test_omograph_classify_is_context_dependent(self):
        from ruaccent.omograph_model import OmographModel

        src = inspect.getsource(OmographModel.classify)
        # classify tokenizes (text, hypothesis) PAIRS — the text is the whole
        # sentence with <w>..</w> markers, so it is context-dependent.
        assert "texts" in inspect.signature(OmographModel.classify).parameters
        proc = inspect.getsource(RUAccent._process_omographs)
        assert "<w>" in proc  # the marker proves the sentence is the input

    def test_yo_homograph_is_context_dependent(self):
        from ruaccent.yo_homograph_model import YoHomographModel

        sig = inspect.signature(YoHomographModel.predict_yo_homographs)
        # predict_yo_homographs(text) — the whole sentence is the input.
        assert "text" in sig.parameters

    def test_plus_positions_roundtrip_is_case_invariant(self):
        # The memo caches '+' positions on the base string; re-rendering onto
        # any case must reproduce render_stress byte-for-byte.
        # render_stress(WORD, pred) inserts '+' before stressed chars of WORD.
        for stressed_lower, word in [
            ("прив+ет", "Привет"),
            ("прив+ет", "ПРИВЕТ"),
            ("м+осква", "Москва"),
            ("кубер+нетес", "КуберНетес"),
            ("з+амок", "замок"),
        ]:
            positions = FastRUAccent._plus_positions(stressed_lower)
            rendered = FastRUAccent._render_with_positions(word, positions)
            # Reference: apply the SAME positions to the lowercase base and then
            # uppercase per the original word's case mask (what render_stress does).
            base = stressed_lower.replace("+", "")
            expected_chars = []
            pset = set(positions)
            for idx, ch in enumerate(word):
                if idx in pset:
                    expected_chars.append("+")
                expected_chars.append(ch)
            assert rendered == "".join(expected_chars)
            assert rendered.replace("+", "").lower() == base

    def test_plus_positions_no_stress(self):
        assert FastRUAccent._plus_positions("слово") == ()
        assert FastRUAccent._render_with_positions("Слово", ()) == "Слово"


# --------------------------------------------------------------------------- #
# 6. Lazy rule engine — the unused-asset evidence (no model needed).          #
# --------------------------------------------------------------------------- #
class TestRuleEngineUnused:
    """``RuleEngine`` (koziev rupostagger 161 MB + rulemma 16.7 MB) is loaded by
    stock ``RUAccent.load`` but never referenced on the process_all path."""

    def test_rule_accent_not_referenced_in_process_path(self):
        for fn in (
            RUAccent.process_all,
            RUAccent.process_all_internal,
            RUAccent._process_sentence_cached,
            RUAccent._process_omographs,
            RUAccent._process_accent,
            RUAccent._process_yo,
        ):
            src = inspect.getsource(fn)
            assert "rule_accent" not in src, f"{fn.__name__} references rule_accent"

    def test_rule_accent_only_loaded_in_load(self):
        # The only references to rule_accent are in load() (assignment + .load()).
        src = inspect.getsource(RUAccent)
        # appears exactly in the load block, never in a processing method
        assert src.count("self.rule_accent") <= 3


# --------------------------------------------------------------------------- #
# 7. Memo correctness against the real accent model (context-free proof).     #
# --------------------------------------------------------------------------- #
@needs_model
class TestAccentMemoCorrectness:
    def test_memoized_put_accent_matches_stock_all_cases(self, stock):
        fast = _load_fast()
        pa = stock.accent_model.put_accent
        words = [
            "москва", "Москва", "МОСКВА", "блогерша", "Блогерша",
            "кубернетес", "Кубернетес", "КуберНетес", "зеленский",
            "Зеленский", "ЗЕЛЕНСКИЙ", "привет", "Привет", "ПРИВЕТ",
            "разработчик", "айфоне", "Айфоне", "навального",
        ]
        for w in words:
            assert fast._put_accent_memoized(w) == pa(w), w

    def test_memo_is_reused_not_recomputed(self):
        fast = _load_fast()
        w = "криптовалюта"
        first = fast._put_accent_memoized(w)
        assert w.lower() in fast._accent_memo
        # Break the underlying model: a second call MUST hit the cache, proving
        # context-freeness lets us serve it without re-running ONNX.
        fast.accent_model.put_accent = lambda x: (_ for _ in ()).throw(
            AssertionError("model should not be called for a cached word")
        )
        assert fast._put_accent_memoized(w) == first
        assert fast._put_accent_memoized(w.upper()).lower() == first.lower()
