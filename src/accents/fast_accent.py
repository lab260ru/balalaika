"""Drop-in fast RUAccent: cross-sentence ONNX batching, context-free accent memo,
and lazy skip of the unused koziev/rulemma rule engine.

Same four ONNX sessions, tokenizers, dictionaries and *exact* per-sentence
decision logic as ``ruaccent.RUAccent`` (verified character-identical on the
``benchmarking/micro/bench_accents.py`` fixture corpus, including homographs in
disambiguating contexts, e-restoration, OOV/pseudo-words and ASR-like text).
The changes are purely mechanical and every one is a separately-toggleable knob:

1. **batch_sentences** — stock runs the stress-usage model, the e-homograph
   model and the omograph model as one ``InferenceSession.run`` *per sentence*
   (3-6 batch-1 calls/sentence).  FastRUAccent splits the whole text into
   sentences once, then runs ONE padded batch through the stress-usage model
   and ONE through the e-homograph model for all sentences, and collects every
   omograph hypothesis across all sentences into one padded batch.  ONNX
   attention masks make padded-batch logits bit-identical to the batch-1 logits
   (measured max abs diff 0.0 on all three models) — see the test module.

2. **memo_accent** — ``AccentModel.put_accent(word)`` depends ONLY on ``word``
   (it lowercases the word, tokenizes that single string, runs the char-level
   accent ONNX, and renders stress on the original-case word; no sentence
   context enters).  This is the one provably context-FREE call path, so its
   result is memoized per worker.  The omograph and e-homograph paths are
   context-DEPENDENT (they read the surrounding sentence) and are NEVER
   memoized.  See ``tests/test_accents_fast.py`` for the code-level evidence.

3. **lazy_rule_engine** — stock ``load()`` always instantiates ``RuleEngine``,
   which opens the koziev rupostagger (161 MB ``ruword2tags.db`` + a 2.4 MB CRF
   model), rulemma (16.7 MB) and the rule_engine JSONs — ~180 MB and ~5 s per
   worker.  ``self.rule_accent`` is never referenced by ``process_all`` /
   ``process_all_internal`` / ``_process_sentence_cached`` (grep-verified;
   pinned by a test), so FastRUAccent skips loading it.

The ONNX intra-op thread cap (item 4) lives in the stage worker init, not here,
because the upstream ``RUAccent.load`` API takes only ``providers`` and gives no
hook for ``SessionOptions``; it is set via ORT/OMP env before the sessions are
created.
"""
from __future__ import annotations

import contextlib
import re
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from ruaccent import RUAccent
from ruaccent.text_postprocessor import fix_capital
from ruaccent.text_preprocessor import TextPreprocessor


@contextlib.contextmanager
def capped_onnx_threads(intra_op_threads: Optional[int]):
    """Force every ``onnxruntime.InferenceSession`` created in this block to use
    at most ``intra_op_threads`` intra-op threads (item 4 thread cap).

    ruAccent's ``load()`` API exposes no ``SessionOptions`` hook, and each of its
    four sessions otherwise defaults ``intra_op_num_threads`` to all 48 logical
    cores.  With multiple workers (sessions x workers) that oversubscribes the
    box and thrashes the two NUMA nodes.  We inject a capped ``SessionOptions``
    by wrapping ``InferenceSession.__init__`` for the duration of the load only;
    when ``intra_op_threads`` is None/<=0 this is a no-op (stock behavior)."""
    if not intra_op_threads or intra_op_threads <= 0:
        yield
        return

    import onnxruntime as ort

    real_init = ort.InferenceSession.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001
        if kwargs.get("sess_options") is None and len(args) < 2:
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(intra_op_threads)
            kwargs["sess_options"] = so
        return real_init(self, *args, **kwargs)

    ort.InferenceSession.__init__ = patched_init
    try:
        yield
    finally:
        ort.InferenceSession.__init__ = real_init


class FastRUAccent(RUAccent):
    """``ruaccent.RUAccent`` with cross-sentence ONNX batching + accent memo."""

    # cache mirrors stock's @lru_cache(maxsize=4096) on _process_sentence_cached,
    # keyed identically (the raw sentence string) and bounded the same way.
    _SENT_CACHE_MAX = 4096

    def __init__(
        self,
        batch_sentences: bool = True,
        memo_accent: bool = True,
        lazy_rule_engine: bool = True,
    ) -> None:
        super().__init__()
        self._batch_sentences = batch_sentences
        self._memo_accent = memo_accent
        self._lazy_rule_engine = lazy_rule_engine
        # context-free accent cache: lower_word -> tuple of '+' insertion
        # positions in the base string.  Positions are case-invariant (the
        # accent model lowercases internally), so they re-render exactly onto
        # any original-case spelling of the word.
        self._accent_memo: Dict[str, tuple] = {}
        self._sent_cache_dict: Dict[str, str] = {}
        self._sent_model_preds: Dict[str, dict] = {}
        self._omo_logits_by_sentence: Dict[str, tuple] = {}

    # ------------------------------------------------------------------ #
    # load — optionally skip the unused rule engine (item 3)             #
    # ------------------------------------------------------------------ #
    def load(self, *args, **kwargs):
        if not self._lazy_rule_engine:
            return super().load(*args, **kwargs)

        # Temporarily neutralize RuleEngine.load so the koziev / rulemma assets
        # are never read.  Restored afterwards so a second non-lazy loader in
        # the same process is unaffected.
        import ruaccent.rule_accent_engine as rae

        real_load = rae.RuleEngine.load
        skipped = {"hit": False}

        def _skip_load(self_re, path):  # noqa: ANN001
            skipped["hit"] = True  # never touches koziev/rulemma/rule_engine

        rae.RuleEngine.load = _skip_load
        try:
            super().load(*args, **kwargs)
        finally:
            rae.RuleEngine.load = real_load
        if skipped["hit"]:
            logger.debug("FastRUAccent: skipped unused koziev/rulemma rule engine load")
        return None

    # ------------------------------------------------------------------ #
    # context-free accent memo (item 2)                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _plus_positions(stressed: str) -> tuple:
        """Indices in the base (no-'+') string before which a '+' is inserted.

        ``render_stress`` builds the stressed string by inserting '+' before the
        stressed character of the ORIGINAL word; the set of stressed positions
        depends only on the lowercase spelling (the model lowercases its input).
        Recovering positions from a stressed string is exact and case-invariant.
        """
        positions = []
        base_i = 0
        for ch in stressed:
            if ch == "+":
                positions.append(base_i)
            else:
                base_i += 1
        return tuple(positions)

    @staticmethod
    def _render_with_positions(word: str, positions: tuple) -> str:
        if not positions:
            return word
        pset = set(positions)
        out = []
        for idx, ch in enumerate(word):
            if idx in pset:
                out.append("+")
            out.append(ch)
        return "".join(out)

    def _put_accent_memoized(self, word: str) -> str:
        """``AccentModel.put_accent`` with the '+' positions cached on
        ``word.lower()``.

        Stock ``put_accent`` lowercases the word, tokenizes that single string,
        runs the char-level accent ONNX, then renders stress back onto the
        ORIGINAL-case ``word`` via ``render_stress`` (inserting '+' before each
        stressed character).  The stress *positions* are a pure function of
        ``word.lower()`` — no sentence context — so we cache them per worker and
        re-render onto the actual word.  Verified byte-identical to stock
        ``put_accent(word)`` across upper/lower/mixed case (see test module).
        """
        lower_word = word.lower()
        positions = self._accent_memo.get(lower_word)
        if positions is None:
            stressed = self.accent_model.put_accent(lower_word)
            positions = self._plus_positions(stressed)
            if len(self._accent_memo) >= 200_000:
                # Crude RAM bound; the memo only saves time, never changes output.
                self._accent_memo.clear()
            self._accent_memo[lower_word] = positions
        return self._render_with_positions(word, positions)

    # ------------------------------------------------------------------ #
    # _process_accent — stock logic, accent call routed through the memo #
    # ------------------------------------------------------------------ #
    def _process_accent(self, text, stress_usages):
        if not self._memo_accent:
            return super()._process_accent(text, stress_usages)
        splitted_text = text
        for i, word in enumerate(splitted_text):
            if "+" in word:
                continue
            if stress_usages[i] == "STRESS":
                lower_word = word.lower()
                stressed_word = self.accents.get(lower_word, lower_word)
                if (
                    stressed_word == lower_word
                    and not self.has_punctuation(lower_word)
                    and self.count_vowels(lower_word) > 1
                ):
                    splitted_text[i] = self._put_accent_memoized(word)
                else:
                    match = re.finditer(r"\+", stressed_word)
                    word_fixed = list(word)
                    for j, e in enumerate(list(match)):
                        word_fixed = (
                            word_fixed[: e.start() + j] + ["+"] + list(word)[e.end() - 1 :]
                        )
                    splitted_text[i] = "".join(word_fixed)
        return splitted_text

    # ------------------------------------------------------------------ #
    # process_batch — batch the per-sentence ONNX calls ACROSS many files #
    # ------------------------------------------------------------------ #
    def process_batch(self, texts: List[str]) -> List[str]:
        """``process_all`` for a slab of ``texts`` (files) at once.

        Real ASR-chunk transcripts are ~1 sentence each, so the only place
        cross-sentence ONNX batching pays off is ACROSS files.  This collects
        every sentence of every text, runs ONE padded batch through the
        stress-usage and e-homograph models and ONE through the omograph model
        for the whole slab, then reassembles each text's output exactly as
        ``process_all`` would.  Character-identical to calling ``process_all``
        per text (verified on the fixture corpus); the only difference is batch
        composition, which is provably output-invariant for all three models.

        ``process_all``'s ``skip_regex`` argument is not supported here (the
        stage never uses it); call ``process_all`` for that path.
        """
        if not self._batch_sentences:
            return [self.process_all(t) for t in texts]

        # Normalize + split every text into sentences (stock process_all_internal
        # prefix), remembering which sentences belong to which text and in what
        # order so we can rejoin them.
        per_text_sentences: List[List[str]] = []
        all_new: List[str] = []
        seen = set()
        for text in texts:
            norm = re.sub(self.normalize, "", text)
            sents = TextPreprocessor.split_by_sentences(norm)
            per_text_sentences.append(sents)
            for s in sents:
                if s not in self._sent_cache_dict and s not in seen:
                    seen.add(s)
                    all_new.append(s)

        if all_new:
            self._prime_sentence_models(all_new)
            self._prime_omograph_logits(all_new)

        outputs = []
        for sents in per_text_sentences:
            outputs.append("".join(self._process_sentence_fast(s) for s in sents))
        self._omo_logits_by_sentence = {}
        self._sent_model_preds = {}
        return outputs

    # ------------------------------------------------------------------ #
    # process_all_internal — batch per-sentence ONNX across sentences    #
    # ------------------------------------------------------------------ #
    def process_all_internal(self, text):
        if not self._batch_sentences:
            return super().process_all_internal(text)

        text = re.sub(self.normalize, "", text)
        sentences = TextPreprocessor.split_by_sentences(text)
        if not sentences:
            return ""

        # Pre-compute the two whole-sentence token-classifier predictions
        # (stress-usage, e-homograph) for EVERY sentence in one padded batch
        # each — exactly the calls stock makes per sentence, bit-identical under
        # the attention mask.  Sentences already in the per-file cache skip both
        # the prime and the loop, mirroring stock's @lru_cache(4096).
        to_predict = list(
            dict.fromkeys(s for s in sentences if s not in self._sent_cache_dict)
        )
        if to_predict:
            self._prime_sentence_models(to_predict)
            # Then pre-compute the omograph classifier logits for every
            # homograph hypothesis across all sentences in ONE padded batch.
            self._prime_omograph_logits(to_predict)

        outputs = [self._process_sentence_fast(s) for s in sentences]
        # Free per-file scratch so a long worker run does not accumulate it.
        self._omo_logits_by_sentence = {}
        self._sent_model_preds = {}
        return "".join(outputs)

    def _prime_sentence_models(self, sentences: List[str]) -> None:
        """Run the stress-usage and e-homograph ONNX models once, batched over
        all ``sentences``, and stash per-sentence results for the fast loop.

        Word-less sentences are skipped entirely: stock's
        ``_process_sentence_cached`` returns before touching any model when
        ``split_by_words`` yields no words, so running the models on them (and
        on punctuation-only strings) would both waste work and crash the
        token-classifier aggregation.  Such sentences are handled by the
        early-return branch in ``_process_sentence_fast``."""
        sentences = [
            s for s in sentences if len(TextPreprocessor.split_by_words(s)[0]) > 0
        ]

        preds = self._sent_model_preds

        # --- stress usage (stock runs it unconditionally per word-ful sentence)
        if not self.tiny_mode:
            stress_batch = self._predict_token_classifier_batch(
                self.stress_usage_predictor, sentences
            )
        else:
            stress_batch = [None] * len(sentences)

        # --- e-homograph (stock runs it only when 'е' is in the sentence) ---
        # _process_yo lowercases the sentence first, so do the same here.
        yo_sentences = [s for s in sentences if "е" in s.lower()]
        yo_inputs = [s.lower() for s in yo_sentences]
        yo_batch = (
            self._predict_token_classifier_batch(self.yo_homograph_model, yo_inputs)
            if yo_inputs
            else []
        )
        yo_map = dict(zip(yo_sentences, yo_batch))

        for i, s in enumerate(sentences):
            preds[s] = {
                "stress": stress_batch[i],
                "yo": yo_map.get(s),  # None when 'е' absent (stock skips it)
            }

    # ------------------------------------------------------------------ #
    # cross-sentence omograph batching (the dominant cost)               #
    # ------------------------------------------------------------------ #
    def _prime_omograph_logits(self, sentences: List[str]) -> None:
        """Run the omograph classifier ONCE for every homograph hypothesis
        across all ``sentences``, and stash per-sentence raw logits.

        The omograph model's raw logits are batch-composition-invariant
        (attention-masked; verified 0.0 diff), so collecting every
        ``(preprocessed_text, hypothesis)`` pair into one padded ONNX call and
        then replaying the EXACT per-sentence ``classify`` selection logic on
        each sentence's slice of those logits is byte-identical to stock — the
        even/odd branch, ``group_words`` grouping and per-row softmax all run
        unchanged per sentence; only the ``session.run`` is shared."""
        om = self.omograph_model
        per_sentence = {}  # sentence -> (preprocessed_texts, hypotheses, num_hyp)
        all_texts: List[str] = []
        all_hyps: List[str] = []

        for sentence in sentences:
            # ``per_sentence`` dedupes (one slice per unique sentence) but the
            # batch and the replay walk are per-occurrence; skip duplicates so the
            # rows we push stay exactly aligned with the offsets we replay below.
            if sentence in per_sentence:
                continue
            words = self._yo_words_for_sentence(sentence)
            if words is None:
                continue
            spec = self._collect_omograph_inputs(list(words))
            if spec is None:
                continue
            texts_batch, hypotheses_batch, num_hypotheses = spec
            # classify() applies this exact regex sub before tokenizing.
            pre = [re.sub(r"\s+(?=(?:[,.?!:;…]))", r"", t) for t in texts_batch]
            per_sentence[sentence] = (pre, hypotheses_batch, num_hypotheses)
            all_texts.extend(pre)
            all_hyps.extend(hypotheses_batch)

        logits_by_sentence = {}
        if all_texts:
            inputs = om.tokenizer(
                all_texts,
                all_hyps,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
            logits = om.session.run(None, inputs)[0]  # (sum_rows, 2)
            offset = 0
            for sentence, (pre, hyps, num_h) in per_sentence.items():
                n = len(pre)
                logits_by_sentence[sentence] = (logits[offset : offset + n], hyps, num_h)
                offset += n

        self._omo_logits_by_sentence = logits_by_sentence

    def _yo_words_for_sentence(self, sentence: str):
        """Return the post-yo word list for ``sentence`` (or None if no words).

        This recomputes exactly what the main loop will compute (yo processing
        is deterministic given the cached e-homograph prediction), so the
        omograph inputs we build here match those _process_omographs would build
        in the loop.  ``split_by_words`` and ``_process_yo_fast`` are pure given
        the sentence + cached prediction, so calling them twice is safe."""
        words, _remaining = TextPreprocessor.split_by_words(sentence)
        if len(words) == 0:
            return None
        preds = self._sent_model_preds.get(sentence, {})
        return self._process_yo_fast(words, sentence, preds.get("yo"))

    def _collect_omograph_inputs(self, splitted_text):
        """Input-building half of ``RUAccent._process_omographs``: find homograph
        words and render each variant into the sentence with ``<w>..</w>``
        markers.  Returns ``(texts_batch, hypotheses_batch, num_hypotheses)`` or
        None when the sentence has no homographs (stock makes no ONNX call)."""
        founded_omographs = []
        hypotheses = []
        for i, word in enumerate(splitted_text):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append({"variants": variants, "position": i})
                hypotheses.append(variants)
        if not founded_omographs:
            return None

        texts_batch = []
        hypotheses_batch = [val for sub in hypotheses for val in sub]
        num_hypotheses = [len(i) for i in hypotheses]
        for o in founded_omographs:
            position = o["position"]
            t_back = splitted_text[position]
            splitted_text[position] = " <w>" + splitted_text[position] + "</w> "
            for _ in range(len(o["variants"])):
                texts_batch.append(
                    self.delete_spaces_before_punc(" ".join(splitted_text.copy()))
                )
            splitted_text[position] = t_back
        return texts_batch, hypotheses_batch, num_hypotheses

    @staticmethod
    def _predict_token_classifier_batch(model, sentences: List[str]):
        """``predict_stress_usage`` / ``predict_yo_homographs`` for each
        sentence, but with ONE padded ONNX run for the whole list.

        Each row is sliced to its own unpadded length BEFORE any softmax /
        reduction, so the per-sentence post-processing is byte-identical to the
        stock batch-1 path; ONNX attention masking makes the kept logits
        bit-identical to the per-call logits (verified 0.0 diff)."""
        if not sentences:
            return []
        enc = model.tokenizer(
            sentences,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="np",
            padding=True,
        )
        offset_mapping = enc.pop("offset_mapping")
        special_tokens_mask = enc.pop("special_tokens_mask")
        attention_mask = enc["attention_mask"]
        inputs = {k: v.astype(np.int64) for k, v in enc.items()}
        logits = model.session.run(None, inputs)[0]

        results = []
        for i, sentence in enumerate(sentences):
            row_len = int(attention_mask[i].sum())  # real (unpadded) tokens
            row_logits = logits[i, :row_len]
            maxes = np.max(row_logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(row_logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
            input_ids = inputs["input_ids"][i, :row_len]
            pre_entities = model.collect_pre_entities(
                sentence,
                input_ids,
                scores,
                offset_mapping[i, :row_len],
                special_tokens_mask[i, :row_len],
            )
            results.append(model.aggregate_words(pre_entities, "AVERAGE"))
        return results

    # ------------------------------------------------------------------ #
    # per-sentence fast path: identical logic, predictions pre-fetched   #
    # ------------------------------------------------------------------ #
    def _process_sentence_fast(self, sentence: str) -> str:
        cached = self._sent_cache_dict.get(sentence)
        if cached is not None:
            return cached

        words, remaining_text = TextPreprocessor.split_by_words(sentence)
        if len(words) == 0:
            result = "".join(remaining_text)
            self._cache_sentence(sentence, result)
            return result

        preds = self._sent_model_preds.get(sentence, {})
        if self.tiny_mode:
            stress_usages = ["STRESS"] * len(words)
        else:
            stress_usages = self.extract_entities(preds["stress"])

        processed_words = self._process_yo_fast(words, sentence, preds.get("yo"))
        processed_words = self._process_omographs_fast(sentence, processed_words)
        processed_words = self._process_accent(processed_words, stress_usages)

        processed_sentence = "".join(
            [l + r for l, r in zip(remaining_text, processed_words)] + [remaining_text[-1]]
        )
        processed_sentence = self.delete_spaces_before_punc(processed_sentence)
        self._cache_sentence(sentence, processed_sentence)
        return processed_sentence

    def _cache_sentence(self, sentence: str, result: str) -> None:
        cache = self._sent_cache_dict
        if len(cache) >= self._SENT_CACHE_MAX:
            cache.clear()
        cache[sentence] = result

    # ------------------------------------------------------------------ #
    # omograph: assignment half of _process_omographs, using cached logits
    # ------------------------------------------------------------------ #
    def _process_omographs_fast(self, sentence, splitted_text):
        """``RUAccent._process_omographs`` assignment half, but the ONNX call is
        served from the cross-sentence batch (``_prime_omograph_logits``).

        Falls back to the stock per-sentence call if this sentence has no cached
        logits (e.g. when called for a sentence not primed in this batch)."""
        entry = self._omo_logits_by_sentence.get(sentence)
        founded_positions = [
            i for i, w in enumerate(splitted_text) if self.omographs.get(w)
        ]
        if not founded_positions:
            return splitted_text
        if entry is None:
            return self._process_omographs(splitted_text)

        logits, hypotheses_batch, num_hypotheses = entry
        cls_batch = self._classify_with_logits(logits, hypotheses_batch, num_hypotheses)
        for cls_index, position in enumerate(founded_positions):
            splitted_text[position] = cls_batch[cls_index]
        return splitted_text

    def _classify_with_logits(self, logits, hypotheses, num_hypotheses):
        """Byte-for-byte replica of ``OmographModel.classify`` selection, but the
        precomputed raw ``logits`` (rows aligned to ``hypotheses``) replace the
        per-call ``session.run``.  Both even and odd branches are reproduced
        exactly, including the global ``softmax`` of the even path and the
        per-row softmax + ``group_words`` of the odd path."""
        om = self.omograph_model
        hypotheses_probs = []
        if not all(i % 2 == 0 for i in num_hypotheses):
            # odd path: group_words over a per-single-row softmax
            outs = []
            grouped_h = om.group_words(hypotheses)
            # rows are aligned 1:1 with `hypotheses`; walk groups in order
            row = 0
            for h in grouped_h:
                probs = []
                for _hp in h:
                    single = om.softmax(logits[row : row + 1])  # softmax of (1,2)
                    probs.append(float(single[0][1]))
                    row += 1
                outs.append(h[probs.index(max(probs))])
            return outs
        else:
            outputs = om.softmax(logits)  # GLOBAL softmax over the whole array
            hyps = [
                (hypotheses[i], hypotheses[i + 1]) for i in range(0, len(hypotheses), 2)
            ]
            for i in range(len(logits)):  # len(texts) == number of rows
                hypotheses_probs.append(float(outputs[i][1]))
            hypotheses_probs = [
                (hypotheses_probs[i], hypotheses_probs[i + 1])
                for i in range(0, len(hypotheses_probs), 2)
            ]
            outs = []
            for pair1, pair2 in zip(hyps, hypotheses_probs):
                outs.append(pair1[pair2.index(max(pair2))])
            return outs

    def _process_yo_fast(self, words, sentence, yo_entities):
        """``RUAccent._process_yo`` with the e-homograph ONNX prediction passed
        in (already computed in the batch) instead of run inline."""
        yo_predictions = (
            self.extract_entities(yo_entities) if yo_entities is not None else None
        )
        for i, word in enumerate(words):
            lower_word = word.lower()
            words[i] = fix_capital(word, self.yo_words.get(lower_word, word))
            if yo_predictions and yo_predictions[i] == "YO":
                words[i] = fix_capital(word, self.yo_homographs.get(lower_word, word))
        return words
