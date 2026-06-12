"""Drop-in fast TryIParu G2P: batched OOV decode + cached dictionary.

Same weights, tokenizer and post-processing as ``tryiparu.G2PModel``; verified
token-identical on the ``benchmarking/micro/bench_g2p.py`` fixture set.  The
differences are purely mechanical:

- unique OOV words of a text are greedy-decoded as ONE padded batch instead of
  word-by-word (stock pays ~80-125 ms per word, every step a GPU sync);
- no ``torch.compile(mode="max-autotune")`` call — in stock it wraps only
  ``forward`` while inference goes through ``.encode``/``.decode`` (never
  compiled), so it costs ~2.8 s per worker and accelerates nothing;
- the 398k-word dictionary CSV is parsed once per node into a pickle cache
  (``cache/g2p_dict.pkl``), not re-parsed by pandas in every worker;
- optionally, OOV decodes persist across runs/workers in a pickle keyed by the
  model-weights fingerprint (``oov_cache_path``), so repeated unknown words
  (names, brands) are never decoded twice.

One deliberate divergence: a word encoding to exactly ``MAX_LEN - 2`` BPE ids
crashes stock (its zero-length pad segment builds an empty *float* tensor and
``torch.cat`` promotes the whole encoder input, blowing up ``nn.Embedding``);
FastG2P decodes it normally.  Words longer than that raise the identical
stock ``ValueError``.  Pinned in tests/test_phonemizer_fast_g2p.py.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import pickle
import string
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
import tryiparu
from loguru import logger
from tokenizers import Tokenizer
from tryiparu.configs.config import config_g2p
from tryiparu.rules import process_word
from tryiparu.transformer import TransformerBlock
from tryiparu.tryiparu import G2PModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DICT_CACHE = REPO_ROOT / "cache" / "g2p_dict.pkl"

_TRYIPARU_DIR = Path(tryiparu.__file__).parent


def _fingerprint(path: Path) -> tuple:
    """(size, content hash) — mtime is unreliable (cp -p / rsync -a / wheels
    built with SOURCE_DATE_EPOCH preserve it across genuinely different
    files); hashing the ~6-15 MB inputs costs ~30 ms once per worker."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return (os.stat(path).st_size, h.hexdigest())


def _atomic_pickle_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _try_read_pickle_cache(cache_path: Path, key: tuple) -> Optional[dict]:
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        data = payload.get("data")
        if payload.get("key") == key and isinstance(data, dict):
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:  # corrupted/stale cache — rebuild
        logger.warning(f"g2p cache {cache_path} unreadable ({exc}); ignoring it")
    return None


def _load_dictionary(csv_path: Path, cache_path: Path) -> Dict[str, str]:
    """words->phonemes dict from the package CSV, via a node-local pickle."""
    key = _fingerprint(csv_path)
    data = _try_read_pickle_cache(cache_path, key)
    if data is not None:
        return data

    # Build under a lock so N simultaneously-spawning workers parse the CSV
    # once; the others block here and then read the fresh pickle.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _try_read_pickle_cache(cache_path, key)
        if data is not None:
            return data

        import csv as csv_mod

        data = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    # last value wins, matching DataFrame.set_index().to_dict()
                    data[row[0]] = row[1]
        try:
            _atomic_pickle_dump({"key": key, "data": data}, cache_path)
        except OSError as exc:
            logger.warning(f"could not write g2p dict cache: {exc}")
        return data


class FastG2P(G2PModel):
    """tryiparu.G2PModel with batched greedy decode and cached dict load."""

    def __init__(
        self,
        device: str,
        batch_size: int = 64,
        oov_cache_path: Optional[str] = None,
        oov_flush_every: int = 256,
        dict_cache_path: Path = DEFAULT_DICT_CACHE,
    ) -> None:
        # Deliberately does NOT call super().__init__ — stock init pays for a
        # no-op torch.compile and a pandas CSV parse; everything it sets up is
        # recreated here.
        pkg = _TRYIPARU_DIR
        self.tokenizer_file = str(pkg / "configs" / "bpe.json")
        self.model_weights = str(pkg / "weights" / "model.pt")
        self.device = device

        self.tokenizer = Tokenizer.from_file(self.tokenizer_file)
        self.model = TransformerBlock(tokenizer=self.tokenizer, config=config_g2p)
        self.model.to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_weights, map_location=self.device, weights_only=False)
        )
        self.model.eval()
        self.max_length = config_g2p.get("MAX_LEN", 64)

        self.bos_token_id = self.tokenizer.encode("<bos>").ids[0]
        self.eos_token_id = self.tokenizer.encode("<eos>").ids[0]
        self.pad_token_id = self.tokenizer.encode("<pad>").ids[0]

        self.batch_size = max(1, int(batch_size))
        self._rules_cache: Dict[str, List[str]] = {}
        self._tril = torch.tril(
            torch.ones((self.max_length, self.max_length), dtype=torch.int64, device=self.device)
        )

        self.data_dict: Dict[str, object] = _load_dictionary(
            pkg / "data" / "cleaned_dataset.csv", Path(dict_cache_path)
        )

        self._weights_key = _fingerprint(Path(self.model_weights))
        if oov_cache_path:
            oov_path = Path(oov_cache_path)
            if not oov_path.is_absolute():
                # Anchor like the dict cache (and resolve_model_path in the
                # denoising stage) — never depend on the worker CWD.
                oov_path = REPO_ROOT / oov_path
            self._oov_cache_path = oov_path
        else:
            self._oov_cache_path = None
        self._oov_pending: Dict[str, List[str]] = {}
        self._oov_flush_every = max(1, int(oov_flush_every))
        if self._oov_cache_path is not None:
            self.data_dict.update(self._read_oov_cache())
            import atexit

            atexit.register(self.flush_oov_cache)

    # ------------------------------------------------------------------ #
    # persistent OOV cache                                               #
    # ------------------------------------------------------------------ #
    def _read_oov_cache(self) -> Dict[str, List[str]]:
        data = _try_read_pickle_cache(self._oov_cache_path, self._weights_key)
        return data if data is not None else {}

    def flush_oov_cache(self) -> None:
        """Merge pending OOV decodes into the shared cache (lock + replace)."""
        if self._oov_cache_path is None or not self._oov_pending:
            return
        import fcntl

        try:
            lock_path = self._oov_cache_path.with_suffix(".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    merged = self._read_oov_cache()
                    merged.update(self._oov_pending)
                    _atomic_pickle_dump(
                        {"key": self._weights_key, "data": merged}, self._oov_cache_path
                    )
                    self._oov_pending.clear()
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception as exc:  # cache is best-effort; never kill a worker
            logger.warning(f"g2p OOV cache flush failed: {exc}")

    # ------------------------------------------------------------------ #
    # batched greedy decode                                              #
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def decode_batch(self, words: List[str]) -> Dict[str, List[str]]:
        """Greedy-decode ``words`` in padded batches; fills ``data_dict``.

        Token-for-token the same computation as stock ``greedy_decode`` (same
        64-padded encoder input, same masks, same per-step argmax), just with
        a batch dimension.  A too-long word raises the stock ValueError; words
        decoded before it in the same call are still cached, like stock.
        """
        todo = [w for w in dict.fromkeys(words) if w not in self.data_dict]
        out: Dict[str, List[str]] = {}
        oversize: Optional[ValueError] = None

        for start in range(0, len(todo), self.batch_size):
            chunk, enc_rows = [], []
            for word in todo[start : start + self.batch_size]:
                src_tokens = self.tokenizer.encode(word).ids
                seq_len = len(src_tokens) + 2
                if seq_len > self.max_length:
                    oversize = ValueError(
                        f"Input sequence too long. Max length: {self.max_length}, "
                        f"got {seq_len}"
                    )
                    break
                enc_rows.append(
                    [self.bos_token_id]
                    + src_tokens
                    + [self.eos_token_id]
                    + [self.pad_token_id] * (self.max_length - seq_len)
                )
                chunk.append(word)

            if chunk:
                rows = self._greedy_decode_rows(enc_rows)
                for word, ids in zip(chunk, rows):
                    phonemes = self._process_decoded_output(self.tokenizer.decode(ids))
                    self.data_dict[word] = phonemes
                    if self._oov_cache_path is not None:
                        self._oov_pending[word] = phonemes
                    out[word] = phonemes

            if oversize is not None:
                break

        if (
            self._oov_cache_path is not None
            and len(self._oov_pending) >= self._oov_flush_every
        ):
            self.flush_oov_cache()
            # Each inline flush re-reads + rewrites the whole pickle, so back
            # off geometrically during heavy-OOV cold starts; atexit still
            # flushes whatever remains.
            self._oov_flush_every = min(self._oov_flush_every * 2, 4096)
        if oversize is not None:
            raise oversize
        return out

    def _greedy_decode_rows(self, enc_rows: List[List[int]]) -> List[List[int]]:
        """Lockstep greedy decode of pre-padded encoder rows; returns id rows
        truncated at (and including) the first <eos>.

        Runs with TF32 matmuls OFF (saved/restored around the decode): with
        TF32 — the pipeline-wide default — even the STOCK model flips argmax
        ties on knife-edge words depending on kernel choice, so TF32 outputs
        are not reproducible run-to-run.  These d=128 GEMMs are launch-bound,
        fp32 costs nothing (measured 3.3 vs 3.4 ms/word), and the result is
        verified token-identical to the stock fp32 reference on the tested
        configurations (CPU and RTX 4060 Ti, batch sizes 1/3/5/64, 1200 OOV
        words + 30 texts) — see report.md §4.9.
        """
        on_cuda = self.device.startswith("cuda")
        if on_cuda:
            ambient_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self._greedy_decode_rows_inner(enc_rows, on_cuda)
        finally:
            if on_cuda:
                torch.backends.cuda.matmul.allow_tf32 = ambient_tf32

    def _greedy_decode_rows_inner(
        self, enc_rows: List[List[int]], on_cuda: bool
    ) -> List[List[int]]:
        encoder_input = torch.tensor(enc_rows, dtype=torch.int64, device=self.device)
        encoder_mask = (
            (encoder_input != self.pad_token_id).unsqueeze(1).unsqueeze(1).int()
        )
        encoder_output = self.model.encode(encoder_input, encoder_mask)

        n_rows = encoder_input.size(0)
        decoder_input = torch.full(
            (n_rows, 1), self.bos_token_id, dtype=torch.int64, device=self.device
        )
        finished = torch.zeros(n_rows, dtype=torch.bool, device=self.device)

        # On CUDA, `bool(finished.all())` is a host sync that also waits out
        # foreign kernels on GPUs shared with training, so only poll every 4th
        # step; the ≤3 extra lockstep steps produce post-<eos> tokens that the
        # truncation below discards, leaving outputs identical.
        sync_every = 4 if on_cuda else 1
        for step in range(self.max_length - 1):
            t = decoder_input.size(1)
            tgt_mask = self._tril[:t, :t].unsqueeze(0)
            decoder_output = self.model.decode(
                encoder_output, encoder_mask, decoder_input, tgt_mask
            )
            logits = self.model.fc_out(decoder_output[:, -1])
            next_token = torch.argmax(logits, dim=1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            finished |= next_token == self.eos_token_id
            if step % sync_every == sync_every - 1 and bool(finished.all()):
                break

        results = []
        for row in decoder_input.tolist():
            if self.eos_token_id in row:
                row = row[: row.index(self.eos_token_id) + 1]
            results.append(row)
        return results

    # ------------------------------------------------------------------ #
    # text entry point — stock logic, OOV pre-decoded as one batch       #
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(self, text: str) -> List[str]:
        tokens = self._split_text(text.lower())

        oov = [t for t in dict.fromkeys(tokens) if t.isalpha() and t not in self.data_dict]
        if oov:
            self.decode_batch(oov)

        # From here on this is the stock tryiparu __call__ loop verbatim
        # (every alpha token now hits data_dict).
        output_tokens = []
        for i, token in enumerate(tokens):
            if not token.isalpha():
                output_tokens.append(token)
            else:
                phoneme_tokens = self.data_dict[token]
                phoneme_tokens = (
                    [phoneme_tokens]
                    if not isinstance(phoneme_tokens, list)
                    else phoneme_tokens
                )
                output_tokens.extend(phoneme_tokens)

                if i < len(tokens) - 1 and tokens[i + 1].isalpha():
                    output_tokens.append(" ")

        processed_tokens = []
        for i, token in enumerate(output_tokens):
            if token == " ":
                if (
                    (i > 0 and output_tokens[i - 1] in string.punctuation)
                    or (i < len(output_tokens) - 1 and output_tokens[i + 1] in string.punctuation)
                    or (i > 0 and output_tokens[i - 1] == " ")
                ):
                    continue
            processed_tokens.append(token)

        if processed_tokens and processed_tokens[-1] == " ":
            processed_tokens.pop()

        return self._process_tokens(processed_tokens)

    def _process_tokens(self, phonemes: List[str]) -> List[str]:
        """``tryiparu.rules.process_text`` with ``process_word`` memoized —
        the same dictionary words recur across every file, and their rule
        split is a pure function of the string."""
        result: List[str] = []
        cache = self._rules_cache
        for phoneme in phonemes:
            if phoneme.strip() == "" or all(ch in string.punctuation for ch in phoneme):
                result.append(phoneme)
            else:
                cached = cache.get(phoneme)
                if cached is None:
                    if len(cache) >= 200_000:
                        # Crude RAM bound (~100 MB worst case per worker); the
                        # memo only saves time, never affects outputs.
                        cache.clear()
                    cached = cache[phoneme] = process_word(phoneme)
                result.extend(cached)
        return result
