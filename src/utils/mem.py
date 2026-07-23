"""Per-worker heap trimming for audio-decode DataLoaders.

torchcodec/ffmpeg decode churn leaves glibc holding freed memory: a DataLoader
worker's RSS ratchets up to a multi-GB high-water mark and, with persistent
loaders, never drops — which OOM-kills workers on a RAM-constrained box.
``periodic_malloc_trim()`` calls ``malloc_trim(0)`` once per N decoded items
(in whichever process decodes — each loader worker keeps its own counter; also
covers inline single-GPU paths), handing that memory back to the OS. Measured
on the transcription group loader: worker heap 1451 -> 221 MB, RSS 2218 -> 988
MB. It only affects memory (never outputs) and is a no-op on non-glibc libc.

The interval comes from ``BALALAIKA_MALLOC_TRIM_EVERY`` (default 128, 0 disables),
emitted by ``runtime_env.py`` from the ``runtime.malloc_trim_every`` config key.
"""
import ctypes
import os

_TRIM_EVERY = int(os.environ.get("BALALAIKA_MALLOC_TRIM_EVERY", "128") or 0)


def _load_malloc_trim():
    if _TRIM_EVERY <= 0:
        return None
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        fn = libc.malloc_trim
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        return fn
    except (OSError, AttributeError):
        return None


_MALLOC_TRIM = _load_malloc_trim()
_decode_counter = 0


def periodic_malloc_trim() -> None:
    """Return retained heap to the OS every ``BALALAIKA_MALLOC_TRIM_EVERY`` items."""
    global _decode_counter
    if _MALLOC_TRIM is None:
        return
    _decode_counter += 1
    if _decode_counter % _TRIM_EVERY == 0:
        _MALLOC_TRIM(0)
