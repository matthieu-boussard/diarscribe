"""Loop / hallucination guards for autoregressive ASR decoding.

Three layers:
  1. ``LoopStoppingCriteria`` stops each row of a batched ``generate`` as soon as its output becomes
     periodic (exact token loop, or the same words repeated with different tokenization).
  2. ``diagnose`` inspects a finished transcription (compression ratio, words/second,
     repeated n-grams, token budget exhausted).
  3. ``collapse_repetitions`` is the last-resort cleanup when every retry still loops.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections.abc import Sequence
from typing import Any

import torch
from transformers import StoppingCriteria

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def find_periodic_suffix(
    seq: Sequence,
    max_period: int = 64,
    min_repeats: int = 3,
    min_span: int = 30,
    short_period: int = 4,
    short_min_repeats: int = 10,
) -> tuple[int, int] | None:
    """Return (period, repeats) if ``seq`` ends with a unit repeated back-to-back, else None.

    Short units (<= ``short_period``) must repeat ``short_min_repeats`` times; longer ones
    ``min_repeats`` times and cover at least ``min_span`` elements.
    """
    n = len(seq)
    for p in range(1, min(max_period, n // 2) + 1):
        unit = list(seq[n - p :])
        repeats = 1
        i = n - 2 * p
        while i >= 0 and list(seq[i : i + p]) == unit:
            repeats += 1
            i -= p
        if p <= short_period:
            if repeats >= short_min_repeats:
                return p, repeats
        elif repeats >= min_repeats and p * repeats >= min_span:
            return p, repeats
    return None


def find_repeated_run(
    words: Sequence[str], max_n: int = 30, min_repeats_long: int = 3, min_repeats_short: int = 6
) -> tuple[int, int, int] | None:
    """Find any n-gram repeated consecutively anywhere in ``words``: (start, n, repeats)."""
    L = len(words)
    for i in range(L):
        for n in range(1, min(max_n, (L - i) // 2) + 1):
            repeats = 1
            while words[i : i + n] == words[i + repeats * n : i + (repeats + 1) * n]:
                repeats += 1
            needed = min_repeats_short if n <= 2 else min_repeats_long
            if repeats >= needed:
                return i, n, repeats
    return None


def normalize_words(text: str) -> list[str]:
    """Lower-cased letter-only words: digits and punctuation vanish."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _WORD_RE.findall(text)


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


class LoopStoppingCriteria(StoppingCriteria):
    """Per-row loop detector for batched ``generate``; ``reasons[row]`` records why a row was stopped."""

    def __init__(self, prompt_len: int, tokenizer: Any, ignore_ids: set[int] = frozenset(),
                 tail_tokens: int = 400, text_check_every: int = 8):
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        self.ignore_ids = ignore_ids
        self.tail_tokens = tail_tokens
        self.text_check_every = text_check_every
        self.reasons: list[str | None] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        if not self.reasons:
            self.reasons = [None] * input_ids.shape[0]
        gen = input_ids[:, self.prompt_len :]
        n = gen.shape[1]
        if n >= 16:
            tails = gen[:, -self.tail_tokens :].tolist()
            for row, tail in enumerate(tails):
                # Finished rows are padded with EOS/pad: that repetition is not a loop.
                if self.reasons[row] is None and tail[-1] not in self.ignore_ids:
                    self.reasons[row] = self._check(tail, n)
        return torch.tensor([r is not None for r in self.reasons], dtype=torch.bool, device=input_ids.device)

    def _check(self, tail: list[int], n: int) -> str | None:
        hit = find_periodic_suffix(tail)
        if hit:
            return f"token loop (period={hit[0]}, x{hit[1]})"
        if n % self.text_check_every == 0:
            # Same words with a different tokenization each time are not token-exact loops.
            words = normalize_words(self.tokenizer.decode(tail, skip_special_tokens=True))
            hit = find_periodic_suffix(words, max_period=40, min_repeats=3, min_span=30, short_period=3, short_min_repeats=6)
            if hit:
                return f"text loop (period={hit[0]} words, x{hit[1]})"
        return None


def diagnose(
    text: str,
    duration: float,
    hit_token_limit: bool = False,
    stop_reason: str | None = None,
    max_wps: float = 6.0,
    max_compression: float = 2.4,
) -> list[str]:
    """Return the list of problems found in a transcription (empty list = looks healthy)."""
    issues = []
    if stop_reason:
        issues.append(f"stopped: {stop_reason}")
    if hit_token_limit:
        issues.append("max_new_tokens reached")

    words = normalize_words(text)
    if len(text) > 60 and compression_ratio(text) > max_compression:
        issues.append(f"compression ratio {compression_ratio(text):.2f}")
    # Slack of a few words: diarization boundaries are tight, short bursts of fast speech are normal.
    if len(words) > max_wps * duration + 8:
        issues.append(f"{len(words) / max(duration, 0.1):.1f} words/s")
    run = find_repeated_run(words)
    if run:
        issues.append(f"repeated {run[1]}-gram x{run[2]}")

    return issues


def collapse_repetitions(text: str, max_n: int = 30) -> str:
    """Keep a single copy of any n-gram repeated back-to-back (2 copies for 1-2 word units)."""
    tokens = text.split()
    keys = [" ".join(normalize_words(t)) or t for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for n in range(1, min(max_n, (len(tokens) - i) // 2) + 1):  # shortest period first
            repeats = 1
            while keys[i : i + n] == keys[i + repeats * n : i + (repeats + 1) * n]:
                repeats += 1
            keep = 2 if n <= 2 else 1
            if repeats > keep and (n >= 3 or repeats >= 3):
                out.extend(tokens[i : i + n * keep])
                i += n * repeats
                break
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

