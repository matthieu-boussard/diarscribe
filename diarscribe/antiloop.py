"""Loop / hallucination guards for VibeVoice-ASR.

Three layers:
  1. ``LoopStoppingCriteria`` aborts ``generate`` as soon as the output becomes periodic
     (exact token loop, or the same text repeated with only the timestamps changing).
  2. ``diagnose`` inspects a finished transcription (compression ratio, words/second,
     repeated n-grams, incoherent timestamps, token budget exhausted).
  3. ``collapse_repetitions`` is the last-resort cleanup when every retry still loops.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zlib
from collections.abc import Sequence
from typing import Any

import torch
from transformers import StoppingCriteria

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_CONTENT_RE = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"?')


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
    """Lower-cased letter-only words: digits (timestamps) and punctuation (JSON) vanish."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _WORD_RE.findall(text)


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def extract_contents(raw: str) -> list[str]:
    """Best-effort extraction of "Content" fields from (possibly truncated) VibeVoice JSON."""
    out = []
    for m in _CONTENT_RE.finditer(raw):
        try:
            out.append(json.loads(f'"{m.group(1)}"'))
        except json.JSONDecodeError:
            out.append(m.group(1))
    return out


class LoopStoppingCriteria(StoppingCriteria):
    """Stops generation (and records why) when the output starts looping."""

    def __init__(self, prompt_len: int, tokenizer: Any, tail_tokens: int = 400, text_check_every: int = 8):
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        self.tail_tokens = tail_tokens
        self.text_check_every = text_check_every
        self.reason: str | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        gen = input_ids[0, self.prompt_len :]
        n = gen.shape[0]
        if self.reason is None and n >= 16:
            tail = gen[-self.tail_tokens :].tolist()
            hit = find_periodic_suffix(tail)
            if hit:
                self.reason = f"token loop (period={hit[0]}, x{hit[1]})"
            elif n % self.text_check_every == 0:
                # Catches "same sentence, new timestamps" loops that are not token-exact.
                words = normalize_words(self.tokenizer.decode(tail, skip_special_tokens=True))
                hit = find_periodic_suffix(words, max_period=40, min_repeats=3, min_span=30, short_period=3, short_min_repeats=6)
                if hit:
                    self.reason = f"text loop (period={hit[0]} words, x{hit[1]})"
        return torch.full((input_ids.shape[0],), self.reason is not None, dtype=torch.bool, device=input_ids.device)


def diagnose(
    text: str,
    duration: float,
    segments: list[dict] | None = None,
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

    if segments:
        prev_start = -1.0
        for seg in segments:
            start, end = _num(seg.get("Start")), _num(seg.get("End"))
            if end is not None and end > duration + 3.0:
                issues.append("timestamps beyond audio")
                break
            if start is not None:
                if start < prev_start - 1.0:
                    issues.append("timestamps going backwards")
                    break
                prev_start = start
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


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
