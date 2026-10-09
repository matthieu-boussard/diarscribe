"""Word timestamps by CTC forced alignment (wav2vec2), as in WhisperX.

The ASR model gives text without timestamps; aligning its words on the audio lets us transcribe long,
multi-speaker windows (more context, better recognition) and still attribute every word to a speaker.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

log = logging.getLogger(__name__)

# Default CTC aligners per language (WhisperX's choices, Apache-2.0 / MIT).
ALIGN_MODELS = {
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
}

_CHAR_MAP = {"’": "'", "‘": "'", "`": "'", "‐": "-", "–": "-"}


@dataclass
class Word:
    start: float
    end: float
    text: str  # as written by the ASR model (punctuation, casing kept)


class Aligner:
    def __init__(self, language: str = "fr", device: str = "cuda", model_id: str | None = None, sample_rate: int = 16_000):
        model_id = model_id or ALIGN_MODELS.get(language)
        if model_id is None:
            raise ValueError(f"pas d'aligneur par défaut pour la langue {language!r} : passez model_id")
        # Plain processor: some aligner repos ship an n-gram LM we do not need (and that needs pyctcdecode).
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.device, self.sample_rate = device, sample_rate
        vocab = self.processor.tokenizer.get_vocab()
        self.vocab = {k.lower(): v for k, v in vocab.items() if len(k) == 1}
        self.blank = vocab.get(self.processor.tokenizer.pad_token, 0)
        self.sep = vocab.get("|")

    @torch.inference_mode()
    def align(self, audio: np.ndarray, start: float, end: float, text: str) -> list[Word]:
        """Word timestamps (absolute seconds) for ``text`` spoken in ``audio[start:end]``."""
        words = _attach_punctuation(text.split())
        if not words:
            return []
        seg = audio[int(start * self.sample_rate) : int(end * self.sample_rate)]
        inputs = self.processor(seg, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        emission = torch.log_softmax(self.model(**inputs).logits[0].float(), dim=-1).cpu().numpy()
        frame_s = (len(seg) / self.sample_rate) / emission.shape[0]

        tokens, owner = [], []  # token ids, and the word index each token belongs to
        for w_idx, w in enumerate(words):
            ids = [self.vocab[c] for c in self._normalize(w) if c in self.vocab]
            if ids and tokens and self.sep is not None:
                tokens.append(self.sep)
                owner.append(-1)
            tokens += ids
            owner += [w_idx] * len(ids)

        spans = ctc_align(emission, tokens, self.blank) if tokens else None
        if spans is None:
            log.info("alignement impossible sur %.1f-%.1fs : répartition uniforme", start, end)
            return _uniform(words, start, end)

        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for (t0, t1), w_idx in zip(spans, owner):
            if w_idx >= 0:
                first.setdefault(w_idx, t0)
                last[w_idx] = t1
        timed = [
            (start + first[i] * frame_s, start + (last[i] + 1) * frame_s) if i in first else None
            for i in range(len(words))
        ]
        return [Word(a, b, w) for (a, b), w in zip(_fill_gaps(timed, start, end), words)]

    @staticmethod
    def _normalize(word: str) -> str:
        word = unicodedata.normalize("NFC", word.lower())
        return "".join(_CHAR_MAP.get(c, c) for c in word)


def ctc_align(emission: np.ndarray, tokens: list[int], blank: int) -> list[tuple[int, int]] | None:
    """Viterbi CTC forced alignment: (first_frame, last_frame) of each token, or None if impossible.

    ``emission`` is (frames, vocab) log-probabilities. At each frame the path either advances to the next
    token or stays on the current one, emitting blank or repeating that token (CTC allows both).
    """
    T, L = emission.shape[0], len(tokens)
    if L == 0 or T < L:
        return None
    tok = np.asarray(tokens)
    trellis = np.full((T + 1, L + 1), -np.inf)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = np.cumsum(emission[:, blank])
    for t in range(T):
        stay = trellis[t, 1:] + np.maximum(emission[t, blank], emission[t, tok])
        change = trellis[t, :-1] + emission[t, tok]
        trellis[t + 1, 1:] = np.maximum(stay, change)
    if not np.isfinite(trellis[T, L]):
        return None

    # Backtrack from the last frame. A token owns the frame where it is emitted, plus the following
    # "stay" frames only while it still beats blank: trailing silence must not stretch a word.
    frames: list[list[int]] = [[] for _ in range(L)]
    j = L
    for t in range(T, 0, -1):
        if j == 0:
            break
        stay = trellis[t - 1, j] + max(emission[t - 1, blank], emission[t - 1, tok[j - 1]])
        change = trellis[t - 1, j - 1] + emission[t - 1, tok[j - 1]]
        if change > stay:
            frames[j - 1].append(t - 1)
            j -= 1
        elif emission[t - 1, tok[j - 1]] > emission[t - 1, blank]:
            frames[j - 1].append(t - 1)
    if j != 0:
        return None
    return [(min(f), max(f)) for f in frames]


def _attach_punctuation(tokens: list[str]) -> list[str]:
    """Glue punctuation-only tokens (French "non ?", "– oui") to the previous word: they have nothing to align."""
    out: list[str] = []
    for tok in tokens:
        if out and not any(c.isalnum() for c in tok):
            out[-1] = f"{out[-1]} {tok}"
        else:
            out.append(tok)
    return out


def _fill_gaps(timed: list[tuple[float, float] | None], start: float, end: float) -> list[tuple[float, float]]:
    """Words with no alignable character (numbers, symbols) get the gap between their neighbours."""
    out: list[tuple[float, float]] = []
    for i, t in enumerate(timed):
        if t is not None:
            out.append(t)
            continue
        prev_end = out[-1][1] if out else start
        nxt = next((x for x in timed[i + 1 :] if x is not None), None)
        next_start = nxt[0] if nxt else end
        out.append((prev_end, max(prev_end, next_start)))
    return out


def _uniform(words: list[str], start: float, end: float) -> list[Word]:
    step = (end - start) / len(words)
    return [Word(start + i * step, start + (i + 1) * step, w) for i, w in enumerate(words)]
