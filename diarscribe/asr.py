"""VibeVoice-ASR transcription wrapped in an anti-loop retry ladder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoProcessor, StoppingCriteriaList, VibeVoiceAsrForConditionalGeneration

from .antiloop import LoopStoppingCriteria, collapse_repetitions, diagnose, extract_contents
from .audio import best_cut
from .chunking import fmt_ts as _ts

log = logging.getLogger(__name__)

MODEL_ID = "microsoft/VibeVoice-ASR-HF"
SAMPLE_RATE = 24_000

# Each rung is tried only if the previous one looped / looked hallucinated.
RETRY_LADDER: list[dict] = [
    {"do_sample": False},
    {"do_sample": False, "repetition_penalty": 1.1},
    {"do_sample": True, "temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.1},
]


@dataclass
class Piece:
    start: float
    end: float
    text: str
    flags: list[str] = field(default_factory=list)


@dataclass
class _Attempt:
    pieces: list[Piece]  # times relative to the chunk
    issues: list[str]

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.pieces)


class Transcriber:
    def __init__(
        self,
        device: str = "mps",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = MODEL_ID,
        context: str | None = None,
        tokens_per_s: float = 15.0,
        base_tokens: int = 80,
        max_wps: float = 6.0,
        max_split_depth: int = 2,
        min_split_s: float = 4.0,
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.device, self.dtype = device, dtype
        self.context = context
        self.tokens_per_s, self.base_tokens = tokens_per_s, base_tokens
        self.max_wps = max_wps
        self.max_split_depth, self.min_split_s = max_split_depth, min_split_s

    def transcribe_span(self, audio: np.ndarray, start: float, end: float, _depth: int = 0) -> list[Piece]:
        """Transcribe ``audio[start:end]`` (24 kHz mono); returned times are absolute."""
        seg = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        duration = len(seg) / SAMPLE_RATE
        if duration < 0.2:
            return []

        attempts: list[_Attempt] = []
        for rung, cfg in enumerate(RETRY_LADDER):
            att = self._generate(seg, duration, cfg)
            if not att.issues:
                return self._absolute(att.pieces, start, ["retry"] if rung else [])
            log.info("[%s-%s] essai %d suspect : %s", _ts(start), _ts(end), rung + 1, "; ".join(att.issues))
            attempts.append(att)

        # Every rung looped: split at the quietest point and try each half separately.
        if _depth < self.max_split_depth and duration >= self.min_split_s:
            cut = best_cut(audio, SAMPLE_RATE, start, end)
            log.info("[%s-%s] découpage à %s", _ts(start), _ts(end), _ts(cut))
            return self.transcribe_span(audio, start, cut, _depth + 1) + self.transcribe_span(audio, cut, end, _depth + 1)

        # Last resort: keep the least-bad attempt, strip the repetitions, cap its length.
        best = min(attempts, key=lambda a: (len(a.issues), len(a.text)))
        max_words = int(self.max_wps * max(duration, 1.0)) + 5
        for p in best.pieces:
            words = collapse_repetitions(p.text).split()
            p.text = " ".join(words[:max_words])
        log.warning("[%s-%s] boucle non résolue, texte nettoyé : %s", _ts(start), _ts(end), "; ".join(best.issues))
        return self._absolute(best.pieces, start, ["loop_trimmed", *best.issues])

    @torch.inference_mode()
    def _generate(self, seg: np.ndarray, duration: float, cfg: dict) -> _Attempt:
        inputs = self.processor.apply_transcription_request(audio=seg, prompt=self.context)
        inputs = inputs.to(self.device, self.dtype)
        prompt_len = inputs["input_ids"].shape[1]
        guard = LoopStoppingCriteria(prompt_len, self.processor.tokenizer)
        max_new = int(self.base_tokens + duration * self.tokens_per_s)

        out = self.model.generate(
            **inputs, max_new_tokens=max_new, stopping_criteria=StoppingCriteriaList([guard]), **cfg
        )
        gen = out[:, prompt_len:]
        hit_limit = gen.shape[1] >= max_new and guard.reason is None

        raw = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        try:
            parsed = self.processor.extract_speaker_dict(raw)  # raises on truncated JSON (loop guard cut)
        except ValueError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(d, dict) for d in parsed):
            segments = parsed
            pieces = [
                Piece(_f(d.get("Start"), 0.0), _f(d.get("End"), duration), str(d.get("Content", "")).strip())
                for d in parsed
            ]
        else:  # malformed / truncated JSON
            segments = None
            pieces = [Piece(0.0, duration, " ".join(extract_contents(raw)) or raw.strip())]

        pieces = [p for p in pieces if p.text]
        text = " ".join(p.text for p in pieces)
        issues = diagnose(text, duration, segments, hit_limit, guard.reason, max_wps=self.max_wps)
        return _Attempt(pieces, issues)

    @staticmethod
    def _absolute(pieces: list[Piece], offset: float, flags: list[str]) -> list[Piece]:
        return [Piece(offset + p.start, offset + p.end, p.text, list(flags)) for p in pieces]


def _f(x, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
