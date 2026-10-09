"""Batched ASR backends (Cohere Transcribe, VibeVoice-ASR) wrapped in a shared anti-loop retry ladder.

``GuardedTranscriber`` owns everything backend-independent: batching, live loop stopping, diagnosis,
retries, recursive splitting and last-resort cleanup. A backend only implements ``_generate_batch``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    StoppingCriteriaList,
    VibeVoiceAsrForConditionalGeneration,
)

from .antiloop import LoopStoppingCriteria, collapse_repetitions, diagnose
from .audio import best_cut
from .chunking import fmt_ts as _ts

log = logging.getLogger(__name__)

# Each rung is tried only on the spans that looped / looked hallucinated at the previous one.
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
    text: str
    issues: list[str]


class GuardedTranscriber:
    """Backend-independent batching + anti-loop logic. Subclasses set ``sample_rate`` and ``_generate_batch``."""

    sample_rate: int

    def __init__(self, model, processor, device: str, dtype: torch.dtype, language: str, batch_size: int,
                 tokens_per_s: float, base_tokens: int, max_wps: float = 6.0,
                 max_split_depth: int = 2, min_split_s: float = 4.0):
        self.model, self.processor = model.to(device).eval(), processor
        self.device, self.dtype = device, dtype
        self.language = language
        self.batch_size = batch_size
        self.tokens_per_s, self.base_tokens = tokens_per_s, base_tokens
        self.max_wps = max_wps
        self.max_split_depth, self.min_split_s = max_split_depth, min_split_s
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        pad = self.processor.tokenizer.pad_token_id
        self.ignore_ids = self.eos_ids | ({pad} if pad is not None else set())

    def transcribe(self, audio: np.ndarray, spans: list[tuple[float, float]], _depth: int = 0) -> list[list[Piece]]:
        """Transcribe each ``audio[start:end]`` (mono at ``sample_rate``); one list of pieces per span."""
        results: list[list[Piece] | None] = [None] * len(spans)
        pending = [i for i, (a, b) in enumerate(spans) if b - a >= 0.2]
        for i in set(range(len(spans))) - set(pending):
            results[i] = []
        attempts: dict[int, list[_Attempt]] = {i: [] for i in pending}

        for rung, cfg in enumerate(RETRY_LADDER):
            if not pending:
                break
            outs = self._generate_many(audio, [spans[i] for i in pending], cfg)
            still = []
            for i, att in zip(pending, outs):
                a, b = spans[i]
                if not att.issues:
                    results[i] = [Piece(a, b, att.text, ["retry"] if rung else [])]
                else:
                    log.info("[%s-%s] essai %d suspect : %s", _ts(a), _ts(b), rung + 1, "; ".join(att.issues))
                    attempts[i].append(att)
                    still.append(i)
            pending = still

        # Every rung looped: split at the quietest point and transcribe all halves in one batched call.
        splittable = [i for i in pending
                      if _depth < self.max_split_depth and spans[i][1] - spans[i][0] >= self.min_split_s]
        if splittable:
            halves = []
            for i in splittable:
                a, b = spans[i]
                cut = best_cut(audio, self.sample_rate, a, b)
                log.info("[%s-%s] découpage à %s", _ts(a), _ts(b), _ts(cut))
                halves += [(a, cut), (cut, b)]
            sub = self.transcribe(audio, halves, _depth + 1)
            for k, i in enumerate(splittable):
                results[i] = sub[2 * k] + sub[2 * k + 1]

        # Last resort: keep the least-bad attempt, strip the repetitions, cap its length.
        for i in set(pending) - set(splittable):
            a, b = spans[i]
            best = min(attempts[i], key=lambda t: (len(t.issues), len(t.text)))
            max_words = int(self.max_wps * max(b - a, 1.0)) + 5
            text = " ".join(collapse_repetitions(best.text).split()[:max_words])
            log.warning("[%s-%s] boucle non résolue, texte nettoyé : %s", _ts(a), _ts(b), "; ".join(best.issues))
            results[i] = [Piece(a, b, text, ["loop_trimmed", *best.issues])]
        return results  # type: ignore[return-value]

    def _generate_many(self, audio: np.ndarray, spans: list[tuple[float, float]], cfg: dict) -> list[_Attempt]:
        # Longest first so each batch holds similar durations (less padding).
        order = sorted(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0], reverse=True)
        out: list[_Attempt | None] = [None] * len(spans)
        for k in range(0, len(order), self.batch_size):
            idx = order[k : k + self.batch_size]
            for i, att in zip(idx, self._generate_batch(audio, [spans[i] for i in idx], cfg)):
                out[i] = att
        return out  # type: ignore[return-value]

    def _slices(self, audio: np.ndarray, spans: list[tuple[float, float]]) -> list[np.ndarray]:
        return [audio[int(a * self.sample_rate) : int(b * self.sample_rate)] for a, b in spans]

    @torch.inference_mode()
    def _run(self, inputs, spans: list[tuple[float, float]], cfg: dict, prompt_len: int) -> tuple[torch.Tensor, LoopStoppingCriteria, list[bool]]:
        """``generate`` with the loop guard; returns new tokens, the guard, and which rows hit the budget."""
        max_new = int(self.base_tokens + max(b - a for a, b in spans) * self.tokens_per_s)
        guard = LoopStoppingCriteria(prompt_len, self.processor.tokenizer, ignore_ids=self.ignore_ids)
        out = self.model.generate(**inputs, max_new_tokens=max_new, stopping_criteria=StoppingCriteriaList([guard]), **cfg)
        gen = out[:, prompt_len:]
        # Budget is per batch (longest span); a row hit it if it never emitted EOS and was not stopped.
        hit = [guard.reasons[r] is None and not any(t in self.eos_ids for t in gen[r].tolist()) for r in range(len(spans))]
        return gen, guard, hit

    def _attempts(self, texts: list[str], spans, guard: LoopStoppingCriteria, hit: list[bool]) -> list[_Attempt]:
        return [
            _Attempt(text, diagnose(text, b - a, hit_token_limit=hit[r], stop_reason=guard.reasons[r], max_wps=self.max_wps))
            for r, ((a, b), text) in enumerate(zip(spans, texts))
        ]

    def _generate_batch(self, audio: np.ndarray, spans: list[tuple[float, float]], cfg: dict) -> list[_Attempt]:
        raise NotImplementedError


class CohereTranscriber(GuardedTranscriber):
    """CohereLabs/cohere-transcribe-03-2026: 2B encoder-decoder, 16 kHz, language given, plain text output."""

    MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
    sample_rate = 16_000

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, model_id: str | None = None,
                 language: str = "fr", punctuation: bool = True, batch_size: int = 16, **kw):
        model_id = model_id or self.MODEL_ID
        super().__init__(CohereAsrForConditionalGeneration.from_pretrained(model_id, dtype=dtype),
                         AutoProcessor.from_pretrained(model_id), device, dtype, language, batch_size,
                         tokens_per_s=12.0, base_tokens=24, **kw)
        self.punctuation = punctuation

    def _generate_batch(self, audio, spans, cfg):
        inputs = self.processor(
            self._slices(audio, spans), sampling_rate=self.sample_rate, return_tensors="pt",
            language=self.language, punctuation=self.punctuation,
        ).to(self.device, self.dtype)
        gen, guard, hit = self._run(inputs, spans, cfg, prompt_len=inputs["decoder_input_ids"].shape[1])
        texts = [t.strip() for t in self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)]
        return self._attempts(texts, spans, guard, hit)


class VibeVoiceTranscriber(GuardedTranscriber):
    """microsoft/VibeVoice-ASR-HF: 7B decoder-only, 24 kHz, auto language, accepts hotwords / context.

    Its JSON output (own speakers + timestamps) is reduced to plain text: in window mode speakers come
    from Nemotron and word times from the aligner, like for Cohere.
    """

    MODEL_ID = "microsoft/VibeVoice-ASR-HF"
    sample_rate = 24_000

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, model_id: str | None = None,
                 language: str = "fr", context: str | None = None, batch_size: int = 1, **kw):
        model_id = model_id or self.MODEL_ID
        # JSON overhead (~25 tokens per segment) on top of the text: larger budget than Cohere.
        super().__init__(VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, dtype=dtype),
                         AutoProcessor.from_pretrained(model_id), device, dtype, language, batch_size,
                         tokens_per_s=15.0, base_tokens=80, **kw)
        self.context = context

    def _generate_batch(self, audio, spans, cfg):
        prompts = [self.context] * len(spans) if self.context else None
        inputs = self.processor.apply_transcription_request(self._slices(audio, spans), prompt=prompts)
        inputs = inputs.to(self.device, self.dtype)
        # Prompts are left-padded, so every row's generation starts at the same index.
        gen, guard, hit = self._run(inputs, spans, cfg, prompt_len=inputs["input_ids"].shape[1])
        raws = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return self._attempts([self._text(raw) for raw in raws], spans, guard, hit)

    def _text(self, raw: str) -> str:
        try:  # raises on JSON truncated by the loop guard
            parsed = self.processor.extract_speaker_dict(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(d, dict) for d in parsed):
            contents = [str(d.get("Content", "")).strip() for d in parsed]
        else:
            contents = extract_contents(raw) or [raw.removeprefix("assistant").strip()]
        return " ".join(c for c in contents if c)


_CONTENT_RE = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"?')


def extract_contents(raw: str) -> list[str]:
    """Best-effort extraction of "Content" fields from (possibly truncated) VibeVoice JSON."""
    out = []
    for m in _CONTENT_RE.finditer(raw):
        try:
            out.append(json.loads(f'"{m.group(1)}"'))
        except json.JSONDecodeError:
            out.append(m.group(1))
    return out


BACKENDS = {"cohere": CohereTranscriber, "vibevoice": VibeVoiceTranscriber}
