"""Cohere Transcribe (batched) wrapped in an anti-loop retry ladder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration, StoppingCriteriaList

from .antiloop import LoopStoppingCriteria, collapse_repetitions, diagnose
from .audio import best_cut
from .chunking import fmt_ts as _ts

log = logging.getLogger(__name__)

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_RATE = 16_000

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


class Transcriber:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = MODEL_ID,
        language: str = "fr",
        punctuation: bool = True,
        batch_size: int = 16,
        tokens_per_s: float = 12.0,
        base_tokens: int = 24,
        max_wps: float = 6.0,
        max_split_depth: int = 2,
        min_split_s: float = 4.0,
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.device, self.dtype = device, dtype
        self.language, self.punctuation = language, punctuation
        self.batch_size = batch_size
        self.tokens_per_s, self.base_tokens = tokens_per_s, base_tokens
        self.max_wps = max_wps
        self.max_split_depth, self.min_split_s = max_split_depth, min_split_s
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        pad = self.processor.tokenizer.pad_token_id
        self.ignore_ids = self.eos_ids | ({pad} if pad is not None else set())

    def transcribe(self, audio: np.ndarray, spans: list[tuple[float, float]], _depth: int = 0) -> list[list[Piece]]:
        """Transcribe each ``audio[start:end]`` (16 kHz mono); one list of pieces per span, absolute times."""
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
                cut = best_cut(audio, SAMPLE_RATE, a, b)
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

    @torch.inference_mode()
    def _generate_batch(self, audio: np.ndarray, spans: list[tuple[float, float]], cfg: dict) -> list[_Attempt]:
        segs = [audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] for a, b in spans]
        inputs = self.processor(
            segs, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=self.language, punctuation=self.punctuation
        ).to(self.device, self.dtype)
        prompt_len = inputs["decoder_input_ids"].shape[1]
        max_dur = max(b - a for a, b in spans)
        max_new = int(self.base_tokens + max_dur * self.tokens_per_s)
        guard = LoopStoppingCriteria(prompt_len, self.processor.tokenizer, ignore_ids=self.ignore_ids)

        out = self.model.generate(
            **inputs, max_new_tokens=max_new, stopping_criteria=StoppingCriteriaList([guard]), **cfg
        )
        gen = out[:, prompt_len:]
        texts = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)

        attempts = []
        for row, ((a, b), text) in enumerate(zip(spans, texts)):
            ids = gen[row].tolist()
            # Budget is per batch (longest span); a row hit it if it never emitted EOS.
            hit_limit = guard.reasons[row] is None and not any(t in self.eos_ids for t in ids)
            text = text.strip()
            issues = diagnose(text, b - a, hit_token_limit=hit_limit, stop_reason=guard.reasons[row], max_wps=self.max_wps)
            attempts.append(_Attempt(text, issues))
        return attempts
