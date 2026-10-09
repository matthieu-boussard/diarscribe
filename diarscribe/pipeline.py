"""Diarize → split into speaker turns → batched transcription (with loop guard) → segments."""

from __future__ import annotations

import logging
import time

from .asr import SAMPLE_RATE, Transcriber
from .audio import best_cut, load_audio
from .chunking import Segment, fmt_ts as _ts, merge_turns, relabel_speakers
from .diarize import Diarizer
from .filters import clean

log = logging.getLogger(__name__)


def run(
    path: str,
    diarizer: Diarizer,
    transcriber: Transcriber,
    max_chunk_s: float = 30.0,
    pad_s: float = 0.15,
    min_turn_s: float = 0.15,
    keep_tags: bool = False,
) -> list[Segment]:
    t0 = time.time()
    if diarizer.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"diarisation à {diarizer.sampling_rate} Hz, ASR à {SAMPLE_RATE} Hz")
    audio = load_audio(path, SAMPLE_RATE)
    total = len(audio) / SAMPLE_RATE
    log.info("Audio : %s (%s)", path, _ts(total))

    turns = diarizer(audio)
    log.info("Diarisation : %d segments, %d locuteurs (%.0fs)", len(turns), len({t.speaker for t in turns}), time.time() - t0)
    if not turns:
        return []

    # Cohere Transcribe has no timestamps / speakers: each ASR span is one Nemotron speaker turn.
    # Micro-turns (< min_turn_s) are diarization noise; the decoder invents "Merci." on them.
    merged = [t for t in merge_turns(turns) if t.duration >= min_turn_s]
    spans = _split_long([(t.start, t.end, t.speaker) for t in merged], audio, max_chunk_s)
    padded = [(max(0.0, a - pad_s), min(total, b + pad_s)) for a, b, _ in spans]

    t1 = time.time()
    results = transcriber.transcribe(audio, padded)
    log.info("Transcription : %d tours en %.0fs", len(spans), time.time() - t1)

    segments: list[Segment] = []
    for (start, end, speaker), pieces in zip(spans, results):
        texts, flags = [], set()
        for p in pieces:
            text, dropped = clean(p.text, transcriber.language, keep_tags, duration=p.end - p.start)
            if dropped:
                log.info("[%s-%s] ignoré (%s)", _ts(p.start), _ts(p.end), dropped)
            elif text:
                texts.append(text)
                flags.update(p.flags)
        if texts:
            segments.append(Segment(start, end, speaker, " ".join(texts), sorted(flags)))

    log.info("Terminé en %.0fs (audio %s)", time.time() - t0, _ts(total))
    return relabel_speakers(segments)


def _split_long(spans: list[tuple[float, float, int]], audio, max_len: float):
    """Split spans longer than ``max_len`` at the quietest point near each boundary."""
    out = []
    for start, end, spk in spans:
        while end - start > max_len:
            cut = best_cut(audio, SAMPLE_RATE, start, end, target=start + max_len - 3.0, search=3.0)
            out.append((start, cut, spk))
            start = cut
        out.append((start, end, spk))
    return out
