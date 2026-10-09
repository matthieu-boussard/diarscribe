"""Diarize → chunk → transcribe (with loop guard) → assemble speaker segments."""

from __future__ import annotations

import logging
import time

from .asr import SAMPLE_RATE as ASR_SR
from .asr import Transcriber
from .audio import best_cut, load_audio
from .chunking import Segment, build_windows, dominant_speaker, fmt_ts as _ts, merge_turns, relabel_speakers
from .diarize import Diarizer
from .filters import clean

log = logging.getLogger(__name__)


def run(
    path: str,
    diarizer: Diarizer,
    transcriber: Transcriber,
    mode: str = "window",
    max_chunk_s: float = 30.0,
    pad_s: float = 0.15,
    lang: str | None = None,
    keep_tags: bool = False,
) -> list[Segment]:
    t0 = time.time()
    audio16 = load_audio(path, diarizer.sampling_rate)
    total = len(audio16) / diarizer.sampling_rate
    log.info("Audio : %s (%s)", path, _ts(total))

    turns = diarizer(audio16)
    del audio16
    log.info("Diarisation : %d segments, %d locuteurs (%.0fs)", len(turns), len({t.speaker for t in turns}), time.time() - t0)
    if not turns:
        return []

    audio24 = load_audio(path, ASR_SR)
    total = len(audio24) / ASR_SR

    if mode == "turns":
        # One ASR call per speaker turn: speaker attribution comes straight from Nemotron.
        spans = [(t.start, t.end, t.speaker) for t in merge_turns(turns)]
    else:
        # Multi-speaker windows: sub-segments are attributed by overlap with the diarization.
        spans = [(a, b, None) for a, b in build_windows(turns, max_chunk_s)]
    spans = _split_long(spans, audio24, max_chunk_s)

    segments: list[Segment] = []
    for i, (start, end, speaker) in enumerate(spans, 1):
        a, b = max(0.0, start - pad_s), min(total, end + pad_s)
        pieces = transcriber.transcribe_span(audio24, a, b)
        for p in pieces:
            p.text, dropped = clean(p.text, lang, keep_tags)
            if dropped:
                log.info("[%s-%s] ignoré (%s)", _ts(p.start), _ts(p.end), dropped)
        pieces = [p for p in pieces if p.text]
        log.info("[%d/%d] %s-%s : %d morceau(x)", i, len(spans), _ts(a), _ts(b), len(pieces))
        if speaker is not None:
            text = " ".join(p.text for p in pieces).strip()
            if text:
                flags = sorted({f for p in pieces for f in p.flags})
                segments.append(Segment(start, end, speaker, text, flags))
        else:
            for p in pieces:
                p_start, p_end = max(a, min(p.start, b)), max(a, min(p.end, b))
                spk = dominant_speaker(p_start, p_end, turns)
                segments.append(Segment(p_start, p_end, spk if spk is not None else -1, p.text, p.flags))

    log.info("Terminé en %.0fs (audio %s)", time.time() - t0, _ts(total))
    return relabel_speakers(sorted(segments, key=lambda s: s.start))


def _split_long(spans: list[tuple[float, float, int | None]], audio, max_len: float):
    """Split spans longer than ``max_len`` at the quietest point near each boundary."""
    out = []
    for start, end, spk in spans:
        while end - start > max_len:
            cut = best_cut(audio, ASR_SR, start, end, target=start + max_len - 3.0, search=3.0)
            out.append((start, cut, spk))
            start = cut
        out.append((start, end, spk))
    return out

