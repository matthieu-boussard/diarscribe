"""Diarize → transcribe (batched, loop-guarded) → attribute text to speakers → segments.

Two modes:
  * ``window`` (default): the ASR backend (Cohere Transcribe or VibeVoice-ASR) transcribes continuous
    multi-speaker windows (up to 30 s of context),
    wav2vec2 aligns every word on the audio, and each word takes the Nemotron speaker active at that time.
  * ``turns``: one ASR call per Nemotron speaker turn. Strict attribution, but the median turn is ~1 s
    long, which leaves the model little context.
"""

from __future__ import annotations

import logging
import time

from .align import Aligner, Word
from .asr import GuardedTranscriber, Piece
from .audio import best_cut, load_audio
from .chunking import Segment, Turn, build_windows, fmt_ts as _ts, merge_turns, relabel_speakers, speaker_at
from .diarize import Diarizer
from .filters import clean

log = logging.getLogger(__name__)


def run(
    path: str,
    diarizer: Diarizer,
    transcriber: GuardedTranscriber,
    aligner: Aligner | None = None,
    mode: str = "window",
    max_chunk_s: float = 30.0,
    pad_s: float = 0.15,
    min_turn_s: float = 0.15,
    keep_tags: bool = False,
) -> list[Segment]:
    t0 = time.time()
    sr = diarizer.sampling_rate  # diarization, alignment and silence cuts work at 16 kHz
    if aligner is not None and aligner.sample_rate != sr:
        raise ValueError(f"diarisation à {sr} Hz, alignement à {aligner.sample_rate} Hz")
    audio = load_audio(path, sr)
    total = len(audio) / sr
    log.info("Audio : %s (%s)", path, _ts(total))

    turns = diarizer(audio)
    log.info("Diarisation : %d segments, %d locuteurs (%.0fs)", len(turns), len({t.speaker for t in turns}), time.time() - t0)
    # Micro-turns (< min_turn_s) are diarization noise; the decoder invents "Merci." on them.
    turns = [t for t in merge_turns(turns) if t.duration >= min_turn_s]
    if not turns:
        return []

    # Padded windows must fit the model's input (Whisper: 30 s).
    max_chunk_s = min(max_chunk_s, transcriber.max_input_s - 2 * pad_s)
    if mode == "window":
        if aligner is None:
            raise ValueError("le mode window demande un aligneur")
        spans = _split_long([(a, b, None) for a, b in build_windows(turns, max_chunk_s)], audio, sr, max_chunk_s)
    else:
        spans = _split_long([(t.start, t.end, t.speaker) for t in turns], audio, sr, max_chunk_s)
    padded = [(max(0.0, a - pad_s), min(total, b + pad_s)) for a, b, _ in spans]

    t1 = time.time()
    # Times are in seconds, so the ASR can run on its own sample rate (VibeVoice: 24 kHz).
    asr_audio = audio if transcriber.sample_rate == sr else load_audio(path, transcriber.sample_rate)
    results = transcriber.transcribe(asr_audio, padded)
    del asr_audio
    log.info("Transcription : %d %s en %.0fs", len(spans), "fenêtres" if mode == "window" else "tours", time.time() - t1)

    if mode == "window":
        t2 = time.time()
        segments = _attribute_words(audio, results, turns, aligner, transcriber.language, keep_tags)
        log.info("Alignement et attribution : %.0fs", time.time() - t2)
    else:
        segments = _turn_segments(spans, results, transcriber.language, keep_tags)

    log.info("Terminé en %.0fs (audio %s)", time.time() - t0, _ts(total))
    return relabel_speakers(segments)


def _attribute_words(audio, results: list[list[Piece]], turns: list[Turn], aligner: Aligner,
                     language: str, keep_tags: bool, max_pause: float = 1.0) -> list[Segment]:
    """Align each window's words, give each word its speaker, and group consecutive words into segments."""
    words: list[tuple[Word, int, list[str]]] = []
    for pieces in results:
        for p in pieces:
            text, dropped = clean(p.text, language, keep_tags)
            if dropped:
                log.info("[%s-%s] ignoré (%s)", _ts(p.start), _ts(p.end), dropped)
                continue
            aligned = aligner.align(audio, p.start, p.end, text)
            speakers = smooth_by_sentence(aligned, [speaker_at(w.start, w.end, turns) for w in aligned])
            words += [(w, spk, p.flags) for w, spk in zip(aligned, speakers)]

    segments: list[Segment] = []
    cur: list[tuple[Word, int, list[str]]] = []

    def flush():
        if not cur:
            return
        seg = Segment(cur[0][0].start, cur[-1][0].end, cur[0][1], " ".join(w.text for w, _, _ in cur),
                      sorted({f for _, _, fl in cur for f in fl}))
        text, dropped = clean(seg.text, language, keep_tags, duration=seg.end - seg.start)
        if dropped:
            log.info("[%s-%s] ignoré (%s)", _ts(seg.start), _ts(seg.end), dropped)
        else:
            seg.text = text
            segments.append(seg)
        cur.clear()

    for item in sorted(words, key=lambda x: x[0].start):
        if cur and (item[1] != cur[-1][1] or item[0].start - cur[-1][0].end > max_pause):
            flush()
        cur.append(item)
    flush()
    return segments


def smooth_by_sentence(words: list[Word], speakers: list[int], min_run: int = 4, max_sentence: int = 40) -> list[int]:
    """One speaker per sentence (majority of words), using the ASR punctuation as sentence ends.

    Word timings near a turn change are fuzzy, so edge words often land on the other speaker. A run of at
    least ``min_run`` words of another speaker inside a sentence is kept: a real change without punctuation.
    A "sentence" longer than ``max_sentence`` words means the text is not punctuated (e.g. --no-punctuation):
    then only isolated single words are relabelled.
    """
    out = list(speakers)
    start = 0
    for i, w in enumerate(words):
        if i == len(words) - 1 or w.text.rstrip("»\"')").endswith((".", "?", "!", "…")):
            idx = range(start, i + 1)
            weight: dict[int, float] = {}
            for k in idx:
                # Count words, not seconds: the aligner stretches the last word of a turn into the next one.
                weight[speakers[k]] = weight.get(speakers[k], 0.0) + 1.0 + 1e-3 * (words[k].end - words[k].start)
            major = max(weight, key=weight.get)
            run_min = min_run if i + 1 - start <= max_sentence else 2
            k = start
            while k <= i:  # relabel minority runs shorter than min_run
                j = k
                while j + 1 <= i and speakers[j + 1] == speakers[k]:
                    j += 1
                short = j - k + 1 < run_min
                if run_min == min_run:
                    fill = major if short and speakers[k] != major else None
                else:  # unpunctuated text: an isolated word takes its left neighbour's speaker
                    fill = out[k - 1] if short and k > start and speakers[k] != out[k - 1] else None
                if fill is not None:
                    for m in range(k, j + 1):
                        out[m] = fill
                k = j + 1
            start = i + 1
    return out


def _turn_segments(spans, results: list[list[Piece]], language: str, keep_tags: bool) -> list[Segment]:
    segments: list[Segment] = []
    for (start, end, speaker), pieces in zip(spans, results):
        texts, flags = [], set()
        for p in pieces:
            text, dropped = clean(p.text, language, keep_tags, duration=p.end - p.start)
            if dropped:
                log.info("[%s-%s] ignoré (%s)", _ts(p.start), _ts(p.end), dropped)
            elif text:
                texts.append(text)
                flags.update(p.flags)
        if texts:
            segments.append(Segment(start, end, speaker, " ".join(texts), sorted(flags)))
    return segments


def _split_long(spans: list[tuple[float, float, int | None]], audio, sr: int, max_len: float):
    """Split spans longer than ``max_len`` at the quietest point near each boundary."""
    out = []
    for start, end, spk in spans:
        while end - start > max_len:
            cut = best_cut(audio, sr, start, end, target=start + max_len - 3.0, search=3.0)
            out.append((start, cut, spk))
            start = cut
        out.append((start, end, spk))
    return out
