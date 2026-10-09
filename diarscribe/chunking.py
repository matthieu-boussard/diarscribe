"""Turn diarization segments into bounded ASR chunks.

Short, speech-only chunks are the first line of defence against decoder loops and hallucinations:
the model never sees long silences (a classic trigger) and its output length is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    start: float
    end: float
    speaker: int
    text: str
    flags: list[str] = field(default_factory=list)


def merge_turns(turns: list[Turn], max_gap: float = 0.6) -> list[Turn]:
    """Merge consecutive turns of the same speaker separated by less than ``max_gap`` s."""
    merged: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        if t.duration <= 0:
            continue
        last = merged[-1] if merged else None
        if last and last.speaker == t.speaker and t.start - last.end <= max_gap:
            last.end = max(last.end, t.end)
        else:
            merged.append(Turn(t.start, t.end, t.speaker))
    return merged


def relabel_speakers(segments: list[Segment]) -> list[Segment]:
    """Renumber speakers 0, 1, 2... in order of first appearance."""
    mapping: dict[int, int] = {}
    for s in sorted(segments, key=lambda s: s.start):
        s.speaker = mapping.setdefault(s.speaker, len(mapping))
    return segments


def fmt_ts(t: float) -> str:
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"
