"""Transcript writers: txt, srt, json."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .chunking import Segment, fmt_ts


def to_txt(segments: list[Segment]) -> str:
    lines = []
    for s in segments:
        warn = " ⚠" if "loop_trimmed" in s.flags else ""
        lines.append(f"[{fmt_ts(s.start)} - {fmt_ts(s.end)}] SPEAKER_{s.speaker}{warn}: {s.text}")
    return "\n".join(lines) + "\n"


def to_srt(segments: list[Segment]) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    blocks = [f"{i}\n{ts(s.start)} --> {ts(s.end)}\n[SPEAKER_{s.speaker}] {s.text}\n" for i, s in enumerate(segments, 1)]
    return "\n".join(blocks)


def to_json(segments: list[Segment]) -> str:
    return json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2)


WRITERS = {"txt": to_txt, "srt": to_srt, "json": to_json}


def write_all(segments: list[Segment], base: Path, formats: list[str]) -> list[Path]:
    paths = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        path.write_text(WRITERS[fmt](segments), encoding="utf-8")
        paths.append(path)
    return paths
