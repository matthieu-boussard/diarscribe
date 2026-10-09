"""Audio loading (any format, via ffmpeg) and silence-aware cut points."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np


def load_audio(path: str, sr: int) -> np.ndarray:
    """Decode any file ffmpeg understands (wav, mp3, m4a, mov...) to mono float32 at ``sr``."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg introuvable : installez-le avec `brew install ffmpeg`.")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué sur {path}: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def best_cut(audio: np.ndarray, sr: int, start: float, end: float, target: float | None = None,
             search: float = 3.0, win: float = 0.05) -> float:
    """Time (s) of the quietest ``win``-long frame near ``target`` inside (start, end)."""
    if target is None:
        target = (start + end) / 2
    lo = max(start + 1.0, target - search)
    hi = min(end - 1.0, target + search)
    if hi <= lo:
        return target
    a, b = int(lo * sr), int(hi * sr)
    hop = max(1, int(win * sr))
    seg = audio[a:b]
    n_frames = len(seg) // hop
    if n_frames < 2:
        return target
    frames = seg[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return lo + (int(rms.argmin()) + 0.5) * win
