"""Two-model fusion (default: Cohere Transcribe + Whisper large-v3 FR) arbitrated by the CTC aligner.

Both models transcribe the same windows. Their word sequences are aligned; where they disagree, the
wav2vec2 aligner (an independent acoustic model) scores each candidate by how well the whole window's
transcription explains the audio (Viterbi CTC log-likelihood + ``token_bonus`` per character token). The
secondary model's words are taken only when they score clearly better (``margin`` nats): the primary model,
more accurate on average, wins ties.

Calibrated on a 5-min French meeting excerpt against two independent referees (VibeVoice, Hojo): without the
token bonus the best path favours blank, so real words of the primary get dropped ("c'est bon.");
token_bonus=6 / margin=10 keeps them and brings the fused text closer to both referees than either model alone.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

from .align import Aligner
from .asr import GuardedTranscriber, Piece

log = logging.getLogger(__name__)


class FusedTranscriber:
    def __init__(self, primary: GuardedTranscriber, secondary: GuardedTranscriber, aligner: Aligner, margin: float = 10.0,
                 token_bonus: float = 6.0):
        if not primary.sample_rate == secondary.sample_rate == aligner.sample_rate:
            raise ValueError("la fusion demande des modèles à la même fréquence d'échantillonnage")
        self.primary, self.secondary, self.aligner, self.margin = primary, secondary, aligner, margin
        self.token_bonus = token_bonus
        self.sample_rate = primary.sample_rate
        self.max_input_s = min(primary.max_input_s, secondary.max_input_s)
        self.language = primary.language
        self.stats: Counter = Counter()

    def transcribe(self, audio: np.ndarray, spans: list[tuple[float, float]]) -> list[list[Piece]]:
        res_a = self.primary.transcribe(audio, spans)
        res_b = self.secondary.transcribe(audio, spans)
        out = []
        for (start, end), pa, pb in zip(spans, res_a, res_b):
            text = self.fuse(audio, start, end, " ".join(p.text for p in pa), " ".join(p.text for p in pb))
            flags = sorted({f for p in pa + pb for f in p.flags})
            out.append([Piece(start, end, text, flags)] if text else [])
        return out

    def fuse(self, audio: np.ndarray, start: float, end: float, text_a: str, text_b: str) -> str:
        wa, wb = text_a.split(), text_b.split()
        ops = SequenceMatcher(None, [_key(w) for w in wa], [_key(w) for w in wb], autojunk=False).get_opcodes()
        self.stats["windows"] += 1
        self.stats["words"] += len(wa)
        if all(op[0] == "equal" for op in ops):
            return text_a
        emission, _ = self.aligner.emission(audio, start, end)
        choice = ["a"] * len(ops)

        def words() -> list[str]:
            return [w for (_, i1, i2, j1, j2), c in zip(ops, choice) for w in (wa[i1:i2] if c == "a" else wb[j1:j2])]

        best = self.aligner.score(emission, words(), self.token_bonus)
        for k, (tag, i1, i2, j1, j2) in enumerate(ops):
            if tag == "equal":
                continue
            self.stats["regions"] += 1
            if _same_numbers(wa[i1:i2], wb[j1:j2]):  # "dix" vs "10": a style difference, not a disagreement
                self.stats["number_variants"] += 1
                continue
            choice[k] = "b"
            alt = self.aligner.score(emission, words(), self.token_bonus)
            if alt > best + self.margin:
                log.debug("[%.1f-%.1f] %r -> %r (+%.1f)", start, end, " ".join(wa[i1:i2]), " ".join(wb[j1:j2]), alt - best)
                best = alt
                self.stats["secondary_chosen"] += 1
            else:
                choice[k] = "a"
        return " ".join(words())

    def summary(self) -> str:
        s = self.stats
        return (f"fusion : {s['regions']} désaccord(s) sur {s['windows']} fenêtres, {s['secondary_chosen']} tranché(s) "
                f"pour le modèle secondaire, {s['number_variants']} simple(s) variante(s) de nombre")


def _key(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower().replace("’", "'"))


def _same_numbers(a: list[str], b: list[str]) -> bool:
    from text_to_num import alpha2digit

    na, nb = (_key(alpha2digit(" ".join(x), "fr")) for x in (a, b))
    return bool(na) and na == nb
