"""Drop hallucinated output ASR models produce on very short / non-speech audio."""

from __future__ import annotations

import re
import unicodedata

from .antiloop import normalize_words

_TAG_RE = re.compile(r"\[[^\]]{0,40}\]")

# Classic ASR training-data leftovers (YouTube outros, subtitle credits).
_HALLUCINATIONS = {
    "make sure to subscribe", "subscribe", "like", "like and subscribe", "thanks for watching",
    "thank you for watching", "merci d avoir regardé", "merci d avoir regardé cette vidéo",
    "sous titres réalisés par la communauté d amara org", "abonnez vous",
}

# Filler the decoder emits on clicks / breaths: only trusted on turns long enough to hold real speech.
_SHORT_TURN_HALLUCINATIONS = {"merci", "thank you", "thanks"}
SHORT_TURN_S = 1.0

_LATIN_LANGS = {"fr", "en", "es", "it", "de", "pt", "nl", "ca", "ro", "pl", "cs", "sv", "da", "no", "fi", "tr", "id", "vi"}


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(unicodedata.name(c, "").startswith("LATIN") for c in letters) / len(letters)


def clean(text: str, lang: str | None = None, keep_tags: bool = False,
          duration: float | None = None) -> tuple[str, str | None]:
    """Return (cleaned_text, reason_dropped). ``reason_dropped`` is set when nothing usable remains."""
    if not keep_tags:
        text = re.sub(r"\s{2,}", " ", _TAG_RE.sub(" ", text)).strip()
    if not text:
        return "", "non-speech tag"
    key = " ".join(normalize_words(text))
    if key in _HALLUCINATIONS:
        return "", "known hallucination"
    if duration is not None and duration < SHORT_TURN_S and key in _SHORT_TURN_HALLUCINATIONS:
        return "", "short-turn hallucination"
    if lang in _LATIN_LANGS and latin_ratio(text) < 0.5:
        return "", "foreign script"
    return text, None
