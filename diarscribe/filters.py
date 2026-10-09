"""Drop hallucinated output VibeVoice produces on very short / non-speech audio."""

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

_LATIN_LANGS = {"fr", "en", "es", "it", "de", "pt", "nl", "ca", "ro", "pl", "cs", "sv", "da", "no", "fi", "tr", "id", "vi"}


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(unicodedata.name(c, "").startswith("LATIN") for c in letters) / len(letters)


def clean(text: str, lang: str | None = None, keep_tags: bool = False) -> tuple[str, str | None]:
    """Return (cleaned_text, reason_dropped). ``reason_dropped`` is set when nothing usable remains."""
    if not keep_tags:
        text = re.sub(r"\s{2,}", " ", _TAG_RE.sub(" ", text)).strip()
    if not text:
        return "", "non-speech tag"
    if " ".join(normalize_words(text)) in _HALLUCINATIONS:
        return "", "known hallucination"
    if lang in _LATIN_LANGS and latin_ratio(text) < 0.5:
        return "", "foreign script"
    return text, None
