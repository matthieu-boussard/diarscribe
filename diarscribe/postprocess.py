"""French text post-processing on finished segments.

Steps, in order (each optional, each counted in ``PostStats``):
  1. language filter: drop segments confidently in another language (hallucinations on short windows);
  2. glossary: restore proper nouns from a term list (explicit aliases + phonetic fuzzy match);
  3. style: "verbatim" keeps the speech as said; "lu" removes fillers (euh, hum) and stutters (je je);
  4. numbers: optional spelled-out -> digits ("dix ans" -> "10 ans");
  5. French typography: ’, non-breaking spaces before ; : ! ? and inside « », …, 1re / 2e.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .chunking import Segment

NNBSP, NBSP = " ", " "  # narrow no-break space (before ; ! ?), no-break space (before :, inside « »)


@dataclass
class PostConfig:
    lang: str = "fr"
    lang_filter: bool = True
    glossary: list[str] = field(default_factory=list)  # "Term" or "Term=alias1|alias2"
    style: str = "verbatim"  # or "lu"
    numbers: str = "keep"  # or "digits"
    typography: bool = True


@dataclass
class PostStats:
    segments_in: int = 0
    dropped_language: list[tuple[str, str]] = field(default_factory=list)  # (lang, text)
    glossary: Counter = field(default_factory=Counter)  # "variant -> Term"
    fillers: int = 0
    stutters: int = 0
    numbers: int = 0
    typography: int = 0
    seconds: float = 0.0


def postprocess(segments: list[Segment], cfg: PostConfig) -> tuple[list[Segment], PostStats]:
    t0 = time.perf_counter()
    stats = PostStats(segments_in=len(segments))
    detector = _language_detector(cfg.lang) if cfg.lang_filter else None
    glossary = Glossary(cfg.glossary) if cfg.glossary else None
    out = []
    for seg in segments:
        text = seg.text
        if detector is not None and (lang := detector(text)):
            stats.dropped_language.append((lang, text))
            continue
        if glossary:
            text = glossary.apply(text, stats.glossary)
        if cfg.style == "lu":
            text = remove_fillers(text, stats)
        if cfg.numbers == "digits" and cfg.lang == "fr":
            text = spelled_numbers_to_digits(text, stats)
        if cfg.typography and cfg.lang == "fr":
            new = french_typography(text)
            stats.typography += new != text
            text = new
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text:
            out.append(Segment(seg.start, seg.end, seg.speaker, text, seg.flags))
    stats.seconds = time.perf_counter() - t0
    return out, stats


# --- 1. language filter ---------------------------------------------------------------------------------------

_LINGUA = {"fr": "FRENCH", "en": "ENGLISH", "es": "SPANISH", "de": "GERMAN", "it": "ITALIAN", "pt": "PORTUGUESE",
           "nl": "DUTCH", "pl": "POLISH"}


def _language_detector(lang: str, min_words: int = 3, min_other: float = 0.6, max_target: float = 0.05):
    """Return ``detect(text) -> other_language | None``. Conservative: a segment is dropped only when it has
    ``min_words`` words, another language scores >= ``min_other`` and the expected one <= ``max_target``,
    so mixed French + English jargon ("Clear Seed Data, je sais pas où") is kept."""
    from lingua import Language, LanguageDetectorBuilder

    target = getattr(Language, _LINGUA[lang])
    candidates = [getattr(Language, name) for name in _LINGUA.values()]
    det = LanguageDetectorBuilder.from_languages(*candidates).build()

    def detect(text: str) -> str | None:
        text = re.sub(r"\S*\w[.@]\w\S*", " ", text)  # URLs, e-mails, file names: not language evidence
        if len(re.findall(r"\w+", text)) < min_words:
            return None
        values = det.compute_language_confidence_values(text)
        top = values[0]
        target_conf = next((v.value for v in values if v.language == target), 0.0)
        if top.language != target and top.value >= min_other and target_conf <= max_target:
            return top.language.iso_code_639_1.name.lower()
        return None

    return detect


# --- 2. glossary ----------------------------------------------------------------------------------------------

_PHONETIC_RULES = [  # rough French grapheme -> sound rules, applied in order on accent-free lowercase text
    ("eaux", "o"), ("eau", "o"), ("au", "o"), ("ph", "f"), ("qu", "k"), ("ck", "k"), ("ch", "S"), ("sh", "S"),
    ("ou", "u"), ("oi", "wa"), ("ai", "e"), ("ei", "e"), ("y", "i"), ("w", "v"), ("x", "ks"), ("z", "s"), ("h", ""),
]


def phonetic(text: str) -> str:
    """Crude French phonetic key: enough to match "Lynx" / "Lins" / "Liins", not a full G2P."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if c.isalpha())
    t = re.sub(r"c(?=[eiy])", "s", t)
    t = re.sub(r"g(?=[ei])", "j", t)
    for a, b in _PHONETIC_RULES:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    t = re.sub(r"(.)\1+", r"\1", t)  # double letters
    return re.sub(r"[stdxe]$", "", t)  # mute final letters


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Glossary:
    """Replace misrecognised proper nouns by their canonical spelling.

    Entries are "Term" (phonetic fuzzy match) or "Term=alias1|alias2" (aliases replaced as-is, case-insensitive).
    The fuzzy match is deliberately strict (measured on 4.5 h of meetings: looser rules turned "bizarre" into
    "Boussard" and "un cabinet en gestion" into "cabinet Liins", and added "AI" after a plain "Craft"):
      * a window has exactly as many words as the term (never fewer: no words are invented), or one more when a
        name was split in two ("Bou ssard");
      * every differing word must be uncommon in the language (wordfreq Zipf < ``max_zipf``): real words such
        as "bizarre", "en", "IA" are never replaced;
      * each differing word must sound like its counterpart (same first sound, similarity >= ``min_word_sim``)
        and the whole window like the term (>= ``min_sim``).
    """

    def __init__(self, entries: list[str], lang: str = "fr", min_sim: float = 0.75, min_word_sim: float = 0.6,
                 max_zipf: float = 4.0, min_key: int = 4):
        self.terms, self.aliases = [], []
        for e in entries:
            term, _, alias = e.partition("=")
            term = term.strip()
            if not term:
                continue
            self.terms.append((term, term.split()))
            for a in filter(None, (x.strip() for x in alias.split("|"))):
                self.aliases.append((re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", re.I), term))
        self.terms.sort(key=lambda t: -len(t[1]))  # "Craft AI" before "Craft"
        self.lang, self.min_sim, self.min_word_sim, self.max_zipf, self.min_key = lang, min_sim, min_word_sim, max_zipf, min_key

    def _common(self, word: str) -> bool:
        from wordfreq import zipf_frequency

        return zipf_frequency(_bare(word), self.lang) >= self.max_zipf

    def _match(self, window: list[str], term_words: list[str]) -> bool:
        bare_w, bare_t = [_bare(w) for w in window], [_bare(w) for w in term_words]
        if bare_w == bare_t or (len(window) > len(term_words) and set(bare_w) & set(bare_t)):
            return False  # already right, or a longer name containing the term ("Craft AI" vs term "Craft")
        diff = [(w, t) for w, t in zip(window, term_words) if _bare(w) != _bare(t)] if len(window) == len(term_words) \
            else [(" ".join(window), " ".join(term_words))]
        if any(self._common(w) for w, _ in diff if " " not in w) or (len(window) != len(term_words) and all(map(self._common, window))):
            return False
        for w, t in diff:
            kw, kt = phonetic(w) or _bare(w), phonetic(t) or _bare(t)  # short words ("AI") can have an empty key
            if not kw or kw[0] != kt[0] or 1 - _lev(kw, kt) / max(len(kw), len(kt)) < self.min_word_sim:
                return False
        kw, kt = phonetic(" ".join(window)), phonetic(" ".join(term_words))
        return 1 - _lev(kw, kt) / max(len(kw), len(kt)) >= self.min_sim

    def apply(self, text: str, hits: Counter) -> str:
        for pattern, term in self.aliases:
            text, n = pattern.subn(term, text)
            if n:
                hits[f"(alias) -> {term}"] += n
        tokens = text.split()
        for term, term_words in self.terms:
            if len(phonetic(term)) < self.min_key:
                continue
            n, i = len(term_words), 0
            while i < len(tokens):
                k = next((k for k in (n, n + 1) if i + k <= len(tokens) and self._match(tokens[i : i + k], term_words)), None)
                if k is None:
                    i += 1
                    continue
                lead = re.match(r"^[^\w]*", tokens[i]).group()
                trail = re.search(r"[^\w]*$", tokens[i + k - 1]).group()
                hits[f"{' '.join(_bare(t) for t in tokens[i : i + k])} -> {term}"] += 1
                tokens[i : i + k] = (lead + term + trail).split()
                i += n
        return " ".join(tokens)


def _bare(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()


# --- 3. style -------------------------------------------------------------------------------------------------

_FILLER = re.compile(r"(?<!\w)(?:euh+|heu+|hum+|hm+|mmh*|mm-hmm|hmm+)(?!\w)[,.…]*\s*", re.I)
# Emphasis, reflexive pronouns ("nous nous sommes"), topicalisation ("elle, elle est", "pour ça, ça coûte"),
# onomatopoeia.
_KEEP_REPEATED = {"oui", "ouais", "non", "si", "très", "bon", "voilà", "allez", "ok", "okay", "d'accord", "là", "vite",
                  "encore", "nous", "vous", "elle", "elles", "lui", "eux", "moi", "toi", "ça", "cela", "tac", "toc", "bla",
                  "ah", "oh"}


def remove_fillers(text: str, stats: PostStats) -> str:
    """'lu' style: drop hesitations and stutters ("je je", "on va on va"), keep emphatic repeats ("non, non")."""
    starts_upper = text[:1].isupper()
    text, n = _FILLER.subn("", text)
    stats.fillers += n
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        for size in (2, 1):  # "on va on va" before "je je"
            a = [w.lower().strip(",") for w in words[i : i + size]]
            b = [w.lower().strip(",") for w in words[i + size : i + 2 * size]]
            if len(b) == size and a == b and not set(a) & _KEEP_REPEATED and all(re.fullmatch(r"[^\W\d_][\w'’-]*", w) for w in a):
                stats.stutters += 1
                i += size  # drop the first copy, keep the second (it carries the punctuation)
                break
        else:
            out.append(words[i])
            i += 1
    text = " ".join(out)
    text = re.sub(r"^[\s,;.…]+", "", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    if starts_upper:  # "Euh, alors" -> "Alors"; a segment starting mid-sentence stays lowercase
        text = text[:1].upper() + text[1:]
    return text


# --- 4. numbers -----------------------------------------------------------------------------------------------

def spelled_numbers_to_digits(text: str, stats: PostStats) -> str:
    from text_to_num import alpha2digit

    new = alpha2digit(text, "fr")
    stats.numbers += sum(1 for _ in re.finditer(r"\d+", new)) - sum(1 for _ in re.finditer(r"\d+", text))
    return new


# --- 5. typography --------------------------------------------------------------------------------------------

def french_typography(text: str) -> str:
    t = re.sub(r"(?<=\w)'(?=\w)", "’", text)
    t = t.replace("...", "…")
    t = re.sub(r"\s*([;!?]+)", NNBSP + r"\1", t)
    t = re.sub(r"(?<!\d)\s*:(?!\d)", NBSP + ":", t)  # not in times (10:30)
    t = re.sub(r"«\s*", "«" + NBSP, t)
    t = re.sub(r"\s*»", NBSP + "»", t)
    t = re.sub(r"\b1(?:ère|ere)s?\b", lambda m: "1res" if m.group().endswith("s") else "1re", t)
    t = re.sub(r"\b(\d+)(?:ème|eme|ième)(s?)\b", r"\1e\2", t)
    t = re.sub(r"([.!?…])\s+([a-zà-ÿ])", lambda m: f"{m.group(1)} {m.group(2).upper()}", t)
    return re.sub(r"^" + "[" + NNBSP + NBSP + r"]+", "", t)


def summary(stats: PostStats) -> str:
    return (f"post-traitement : {len(stats.dropped_language)} segment(s) écarté(s) (langue), "
            f"{sum(stats.glossary.values())} terme(s) du glossaire, {stats.fillers} hésitation(s), "
            f"{stats.stutters} bégaiement(s), {stats.numbers} nombre(s), {stats.typography} retouche(s) typo "
            f"({1000 * stats.seconds:.0f} ms)")
