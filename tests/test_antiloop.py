import json

import torch

from diarscribe.antiloop import (
    LoopStoppingCriteria,
    collapse_repetitions,
    diagnose,
    extract_contents,
    find_periodic_suffix,
)
from diarscribe.chunking import Turn, build_windows, dominant_speaker, merge_turns


def test_periodic_suffix_detects_long_loop():
    seq = list(range(50)) + [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] * 4
    assert find_periodic_suffix(seq) == (11, 4)


def test_periodic_suffix_short_unit_needs_many_repeats():
    assert find_periodic_suffix([1, 2, 3] + [5] * 6) is None
    assert find_periodic_suffix([1, 2, 3] + [5] * 12) is not None


def test_periodic_suffix_ignores_normal_text():
    seq = list(range(300))
    assert find_periodic_suffix(seq) is None


class _FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


def test_stopping_criteria_fires_and_records_reason():
    prompt = torch.arange(100, 120)
    gen = torch.tensor(list(range(200, 230)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4)
    ids = torch.cat([prompt, gen]).unsqueeze(0)
    crit = LoopStoppingCriteria(prompt_len=20, tokenizer=_FakeTok())
    assert crit(ids, None).item() is True
    assert "token loop" in crit.reason


def test_stopping_criteria_catches_same_sentence_with_new_timestamps():
    # Distinct tokens each time (timestamps change) but the words loop.
    segs = [{"Start": i * 2, "End": i * 2 + 2, "Speaker": 0, "Content": "merci beaucoup à tous pour votre attention"} for i in range(5)]
    text = json.dumps(segs, ensure_ascii=False)

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return text

    ids = torch.arange(1000, 1000 + 64).unsqueeze(0)  # 64 unique tokens: no token-level loop
    crit = LoopStoppingCriteria(prompt_len=0, tokenizer=Tok())
    assert crit(ids, None).item() is True
    assert "text loop" in crit.reason


def test_stopping_criteria_quiet_on_normal_output():
    ids = torch.arange(0, 200).unsqueeze(0)
    crit = LoopStoppingCriteria(prompt_len=10, tokenizer=_FakeTok())
    assert crit(ids, None).item() is False


def test_diagnose_flags_loops_and_passes_clean_text():
    clean = "Bonjour à tous, aujourd'hui on va parler de la diarisation sur Mac."
    assert diagnose(clean, duration=5.0) == []

    loop = "Bonjour à tous. " + "Merci de votre attention. " * 12
    issues = diagnose(loop, duration=6.0)
    assert any("compression" in i for i in issues)
    assert any("repeated" in i for i in issues)
    assert any("words/s" in i for i in issues)


def test_diagnose_timestamps():
    segs = [{"Start": 0, "End": 5}, {"Start": 5, "End": 50}]
    assert "timestamps beyond audio" in diagnose("ok", 10.0, segs)
    segs = [{"Start": 0, "End": 5}, {"Start": 6, "End": 8}, {"Start": 1, "End": 3}]
    assert "timestamps going backwards" in diagnose("ok", 10.0, segs)


def test_collapse_repetitions():
    assert collapse_repetitions("je pense que " * 5 + "oui") == "je pense que oui"
    assert collapse_repetitions("non non non non non non d'accord") == "non non d'accord"
    assert collapse_repetitions("très très bien") == "très très bien"


def test_extract_contents_on_truncated_json():
    raw = '[{"Start":0,"End":2,"Speaker":0,"Content":"Salut \\"toi\\""},{"Start":2,"End":4,"Speaker":1,"Content":"ça va'
    assert extract_contents(raw) == ['Salut "toi"', "ça va"]


def test_merge_and_windows():
    turns = [Turn(0, 2, 0), Turn(2.3, 4, 0), Turn(4.1, 6, 1), Turn(10, 12, 1)]
    merged = merge_turns(turns)
    assert [(t.start, t.end, t.speaker) for t in merged] == [(0, 4, 0), (4.1, 6, 1), (10, 12, 1)]
    assert build_windows(turns, max_len=30, max_gap=2.0) == [(0, 6), (10, 12)]
    assert dominant_speaker(3.5, 5.0, turns) == 1


def test_diagnose_tolerates_short_fast_burst():
    assert diagnose("Est-ce que c'est clair ? Est-ce que vous êtes écouté ?", duration=1.5) == []


def test_filters():
    from diarscribe.filters import clean

    assert clean("[Unintelligible Speech]")[1] == "non-speech tag"
    assert clean("Make sure to subscribe.")[1] == "known hallucination"
    assert clean("嗯嗯，嗯嗯。", lang="fr")[1] == "foreign script"
    assert clean("Ну, зачем?", lang="fr")[1] == "foreign script"
    assert clean("[Laughter] Oui, d'accord.", lang="fr") == ("Oui, d'accord.", None)
    assert clean("嗯嗯", lang=None) == ("嗯嗯", None)
