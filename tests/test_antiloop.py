import torch

from diarscribe.antiloop import LoopStoppingCriteria, collapse_repetitions, diagnose, find_periodic_suffix
from diarscribe.chunking import Turn, merge_turns


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
    assert crit(ids, None).tolist() == [True]
    assert "token loop" in crit.reasons[0]


def test_stopping_criteria_catches_word_loop_with_distinct_tokens():
    # Distinct tokens each time (different tokenization) but the words loop.
    text = "merci beaucoup à tous pour votre attention et à bientôt " * 4

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return text

    ids = torch.arange(1000, 1000 + 64).unsqueeze(0)  # 64 unique tokens: no token-level loop
    crit = LoopStoppingCriteria(prompt_len=0, tokenizer=Tok())
    assert crit(ids, None).tolist() == [True]
    assert "text loop" in crit.reasons[0]


def test_stopping_criteria_is_per_row_and_ignores_padding():
    PAD = 0
    looping = list(range(200, 230)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4
    finished = list(range(300, 320)) + [PAD] * 50  # done early, padded: must not count as a loop
    normal = list(range(400, 470))
    ids = torch.tensor([looping, finished, normal])
    crit = LoopStoppingCriteria(prompt_len=0, tokenizer=_FakeTok(), ignore_ids={PAD})
    assert crit(ids, None).tolist() == [True, False, False]
    assert crit.reasons[1] is None and crit.reasons[2] is None


def test_stopping_criteria_quiet_on_normal_output():
    ids = torch.arange(0, 200).unsqueeze(0)
    crit = LoopStoppingCriteria(prompt_len=10, tokenizer=_FakeTok())
    assert crit(ids, None).tolist() == [False]


def test_diagnose_flags_loops_and_passes_clean_text():
    clean = "Bonjour à tous, aujourd'hui on va parler de la diarisation sur Mac."
    assert diagnose(clean, duration=5.0) == []

    loop = "Bonjour à tous. " + "Merci de votre attention. " * 12
    issues = diagnose(loop, duration=6.0)
    assert any("compression" in i for i in issues)
    assert any("repeated" in i for i in issues)
    assert any("words/s" in i for i in issues)


def test_collapse_repetitions():
    assert collapse_repetitions("je pense que " * 5 + "oui") == "je pense que oui"
    assert collapse_repetitions("non non non non non non d'accord") == "non non d'accord"
    assert collapse_repetitions("très très bien") == "très très bien"


def test_merge_turns():
    turns = [Turn(0, 2, 0), Turn(2.3, 4, 0), Turn(4.1, 6, 1), Turn(10, 12, 1)]
    merged = merge_turns(turns)
    assert [(t.start, t.end, t.speaker) for t in merged] == [(0, 4, 0), (4.1, 6, 1), (10, 12, 1)]


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
    assert clean("Merci.", lang="fr", duration=0.4)[1] == "short-turn hallucination"
    assert clean("Merci.", lang="fr", duration=2.5) == ("Merci.", None)
    assert clean("Merci, bonne soirée.", lang="fr", duration=0.8) == ("Merci, bonne soirée.", None)
