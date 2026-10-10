import numpy as np

from diarscribe.fusion import FusedTranscriber


class FakeAligner:
    """Scores a hypothesis by how many of its words are in the "spoken" set, minus extra words."""

    sample_rate = 16_000

    def __init__(self, spoken):
        self.spoken = set(spoken)

    def emission(self, audio, start, end):
        return np.zeros((10, 3)), 0.02

    def score(self, emission, words, token_bonus=0.0):
        keys = [w.lower().strip(",.?!") for w in words]
        return 5.0 * sum(k in self.spoken for k in keys) - 5.0 * sum(k not in self.spoken for k in keys)


def make(spoken, margin=2.0):
    f = FusedTranscriber.__new__(FusedTranscriber)
    f.aligner, f.margin, f.token_bonus = FakeAligner(spoken), margin, 0.0
    from collections import Counter

    f.stats = Counter()
    return f


def test_fusion_takes_secondary_only_when_acoustically_better():
    f = make("ok on va vider les caches".split())
    out = f.fuse(None, 0, 1, "Ok, on va éviter les caches.", "on va vider les caches")
    assert out == "Ok, on va vider les caches."  # "éviter" -> "vider"; primary's punctuation/extra words kept
    assert f.stats["secondary_chosen"] == 1


def test_fusion_keeps_primary_on_ties_and_number_variants():
    f = make([])
    assert f.fuse(None, 0, 1, "ça fait dix ans", "ça fait 10 ans") == "ça fait dix ans"
    assert f.stats["number_variants"] == 1
    assert f.fuse(None, 0, 1, "identique", "identique") == "identique"
