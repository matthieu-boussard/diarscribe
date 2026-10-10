from collections import Counter

from diarscribe.chunking import Segment
from diarscribe.postprocess import (
    NBSP,
    NNBSP,
    Glossary,
    PostConfig,
    PostStats,
    french_typography,
    phonetic,
    postprocess,
    remove_fillers,
)


def test_phonetic_key_groups_spelling_variants():
    assert phonetic("Liins") == phonetic("Lins")
    assert phonetic("Lynx")[:3] == phonetic("Liins")[:3]


def test_glossary_fuzzy_and_alias():
    g = Glossary(["cabinet Liins", "Craft AI=crafty ray|clarge tri", "Boussard"])
    hits = Counter()
    assert g.apply("moi je suis Virginie, cabinet Lynx.", hits) == "moi je suis Virginie, cabinet Liins."
    assert g.apply("tu viens de Clarge TRI, non ?", hits) == "tu viens de Craft AI, non ?"
    assert g.apply("monsieur Bousard est là", hits) == "monsieur Boussard est là"
    assert g.apply("on va faire la cuisine", hits) == "on va faire la cuisine"  # ordinary words untouched
    # Regressions seen on real meetings: common words, missing words, word eaten by the match.
    assert g.apply("c'est bizarre, regarde", hits) == "c'est bizarre, regarde"
    assert g.apply("un cabinet en gestion de patrimoine", hits) == "un cabinet en gestion de patrimoine"
    assert g.apply("je travaille chez Kraft, c'est bien", hits) == "je travaille chez Kraft, c'est bien"
    assert sum(hits.values()) == 3
    assert Glossary(["Craft AI"]).apply("le CSE de Craft eh", Counter()) == "le CSE de Craft eh"  # empty key word
    assert Glossary(["Craft"]).apply("chez Kraft, c'est bien", Counter()) == "chez Craft, c'est bien"
    assert Glossary(["Craft", "Craft AI"]).apply("chez Craft AI et chez Kraft", Counter()) == "chez Craft AI et chez Craft"


def test_style_lu_removes_fillers_and_stutters_keeps_emphasis():
    st = PostStats()
    assert remove_fillers("Euh, alors je je pense que, euh, on va on va le faire.", st) == "Alors je pense que, on va le faire."
    assert remove_fillers("Non, non, non, c'est clair.", st) == "Non, non, non, c'est clair."
    assert remove_fillers("nous nous sommes vus, elle, elle est partie, 1 1 2", st) == "nous nous sommes vus, elle, elle est partie, 1 1 2"
    assert remove_fillers("comme ça, ça sera bien", st) == "comme ça, ça sera bien"
    assert st.fillers == 2 and st.stutters == 2


def test_french_typography():
    t = french_typography("C'est clair? Oui: la 1ère fois, la 2ème... «bien»!")
    assert t == f"C’est clair{NNBSP}? Oui{NBSP}: la 1re fois, la 2e… «{NBSP}bien{NBSP}»{NNBSP}!"
    assert french_typography("rendez-vous à 10:30.") == "rendez-vous à 10:30."


def test_postprocess_language_filter_and_numbers():
    segs = [
        Segment(0, 1, 0, "en el año 1998 se fundó el grupo de música rock"),
        Segment(1, 2, 0, "ça fait dix ans que je fais le métier"),
        Segment(2, 3, 1, "Clear Seed Data, je sais pas où il faut aller"),
        Segment(3, 4, 1, "Ok."),
    ]
    segs.append(Segment(4, 5, 1, "D'accord, test.adcraft.in."))
    out, st = postprocess(segs, PostConfig(numbers="digits"))
    assert [s.text for s in out] == ["ça fait 10 ans que je fais le métier", "Clear Seed Data, je sais pas où il faut aller", "Ok.",
                                     "D’accord, test.adcraft.in."]
    assert st.dropped_language[0][0] == "es" and st.numbers == 1
