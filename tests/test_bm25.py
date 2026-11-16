from collections import Counter

import numpy as np

from tesseract_mcp import bm25
from tesseract_mcp.bm25 import tokenize


def score(corpus, query):
    paths = list(corpus)
    tokens = [tokenize(corpus[p]) for p in paths]
    totals = np.array([len(t) for t in tokens], dtype=np.float64)
    return dict(zip(paths, bm25.scores([Counter(t) for t in tokens], totals, query)))


def test_tokenize_lowercases_and_splits():
    assert tokenize("Sentinel-ESG Pipeline!") == ["sentinel", "esg", "pipeline"]


def test_score_is_zero_without_a_shared_token():
    corpus = {
        "a.md": "the sentinel esg pipeline ingests incident data",
        "b.md": "an unrelated note about weather patterns",
    }
    got = score(corpus, "sentinel pipeline")
    assert got["a.md"] > 0
    assert got["b.md"] == 0


def test_score_favors_rare_term_match():
    corpus = {
        "common.md": "the the the the the project project",
        "rare.md": "zephyr appears exactly once here",
    }
    got = score(corpus, "zephyr")
    assert got["rare.md"] > 0 and got["common.md"] == 0


def test_score_levels_off_as_a_word_repeats():
    corpus = {
        "once.md": "budget " + "filler " * 40,
        "twice.md": "budget budget " + "filler " * 39,
        "many.md": "budget " * 40 + "filler",
        "other.md": "unrelated words only",
    }
    got = score(corpus, "budget")
    # Doubling the count adds less than the first mention did.
    assert got["twice.md"] - got["once.md"] < got["once.md"]
    assert got["many.md"] < 3 * got["once.md"]


def test_stuffed_note_does_not_run_away_on_a_repeated_query_word():
    corpus = {
        "short.md": "zephyr launch notes",
        "stuffed.md": "zephyr " * 60 + "alpha beta gamma delta " * 10,
        "other.md": "weather patterns today",
        "other2.md": "another unrelated page",
    }
    got = score(corpus, "zephyr zephyr zephyr")
    assert got["short.md"] > got["stuffed.md"] / 3


def test_score_empty_corpus_returns_empty():
    assert len(bm25.scores([], np.zeros(0), "anything")) == 0


def test_score_no_match_is_all_zero():
    assert not score({"a.md": "completely unrelated content"}, "zzzznomatch")["a.md"]
