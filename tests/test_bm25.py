from tesseract_mcp.bm25 import rank, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Sentinel-ESG Pipeline!") == ["sentinel", "esg", "pipeline"]


def test_rank_favors_exact_term_over_no_match():
    corpus = {
        "a.md": "the sentinel esg pipeline ingests incident data",
        "b.md": "an unrelated note about weather patterns",
    }
    results = rank(corpus, "sentinel pipeline")
    assert [p for p, _ in results] == ["a.md"]


def test_rank_favors_rare_term_match():
    corpus = {
        "common.md": "the the the the the project project",
        "rare.md": "zephyr appears exactly once here",
    }
    results = rank(corpus, "zephyr")
    assert results and results[0][0] == "rare.md"


def test_rank_empty_corpus_returns_empty():
    assert rank({}, "anything") == []


def test_rank_no_match_returns_empty():
    corpus = {"a.md": "completely unrelated content"}
    assert rank(corpus, "zzzznomatch") == []


def test_rank_respects_limit():
    corpus = {f"n{i}.md": "shared keyword appears here" for i in range(10)}
    assert len(rank(corpus, "shared", limit=3)) == 3


FILLER = "alpha beta gamma delta epsilon zeta eta theta"


def _score(corpus, query, path):
    return dict(rank(corpus, query, limit=100))[path]


def test_repeated_term_saturates():
    # Same filler in every version: only the term frequency differs. The contribution is
    # bounded by idf*(k1+1), so going 5 -> 50 occurrences barely moves it.
    def note(n):
        return " ".join(["zephyr"] * n + FILLER.split() * 10)

    def corpus(n):
        return {"t.md": note(n), "o.md": "unrelated words only " * 10}

    s1, s5, s50 = (_score(corpus(n), "zephyr", "t.md") for n in (1, 5, 50))
    assert s1 < s5 < s50
    assert s50 - s5 < s5 - s1


def test_repeated_query_word_does_not_scale_with_stuffing():
    # A note stuffed with the term must not beat a short note that mentions
    # it once by more than the saturation bound (idf*(k1+1) = 2.5 idf).
    corpus = {
        "short.md": "zephyr launch notes",
        "stuffed.md": " ".join(["zephyr"] * 60 + FILLER.split() * 5),
        "other.md": "weather patterns today",
        "other2.md": "another unrelated page",
    }
    s = dict(rank(corpus, "zephyr zephyr zephyr", limit=10))
    assert s["short.md"] > s["stuffed.md"] / 3
