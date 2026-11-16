"""In-memory BM25L keyword ranking over vault notes.

Rebuilt fresh per query from the current vault scan rather than persisted:
a personal vault (hundreds to low-thousands of notes) is cheap to re-tokenize
and re-rank on every call.

k1 1.5, b 0.75, delta 0.5. BM25L, not BM25Okapi: Okapi's Robertson IDF yields
zero scores on the small corpora typical of per-query vault scans and the
test fixtures. A term's contribution to a note that contains it is
``idf * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta)`` with ``ctd`` the term
frequency over the note's relative length, so it levels off as the word
repeats; a note without the term gets nothing from it. (rank_bm25's BM25L
multiplies by the query-term frequency and gives absent terms a positive
score, so it was not used.)
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75
DELTA = 0.5


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rank(corpus: dict[str, str], query: str, limit: int = 50) -> list[tuple[str, float]]:
    if not corpus:
        return []
    paths = list(corpus.keys())
    counts = [Counter(tokenize(corpus[p])) for p in paths]
    lengths = [sum(c.values()) for c in counts]
    n = len(paths)
    total = sum(lengths)
    if total == 0:
        return []
    avgdl = total / n
    norms = [1 - B + B * length / avgdl for length in lengths]
    scores = [0.0] * n
    for token in tokenize(query):
        df = sum(1 for c in counts if token in c)
        if not df:
            continue
        idf = math.log(n + 1) - math.log(df + 0.5)
        for i, c in enumerate(counts):
            freq = c.get(token, 0)
            if freq:
                ctd = freq / norms[i]
                scores[i] += idf * (K1 + 1) * (ctd + DELTA) / (K1 + ctd + DELTA)
    ranked = sorted(zip(paths, scores), key=lambda pair: pair[1], reverse=True)
    return [(p, s) for p, s in ranked if s > 0][:limit]
