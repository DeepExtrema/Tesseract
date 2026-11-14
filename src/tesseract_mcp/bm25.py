"""BM25L keyword ranking over per-note term counts.

Scores are those of ``rank_bm25.BM25L`` (k1 1.5, b 0.75, delta 0.5), computed
with numpy in the same operation order so they match bit for bit. BM25L, not
BM25Okapi: Okapi's Robertson IDF yields zero scores on the small corpora of
a filtered search and of the test fixtures.

Known defect, kept on purpose: each query term's contribution is multiplied by
the raw count of the term in the note (as ``BM25L.get_scores`` does), so a
note's score grows with every repeat of a query word instead of levelling off.
Changing it changes every ranking; it needs to be measured against the evals.
"""

from __future__ import annotations

import math
import re
from itertools import repeat
from collections.abc import Mapping, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75
DELTA = 0.5


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rank(
    counts: Sequence[Mapping[str, int]],
    totals: np.ndarray,
    query: str,
    limit: int = 50,
) -> list[tuple[int, float]]:
    """(position, score) for the best ``limit`` documents scoring above zero.

    ``counts[i]`` maps each token of document i to its occurrences and
    ``totals[i]`` is the document's token count (float64). Equal scores keep
    document order.
    """
    n = len(counts)
    if not n:
        return []
    total_tokens = float(totals.sum())
    if total_tokens == 0:
        return []
    avgdl = total_tokens / n
    norm = 1 - B + B * totals / avgdl
    score = np.zeros(n)
    contributions: dict[str, np.ndarray | None] = {}
    for token in tokenize(query):
        if token not in contributions:
            freq = np.fromiter(
                map(dict.get, counts, repeat(token), repeat(0)), np.float64, n
            )
            df = int(np.count_nonzero(freq))
            if df:
                idf = math.log(n + 1) - math.log(df + 0.5)
                ctd = freq / norm
                contributions[token] = (
                    idf * freq * (K1 + 1) * (ctd + DELTA) / (K1 + ctd + DELTA)
                )
            else:
                contributions[token] = None
        contribution = contributions[token]
        if contribution is not None:
            score += contribution
    positive = np.flatnonzero(score > 0)
    order = positive[np.argsort(-score[positive], kind="stable")][:limit]
    return [(int(i), float(score[i])) for i in order]
