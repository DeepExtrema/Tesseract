"""BM25L keyword scoring over per-note term counts.

k1 1.5, b 0.75, delta 0.5. BM25L, not BM25Okapi: Okapi's Robertson IDF yields
zero scores on the small corpora of a filtered search and of the test
fixtures. A term's contribution to a note that contains it is
``idf * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta)`` with ``ctd`` the term
frequency over the note's relative length, so it levels off as the word
repeats; a note without the term gets nothing from it.
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


def scores(
    counts: Sequence[Mapping[str, int]],
    totals: np.ndarray,
    query: str,
) -> np.ndarray:
    """Keyword score of every document for the query, in document order; 0
    for a document that shares no token with it.

    ``counts[i]`` maps each token of document i to its occurrences and
    ``totals[i]`` is the document's token count (float64).
    """
    n = len(counts)
    score = np.zeros(n)
    if not n:
        return score
    total_tokens = float(totals.sum())
    if total_tokens == 0:
        return score
    avgdl = total_tokens / n
    norm = 1 - B + B * totals / avgdl
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
                contributions[token] = np.where(
                    freq > 0,
                    idf * (K1 + 1) * (ctd + DELTA) / (K1 + ctd + DELTA),
                    0.0,
                )
            else:
                contributions[token] = None
        contribution = contributions[token]
        if contribution is not None:
            score += contribution
    return score
