"""Hybrid retrieval: BM25L keyword score + vector similarity, combined as a
weighted sum. Every candidate note gets the keyword score and the cosine
similarity, each scaled to [0, 1] over the candidates, and
``KEYWORD_SHARE`` of the keyword plus the rest of the vector is its score.
Notes the graph wrote about other notes (``is_derived``) are weighted down;
a note whose name is the query goes first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import bm25 as bm25_mod
from . import corpus, embeddings
from .embeddings import Embedder
from .graphstore import GRAPH_ROOT
from .search import Hit, body_text
from .vault import Vault

# Share of a note's score that comes from the keyword lane, the rest from the
# vector lane.
KEYWORD_SHARE = 0.5

# Factor on the score of a derived note when sources are preferred. Entity
# notes outnumber the notes they describe on a mature vault, and being short
# and name-dense they would otherwise crowd the sources out of the top ten.
DERIVED_WEIGHT = 0.5


def is_derived(rel: str) -> bool:
    """True for a note the graph wrote about other notes, not a source.

    Entity notes under Claude/Graph are definitions extracted from the vault.
    On a mature vault they outnumber the notes they describe, so in a single
    ranking they crowd out the sources that are the actual answer to a query.
    """
    return rel.startswith(GRAPH_ROOT + "/")


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] between the smallest and largest value. A constant list
    has no spread to scale: all zeros if the value is 0, all ones otherwise,
    so a search scoped to one note, or to notes that tie, still returns what
    matches."""
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.full_like(x, 1.0 if hi > 0 else 0.0)


def _cosine(
    resolved: embeddings.NoteVectors,
    picked: np.ndarray,
    query_vec: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of each candidate note to the query, floored at 0. A
    note whose vector has another length than the query's (a different model)
    or a zero norm scores 0."""
    matrix, norms = resolved.matrix(len(query_vec))
    denom = norms[picked] * (float(query_vec @ query_vec) ** 0.5)
    # einsum, not matmul: a multithreaded BLAS gemv costs more in thread
    # hand-offs than the 1.5 MFLOPs it computes when the machine is busy.
    dots = np.einsum("ij,j->i", matrix, query_vec)[picked]
    cosine = np.divide(dots, denom, out=np.zeros(len(picked)), where=denom != 0)
    return np.maximum(cosine, 0.0)


def _frontmatter_lines(text: str, body: str) -> list[str]:
    """Frontmatter field lines, in order, without the --- fences."""
    if body == text:
        return []
    front = text[: len(text) - len(body)]
    return [ln.strip() for ln in front.splitlines() if ln.strip().strip("-")]


def _excerpt(text: str, rel: str, query: str) -> str:
    stem = rel.rsplit("/", 1)[-1][:-3]
    q = query.lower()
    if q in stem.lower():
        return "(title match)"
    body = body_text(text)
    for line in body.splitlines():
        if q in line.lower():
            return line.strip()
    # BM25 and the embeddings rank the whole file, frontmatter included, so a
    # note can be a hit on a field alone. Notes imported from a database tool
    # keep all their content in frontmatter and have an empty body; previewing
    # only the body returned "" for those, leaving no way to triage a hit
    # without a read_note per result. Fields can be long, so cap them.
    front_lines = _frontmatter_lines(text, body)
    for line in front_lines:
        if q in line.lower():
            return line[:200]
    # Semantic-only hit: no literal match anywhere. First body line beats
    # the old behavior of returning the raw file's first line ("---").
    for line in body.splitlines():
        if line.strip():
            return line.strip()[:120]
    return front_lines[0][:120] if front_lines else ""


def _substring_match(notes: list[corpus.Note], query: str) -> np.ndarray:
    """1.0 for each note whose file name or any line contains the query,
    case-insensitively, else 0.0."""
    q = query.lower()
    return np.fromiter(
        (
            q in note.rel.rsplit("/", 1)[-1][:-3].lower()
            or any(q in line.lower() for line in note.text.splitlines())
            for note in notes
        ),
        np.float64,
        len(notes),
    )


def _name_pins(
    snap: corpus.Snapshot,
    picked: np.ndarray,
    query: str,
    score: np.ndarray,
    derived: np.ndarray,
    prefer_source: bool,
) -> list[int]:
    """Candidate indexes of the notes whose own name or alias is the query,
    best score first, and source notes before derived ones when
    ``prefer_source``. The match is token equality, not containment: "who is
    Alice Zhang" is a question about her, and the notes that answer it are
    the sources."""
    q = tuple(bm25_mod.tokenize(query))
    pins: list[int] = []
    for position in snap.named.get(q, ()) if q else ():
        j = int(np.searchsorted(picked, position))
        if j < len(picked) and picked[j] == position:
            pins.append(j)
    pins.sort(key=lambda j: -score[j])
    if prefer_source:
        pins.sort(key=lambda j: derived[j])
    return pins


def hybrid_search(
    vault: Vault,
    state_root: str | Path,
    embedder: Embedder,
    query: str,
    tags: list[str] | None = None,
    folder: str | None = None,
    limit: int = 20,
    prefer_source: bool = True,
) -> list[Hit]:
    folder_parts = corpus.folder_parts(vault, folder)
    snap = corpus.refresh(vault)
    picked = snap.select(tags, folder_parts)
    if not len(picked):
        return []
    live = snap.live

    keyword = bm25_mod.scores(
        [live[i].counts for i in picked], snap.totals[picked], query
    )
    if keyword.max() > 0:
        keyword = _minmax(keyword)
    else:
        # BM25 tokenizes [a-z0-9]+, so a query it cannot token-match (a single
        # character, punctuation only) has no keyword score at all. Substring
        # matching stands in for it.
        keyword = _substring_match([live[i] for i in picked], query)

    resolved = embeddings.resolve_vectors(vault, state_root, embedder, snap)
    query_vec = np.asarray(embedder.embed_batch([query])[0], dtype=np.float64)
    vector = _minmax(_cosine(resolved, picked, query_vec))

    score = KEYWORD_SHARE * keyword + (1 - KEYWORD_SHARE) * vector
    derived = np.fromiter(
        (is_derived(live[i].rel) for i in picked), np.bool_, len(picked)
    )
    if prefer_source:
        score = np.where(derived, score * DERIVED_WEIGHT, score)

    positive = np.flatnonzero(score > 0)
    order = positive[np.argsort(-score[positive], kind="stable")].tolist()
    pins = _name_pins(snap, picked, query, score, derived, prefer_source)
    pinned = set(pins)
    ranked = pins + [j for j in order if j not in pinned]
    return [
        Hit(live[picked[j]].rel, _excerpt(live[picked[j]].text, live[picked[j]].rel, query))
        for j in ranked[:limit]
    ]
