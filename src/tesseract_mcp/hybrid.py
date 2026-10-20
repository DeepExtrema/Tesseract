"""Hybrid retrieval: BM25 keyword ranking + vector similarity, fused via
Reciprocal Rank Fusion. RRF merges by rank position rather than raw score,
so BM25 scores and cosine similarities never need to be normalized against
each other.
"""

from __future__ import annotations

from pathlib import Path

from . import bm25 as bm25_mod
from .embeddings import Embedder, cosine, get_note_vectors
from .graphstore import GRAPH_ROOT
from .search import Hit, body_text, iter_candidate_notes, parse_frontmatter
from .vault import Vault

RRF_K = 60

# How many candidates each lane contributes to the fusion. Source-first
# reorders the fused list, so the lanes must reach deep enough to hold more
# than `limit` source notes: on a mature vault most notes are derived, and at
# the mixed-mode depth a query can fuse to 50 derived notes and no source note at all. Only
# source-first pays this, so mixed-mode ranking is unchanged.
LANE_DEPTH = 50
SOURCE_FIRST_LANE_DEPTH = 300


def is_derived(rel: str) -> bool:
    """True for a note the graph wrote about other notes, not a source.

    Entity notes under Claude/Graph are definitions extracted from the vault.
    On a mature vault they outnumber the notes they describe, so in a single
    ranking they crowd out the sources that are the actual answer to a query.
    """
    return rel.startswith(GRAPH_ROOT + "/")


def is_named_by(text: str, rel: str, query: str) -> bool:
    """True when the query is exactly this note's own name or one of its
    aliases, compared on BM25 tokens rather than raw text. Every character
    notes.safe_filename strips from an entity name (``/:*?"<>|[]#^``) is a
    character the tokenizer drops too, so "Project X: Phase 2" still matches
    the "Project X Phase 2.md" it was written to.

    Demoting every derived note cost entity-by-name lookup: "Alice Zhang"
    stopped returning the Alice Zhang entity note at all, because every
    source note that merely mentions her outranked her definition. A query
    that *is* the name is asking for the definition, so it is exempt.
    Equality, not containment: "who is Alice Zhang" is a question about her,
    and the notes that answer it are the sources.
    """
    q = bm25_mod.tokenize(query)
    if not q:
        return False
    stem = rel.rsplit("/", 1)[-1][:-3]
    if bm25_mod.tokenize(stem) == q:
        return True
    aliases = parse_frontmatter(text).get("aliases") or []
    if not isinstance(aliases, list):
        aliases = [aliases]
    return any(bm25_mod.tokenize(str(a)) == q for a in aliases)


def source_first(
    ranked: list[str], corpus: dict[str, str], query: str
) -> list[str]:
    """Stable partition of a ranked list: source notes, then derived ones.

    Relevance order is preserved inside each group, and a homogeneous list
    comes back untouched — so a search scoped to Claude/Graph, or a query only
    entity notes match, ranks exactly as before. A derived note the query
    names outright keeps its place among the sources.
    """
    def demote(rel: str) -> bool:
        return is_derived(rel) and not is_named_by(corpus[rel], rel, query)

    kept = [rel for rel in ranked if not demote(rel)]
    if len(kept) == len(ranked):
        return ranked
    return kept + [rel for rel in ranked if demote(rel)]


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _vector_rank(
    vectors: dict[str, list[float]], candidate_paths: set[str], query_vec: list[float], limit: int
) -> list[str]:
    scored = [
        (path, cosine(vec, query_vec))
        for path, vec in vectors.items()
        if path in candidate_paths
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [p for p, s in scored if s > 0][:limit]


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


def _substring_rank(corpus: dict[str, str], query: str, limit: int) -> list[str]:
    q = query.lower()
    ranked: list[str] = []
    for rel, text in sorted(corpus.items()):
        stem = rel.rsplit("/", 1)[-1][:-3]
        if q in stem.lower():
            ranked.append(rel)
        elif any(q in line.lower() for line in text.splitlines()):
            ranked.append(rel)
        if len(ranked) >= limit:
            break
    return ranked


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
    candidates = iter_candidate_notes(vault, tags, folder)
    if not candidates:
        return []
    corpus = dict(candidates)
    candidate_paths = set(corpus.keys())
    depth = SOURCE_FIRST_LANE_DEPTH if prefer_source else LANE_DEPTH

    bm25_ranked = [p for p, _ in bm25_mod.rank(corpus, query, limit=depth)]

    # An unfiltered candidate scan IS the full-vault {rel: text} map — hand
    # it to the vector layer so every note isn't read from disk twice.
    all_vectors = get_note_vectors(
        vault, state_root, embedder,
        note_texts=corpus if not (tags or folder) else None,
    )
    query_vec = embedder.embed_batch([query])[0]
    vector_ranked = _vector_rank(all_vectors, candidate_paths, query_vec, limit=depth)

    ranked_lists = [bm25_ranked, vector_ranked]
    if not bm25_ranked:
        # Fallback signal only: BM25 tokenizes [a-z0-9]+, so queries it cannot
        # token-match (e.g. single characters, punctuation-only) fall through
        # to substring matching. When BM25 has results, the alphabetically-
        # ordered substring list would just pollute the fusion.
        ranked_lists.append(_substring_rank(corpus, query, limit=depth))
    fused = rrf_fuse(ranked_lists)
    if prefer_source:
        fused = source_first(fused, corpus, query)
    fused = fused[:limit]
    return [Hit(rel, _excerpt(corpus[rel], rel, query)) for rel in fused]
