# Changelog

## 0.2.0 (unreleased)

- Mentions of notes whose name contains `]` are no longer dropped from the graph, so `related_notes` finds them.
- Every dot-folder in a vault is hidden from search, embeddings, tasks, sheets, the graph and the mover.
- A search hit on a note with an empty body shows an excerpt from its frontmatter.
- Retired and merged entity notes no longer appear in search and are never embedded.
- Source notes rank above derived `Claude/Graph` entity notes in search; `search_brain` takes `prefer_source` (default true) to turn this off.
- `find_entity` takes a `limit` and returns the best matches first: exact, then prefix, then substring, then most mentions.
- Search answers from an in-process note index that re-reads only changed notes, scores with numpy BM25L, and keeps fallback vectors in `note_vectors.npz`; the `rank-bm25` dependency is removed.
- On a 4,000-note vault a search takes about 40 ms instead of about 3.4 s; results are unchanged.
- The read-only stale-vector check no longer converts the old JSON vector store.
- Added `scripts/bench_mcp.py`, `scripts/bench_workload.py` and `scripts/search_snapshot.py` for measuring search latency and comparing results between two checkouts.

## 0.1.0

First public version.
