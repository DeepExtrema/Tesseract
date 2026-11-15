# Search evals

Benchmark assets for `python -m tesseract_mcp.evals`. Everything here is
**synthetic** — this repo is public, so real vault content never goes in.

- `vault/` — a ~20-note fixture mini-vault shaped like a real one
  (Projects/Areas/Notes/Inbox plus Claude/Sessions, Concepts, Graph).
- `golden.yaml` — 17 golden queries covering every retrieval lane:
  BM25 keyword, vector paraphrase, title match, tag/folder filters, the
  substring fallback (untokenizable query), granularity traps, entity
  notes, and a deliberate embedder-ceiling sentinel (zero-overlap
  paraphrase expected to strain the embedder).

## Query schema

```yaml
- id: unique-slug
  query: the search text
  expect: [Path/Must Find.md]      # recall is computed on these
  accept: [Path/Also Fine.md]      # relevant too, never punished
  tags: [optional]                 # forwarded to hybrid_search
  folder: optional/subfolder       # forwarded to hybrid_search
  note: why this query exists
  kind: optional-label             # groups the per-kind table
```

Adding a query: cover a behavior, not a note. If it documents a known
weakness (like the granularity traps), say so in `note` — those rows are
the before/after scoreboard for ranking changes.

Metrics: success@k / recall@k (k = 1, 5, 10) and MRR at retrieval depth 20.
Each success rate prints a 95% Wilson interval and MRR a 95% bootstrap
interval over questions (fixed seed, so a rerun prints the same numbers).
With an optional `kind:` on any row, the report ends with success@1 and
success@5 per kind (`--json` carries the same).
Thresholds live in `tests/test_evals.py` (env-guarded by
`TESSERACT_RUN_EVALS=1`). Rule: if the baseline sits below a floor, fix
the fixture or the golden set — never lower the floor to pass.

## Baseline

(recorded by Task 7 after the first real run)

| date | git | success@1 | success@5 | success@10 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|---|
| 2026-07-10 | 55ce90e | - | 0.875 | 0.9375 | 0.875 | 0.9375 | 0.889 |
| 2026-07-11 | 1ba005b | - | 0.94 | 1.00 | 0.94 | 1.00 | 0.86 |
| 2026-11-15 | e2f6ff7 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 0.87 |

2026-07-11 row: salvaged the `Quick capture dentist.md` fixture fix (a
second line closes the para-dentist paraphrase miss; it now ranks 5th)
from a stale worktree, and added `para-ceiling-espresso` as a new
embedder-ceiling sentinel (17 queries total, up from 16). Despite the
harder query, success@10/recall@10 hit 1.00 honestly — bge-micro-v2
still placed the target 3rd behind two thematically-confusable
distractors (Sourdough Starter, Long Ramble), which is real signal that
the ranking is close, not a miss. MRR dropped slightly (0.889 -> 0.86)
because that extra hard query pulls the mean down even when it succeeds
at rank 3.

2026-11-15 row: after the weighted-sum ranking (keyword levels off, file
name counts three extra times, normalised scores added, entity notes
weighted, an exact name first). success@1 is new in the report; its 95%
interval over 17 questions is 0.59-0.94, so a difference of a question or
two is noise. MRR interval 0.74-1.00.
