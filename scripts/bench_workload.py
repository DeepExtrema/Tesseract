"""Build a seeded benchmark workload from a vault, for bench_mcp.py.

The workload holds note titles and paths of the vault it was built from, so
write it outside the repository when the vault is private.

    python scripts/bench_workload.py VAULT OUT.json [--queries FILE]
        [--project NAME] [--tag TAG] [--seed N]

Search queries are note titles (what an agent types when it half-remembers a
note), graph entity names, and the free-form questions in --queries (one per
line): title queries always share words with a note, so add questions worded
the way you would ask them.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

GRAPH_PREFIX = "Claude/Graph/"


def stem(rel: str) -> str:
    return rel.rsplit("/", 1)[-1][:-3]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a seeded benchmark workload from a vault.")
    ap.add_argument("vault")
    ap.add_argument("out")
    ap.add_argument("--queries", help="file of free-form search questions, one per line")
    ap.add_argument("--project", help="a frontmatter project value, for query_notes and recall_bundle")
    ap.add_argument("--tag", help="a frontmatter tag, for query_notes")
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()

    root = Path(a.vault).resolve()
    rng = random.Random(a.seed)
    notes = sorted(
        "/".join(p.relative_to(root).parts)
        for p in root.rglob("*.md")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    source = [n for n in notes if not n.startswith(GRAPH_PREFIX)]
    entity = [n for n in notes if n.startswith(GRAPH_PREFIX)]
    if not source:
        raise SystemExit(f"no notes found under {root}")

    queries = [stem(n) for n in rng.sample(source, min(50, len(source)))]
    queries += [stem(n) for n in rng.sample(entity, min(20, len(entity)))]
    if a.queries:
        queries += [q.strip() for q in Path(a.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    rng.shuffle(queries)

    paths = rng.sample(source, min(30, len(source)))
    folders = Counter(n.rsplit("/", 1)[0] for n in source if "/" in n)
    busiest = folders.most_common(1)[0][0] if folders else None

    note_filters: list[dict] = [{}]
    if busiest:
        note_filters.append({"folder": busiest})
    if a.project:
        note_filters.append({"project": a.project})
    if a.tag:
        note_filters.append({"tags": [a.tag]})
    recall = [{"mode": "digest"}]
    if a.project:
        recall.append({"mode": "resume", "project": a.project})

    ops = {
        "search_brain": [{"query": q, "limit": 20} for q in queries],
        "context_bundle": [{"query": q, "limit": 10} for q in queries[:20]],
        "query_notes": note_filters * 4,
        "list_recent": [{"n": 10}] * 20,
        "list_tasks": [{}] * 10,
        "get_backlinks": [{"path": p} for p in paths[:20]],
        "recall_bundle": recall * 5,
        "read_note": [{"path": p} for p in rng.sample(notes, min(200, len(notes)))],
        "related_notes": [{"path": p, "hops": 2} for p in paths[:20]],
        "graph_stats": [{}] * 10,
    }
    if entity:
        ops["find_entity"] = [{"query": stem(p)} for p in rng.sample(entity, min(10, len(entity)))] * 2
    Path(a.out).write_text(
        json.dumps({"notes": len(notes), "source": len(source), "entity": len(entity), "ops": ops}, indent=1),
        encoding="utf-8",
    )
    print(f"{len(notes)} notes ({len(source)} source, {len(entity)} entity); "
          + ", ".join(f"{k} {len(v)}" for k, v in ops.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
