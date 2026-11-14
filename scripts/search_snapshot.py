"""Dump hybrid-search results for a benchmark workload, to compare two checkouts.

A change that must not alter ranking (a speed-up, a refactor) is proven by
dumping results before and after and comparing the files byte for byte: same
paths, same order, same excerpts. Run both sides against copies of the same
vault and state folder; the dump holds note paths and excerpts, so keep it
outside the repository when the vault is private.

    TESSERACT_VAULT_PATH=VAULT TESSERACT_STATE_DIR=STATE \
        PYTHONPATH=<checkout>/src python scripts/search_snapshot.py WORKLOAD.json OUT.json
    cmp before.json after.json

The workload comes from scripts/bench_workload.py.
"""

from __future__ import annotations

import json
import os
import sys
import time

from tesseract_mcp import hybrid, indexer
from tesseract_mcp.embeddings import SentenceTransformerEmbedder
from tesseract_mcp.vault import Vault


def cases(workload: dict) -> list[dict]:
    searches = workload["ops"]["search_brain"]
    out = [{"query": a["query"], "limit": a.get("limit", 20)} for a in searches]
    # mixed-mode ranking, the filters, and queries the keyword lane cannot
    # tokenize each take a different branch of the search
    out += [{"query": a["query"], "limit": 20, "prefer_source": False} for a in searches[:25]]
    probe = searches[0]["query"]
    for filters in workload["ops"].get("query_notes", []):
        scoped = {k: filters[k] for k in ("folder", "tags") if k in filters}
        case = {"query": probe, "limit": 20, **scoped}
        if scoped and case not in out:
            out.append(case)
    out += [{"query": q, "limit": 20} for q in ("%", "a", "")]
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    workload = json.load(open(sys.argv[1], encoding="utf-8"))
    vault = Vault(os.environ["TESSERACT_VAULT_PATH"])
    state = indexer.state_dir(vault.root)
    embedder = SentenceTransformerEmbedder()
    dump = []
    start = time.perf_counter()
    for case in cases(workload):
        try:
            hits = hybrid.hybrid_search(vault, state, embedder, **case)
            dump.append({"case": case, "hits": [[h.path, h.excerpt] for h in hits]})
        except Exception as e:  # recorded, so both sides must fail the same way
            dump.append({"case": case, "error": f"{type(e).__name__}: {e}"})
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=0)
    print(f"{len(dump)} cases in {time.perf_counter() - start:.1f}s -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
