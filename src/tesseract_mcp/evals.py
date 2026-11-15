"""Golden-query evaluation harness for hybrid search.

Scores the production retrieval path (hybrid.hybrid_search with real
embeddings) against golden query sets: a synthetic fixture corpus
committed under evals/, and optionally a private set stored in the live
vault at Claude/Evals.md. Metrics are rank-based (success@k, recall@k,
MRR) with threshold floors asserted in tests -- never exact-order
assertions, which punish semantic recall improvements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePath

import yaml

from . import hybrid, indexer
from .vault import Vault, VaultError

# Source-checkout layout (src/tesseract_mcp/evals.py -> repo root). The
# fixture paths only make sense in a checkout, not an installed wheel;
# this harness is a repo-local dev tool.
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_VAULT = REPO_ROOT / "evals" / "vault"
FIXTURE_GOLDEN = REPO_ROOT / "evals" / "golden.yaml"
LIVE_GOLDEN_REL = "Claude/Evals.md"
HISTORY_FILE = "eval_history.jsonl"
RETRIEVE_LIMIT = 20
KS = (1, 5, 10)
CONFIDENCE_Z = 1.96  # 95% interval
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260715  # fixed: the same golden set always prints the same interval
NO_KIND = "(none)"


class EvalConfigError(Exception):
    """Bad golden set, bad paths, or bad invocation -- exit code 2."""


@dataclass
class GoldenQuery:
    id: str
    query: str
    expect: list[str]
    accept: list[str] = field(default_factory=list)
    tags: list[str] | None = None
    folder: str | None = None
    note: str = ""
    kind: str | None = None


@dataclass
class QueryResult:
    id: str
    hits: list[str]
    first_rank: int | None
    recall_at: dict[int, float]
    success_at: dict[int, bool]
    missing: list[str]
    skipped: bool = False
    kind: str | None = None


@dataclass
class Scorecard:
    results: list[QueryResult]
    success_at: dict[int, float]
    recall_at: dict[int, float]
    mrr: float
    skipped: int
    success_ci: dict[int, tuple[float, float]] = field(default_factory=dict)
    mrr_ci: tuple[float, float] = (0.0, 0.0)
    by_kind: dict[str, dict] = field(default_factory=dict)


def first_relevant_rank(hits: list[str], relevant: set[str]) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h in relevant:
            return i
    return None


def recall_at_k(hits: list[str], expect: set[str], k: int) -> float:
    if not expect:
        return 0.0
    return len(set(hits[:k]) & expect) / len(expect)


def success_at_k(hits: list[str], relevant: set[str], k: int) -> bool:
    return any(h in relevant for h in hits[:k])


def wilson_interval(successes: int, n: int, z: float = CONFIDENCE_Z) -> tuple[float, float]:
    """Wilson score interval for a success rate over ``n`` questions."""
    if not n:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_interval(
    values: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean, resampling the questions."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return (means[int(0.025 * resamples)], means[math.ceil(0.975 * resamples) - 1])


_YAML_FENCE_RE = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)


def _as_str_list(value, field_name: str, entry: str) -> list[str]:
    """Normalize a YAML field to a list of strings. A bare scalar becomes a
    1-item list — iterating it directly would split a string into characters."""
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        return [str(x) for x in value]
    raise EvalConfigError(
        f"golden {entry}: '{field_name}' must be a string or list, got {type(value).__name__}"
    )


def load_golden(path: str | Path) -> list[GoldenQuery]:
    p = Path(path)
    if not p.is_file():
        raise EvalConfigError(f"golden file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".md":
        m = _YAML_FENCE_RE.search(text)
        if not m:
            raise EvalConfigError(f"no ```yaml block found in {p}")
        text = m.group(1)
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise EvalConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(raw, list):
        raise EvalConfigError(f"golden set must be a YAML list: {p}")
    queries: list[GoldenQuery] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "id" not in item or "query" not in item:
            raise EvalConfigError(f"entry {i} in {p} needs 'id' and 'query'")
        if not str(item["id"]).strip() or not str(item["query"]).strip():
            raise EvalConfigError(
                f"entry {i} in {p}: 'id' and 'query' must be non-empty strings"
            )
        queries.append(
            GoldenQuery(
                id=str(item["id"]),
                query=str(item["query"]),
                expect=_as_str_list(item.get("expect"), "expect", str(item["id"])),
                accept=_as_str_list(item.get("accept"), "accept", str(item["id"])),
                tags=_as_str_list(item["tags"], "tags", str(item["id"])) if item.get("tags") else None,
                folder=str(item["folder"]) if item.get("folder") else None,
                note=str(item.get("note") or ""),
                kind=str(item["kind"]) if item.get("kind") else None,
            )
        )
    seen: set[str] = set()
    for q in queries:
        if q.id in seen:
            raise EvalConfigError(f"duplicate golden id: {q.id}")
        seen.add(q.id)
        if not q.expect:
            raise EvalConfigError(f"golden {q.id}: 'expect' must be non-empty")
    return queries


def _escapes_vault(rel: str) -> bool:
    """Lexical containment check: absolute paths and ..-traversal never get
    probed on disk — golden paths are vault-relative by contract."""
    pp = PurePath(rel)
    return pp.is_absolute() or bool(pp.drive) or ".." in pp.parts


def validate_paths(
    queries: list[GoldenQuery], vault_root: str | Path, strict: bool
) -> dict[str, list[str]]:
    """Map of query id -> expect/accept paths missing from the vault."""
    root = Path(vault_root)
    missing: dict[str, list[str]] = {}
    for q in queries:
        gone = [
            p for p in q.expect + q.accept
            if _escapes_vault(p) or not (root / p).is_file()
        ]
        if gone:
            missing[q.id] = gone
    if strict and missing:
        detail = "; ".join(f"{qid}: {', '.join(ps)}" for qid, ps in missing.items())
        raise EvalConfigError(f"golden paths missing from vault: {detail}")
    return missing


def run_evals(
    vault: Vault,
    state_root: str | Path,
    embedder,
    queries: list[GoldenQuery],
    ks: tuple[int, ...] = KS,
    limit: int = RETRIEVE_LIMIT,
    lenient: bool = False,
) -> Scorecard:
    # A fresh state dir must not crash the first fallback-cache save
    # (the vector store writes without mkdir).
    Path(state_root).mkdir(parents=True, exist_ok=True)
    missing = validate_paths(queries, vault.root, strict=not lenient)
    results: list[QueryResult] = []
    for q in queries:
        gone = missing.get(q.id, [])
        if lenient and set(gone) >= set(q.expect):
            results.append(
                QueryResult(q.id, [], None, {k: 0.0 for k in ks},
                            {k: False for k in ks}, gone, skipped=True,
                            kind=q.kind)
            )
            continue
        hits = hybrid.hybrid_search(
            vault, state_root, embedder, q.query,
            tags=q.tags, folder=q.folder, limit=limit,
        )
        paths = [h.path for h in hits]
        expect = set(q.expect) - set(gone)
        relevant = (set(q.expect) | set(q.accept)) - set(gone)
        results.append(
            QueryResult(
                q.id, paths,
                first_relevant_rank(paths, relevant),
                {k: recall_at_k(paths, expect, k) for k in ks},
                {k: success_at_k(paths, relevant, k) for k in ks},
                gone,
                kind=q.kind,
            )
        )
    scored = [r for r in results if not r.skipped]
    n = len(scored) or 1
    reciprocal = [1.0 / r.first_rank if r.first_rank else 0.0 for r in scored]
    by_kind: dict[str, dict] = {}
    if any(r.kind for r in results):
        for kind in sorted({r.kind or NO_KIND for r in scored}):
            rows = [r for r in scored if (r.kind or NO_KIND) == kind]
            by_kind[kind] = {
                "n": len(rows),
                **{f"success_at_{k}": sum(r.success_at[k] for r in rows) / len(rows)
                   for k in ks},
            }
    return Scorecard(
        results=results,
        success_at={k: sum(r.success_at[k] for r in scored) / n for k in ks},
        recall_at={k: sum(r.recall_at[k] for r in scored) / n for k in ks},
        mrr=sum(reciprocal) / n,
        skipped=len(results) - len(scored),
        success_ci={
            k: wilson_interval(sum(r.success_at[k] for r in scored), len(scored))
            for k in ks
        },
        mrr_ci=bootstrap_interval(reciprocal),
        by_kind=by_kind,
    )


def format_table(sc: Scorecard) -> str:
    # Derive k values from the scorecard so custom ks never KeyError.
    kmax = max(sc.recall_at)
    lines = [f"{'id':<28} {'rank':>4} {'r@' + str(kmax):>5}  miss"]
    for r in sc.results:
        if r.skipped:
            lines.append(f"{r.id:<28} {'skip':>4} {'-':>5}  {', '.join(r.missing)}")
            continue
        rank = str(r.first_rank) if r.first_rank else "-"
        lines.append(
            f"{r.id:<28} {rank:>4} {r.recall_at[kmax]:>5.2f}  {', '.join(r.missing)}"
        )
    scored = len(sc.results) - sc.skipped
    lines.append("-" * 60)
    lines.append(f"queries {scored}  skipped {sc.skipped}")
    for k in sorted(sc.success_at):
        lo, hi = sc.success_ci[k]
        lines.append(f"success@{k:<3} {sc.success_at[k]:.2f}  95% CI {lo:.2f}-{hi:.2f}")
    lo, hi = sc.mrr_ci
    lines.append(f"MRR        {sc.mrr:.2f}  95% CI {lo:.2f}-{hi:.2f}")
    lines.append(
        "recall     "
        + "  ".join(f"@{k} {sc.recall_at[k]:.2f}" for k in sorted(sc.recall_at))
    )
    if sc.by_kind:
        width = max(len(kind) for kind in sc.by_kind)
        ks = [k for k in (1, 5) if k in sc.success_at]
        lines.append("")
        lines.append(
            f"{'kind':<{width}} {'n':>4}" + "".join(f" {'success@' + str(k):>10}" for k in ks)
        )
        for kind, row in sc.by_kind.items():
            lines.append(
                f"{kind:<{width}} {row['n']:>4}"
                + "".join(f" {row[f'success_at_{k}']:>10.2f}" for k in ks)
            )
    return "\n".join(lines)


def to_json(sc: Scorecard) -> dict:
    kmax = max(sc.recall_at)
    d: dict = {f"success_at_{k}": v for k, v in sorted(sc.success_at.items())}
    d.update({f"success_at_{k}_ci": list(v) for k, v in sorted(sc.success_ci.items())})
    d.update({f"recall_at_{k}": v for k, v in sorted(sc.recall_at.items())})
    d["mrr"] = sc.mrr
    d["mrr_ci"] = list(sc.mrr_ci)
    d["by_kind"] = sc.by_kind
    d["skipped"] = sc.skipped
    d["n_queries"] = len(sc.results)
    d["queries"] = [
        {
            "id": r.id,
            "kind": r.kind,
            "first_rank": r.first_rank,
            "skipped": r.skipped,
            "recall": r.recall_at.get(kmax),
            "missing": r.missing,
        }
        for r in sc.results
    ]
    return d


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def append_history(
    state_root: str | Path, sc: Scorecard, vault_path, golden_path
) -> Path:
    state = Path(state_root)
    state.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _git_sha(),
        "vault": str(vault_path),
        "golden": str(golden_path),
        **to_json(sc),
    }
    p = state / HISTORY_FILE
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return p


def _make_embedder():
    from .embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder()


LIVE_TEMPLATE = '''# Search eval golden set (live vault)

Queries for `python -m tesseract_mcp.evals --live`. Paths are
vault-relative. Stale paths are skipped and reported here, never fatal:
this vault legitimately drifts. Add a query whenever a search annoys
you -- that is the whole curation strategy.

```yaml
- id: example-constitution
  query: what are the rules for agents writing to the vault
  expect:
    - Claude/README.md
  note: seed example -- replace or extend freely
```
'''


def init_live(vault: Vault) -> tuple[Path, bool]:
    target = Path(vault.root) / LIVE_GOLDEN_REL
    if target.exists():
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(LIVE_TEMPLATE, encoding="utf-8")
    return target, True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tesseract_mcp.evals",
        description="Golden-query evaluation for hybrid search.",
    )
    p.add_argument("--vault", help="vault root (default: fixture corpus)")
    p.add_argument("--golden", help="golden file (default: evals/golden.yaml)")
    p.add_argument("--live", action="store_true",
                   help="use TESSERACT_VAULT_PATH and Claude/Evals.md "
                        "(takes precedence over --vault/--golden)")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--init-live", action="store_true",
                   help="create Claude/Evals.md template if absent, then exit")
    args = p.parse_args(argv)
    try:
        if args.live or args.init_live:
            root = os.environ.get("TESSERACT_VAULT_PATH")
            if not root:
                raise EvalConfigError(
                    "--live/--init-live require TESSERACT_VAULT_PATH"
                )
            vault_path = Path(root)
            golden_path = vault_path / LIVE_GOLDEN_REL
            lenient = True
        else:
            vault_path = Path(args.vault) if args.vault else FIXTURE_VAULT
            golden_path = Path(args.golden) if args.golden else FIXTURE_GOLDEN
            lenient = False
        vault = Vault(vault_path)
        if args.init_live:
            target, created = init_live(vault)
            print(f"{'created' if created else 'already exists'}: {target}")
            return 0
        queries = load_golden(golden_path)
        sc = run_evals(
            vault, indexer.state_dir(vault.root), _make_embedder(),
            queries, lenient=lenient,
        )
    except (EvalConfigError, VaultError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(to_json(sc), indent=2) if args.as_json else format_table(sc))
    if not args.no_history:
        try:
            append_history(indexer.state_dir(vault.root), sc, vault_path, golden_path)
        except OSError as e:
            # History is optional persistence (--no-history exists); a failed
            # append must not turn a successful eval into a traceback.
            print(f"warning: could not write eval history: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
