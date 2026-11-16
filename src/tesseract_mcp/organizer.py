"""Autonomous vault organizer: files notes where their semantic neighbors
live, via cosine-weighted K-nearest-neighbor folder vote.

Standing permission for autonomous moves in the human topical tree was
granted by Taimoor 2026-07-08 (see the Organizer section of the
constitution). Rails: journal + undo, proposals queue below the confidence
threshold, hard exclusions below.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import cache, indexer, platforms
from . import embeddings as embeddings_mod
from . import sheets
from .embeddings import cosine
from .mover import (
    ProtectedLinkError,
    _move_file,
    _plan_rewrites,
    duplicate_stem_exists,
    move_note,
    rename_vector_cache_entry,
    reverse_rewrites,
)
from .search import parse_frontmatter
from .vault import Vault, VaultError, _retry_locked, _write_all, write_text_atomic

EXCLUDED_DIRS = frozenset({
    "Claude", "00 - Maps of Content", ".obsidian", ".smart-env",
    ".trash", ".space", "copilot",
})
# vault-root agent guides installed by conventions.install(); Claudian and
# Codex read these from the root — the organizer must never relocate them
EXCLUDED_ROOT_FILES = frozenset({"CLAUDE.md", "AGENTS.md", "README.md"})
VOTE_K = 10
VOTE_THRESHOLD = 0.7
ORGANIZER_NOTE = "Claude/Organizer.md"
_NOTE_SEED = (
    "# Organizer\n\nAutonomous move log and proposals. "
    "See constitution → Organizer.\n\n## Log\n"
)


def discover_taxonomy(vault: Vault) -> list[str]:
    """Existing top-level folders = the frozen taxonomy. All dot-directories
    are config/tooling (e.g. .claude slash commands), never topical."""
    return sorted(
        p.name for p in vault.root.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_DIRS and not p.name.startswith(".")
        and not sheets.is_sheet_folder(vault, p.name)
    )


def _in_sheet_folder(vault: Vault, rel: str) -> bool:
    parts = rel.split("/")
    if len(parts) < 2:
        return False
    return sheets.is_sheet_folder(vault, "/".join(parts[:-1]))


def sheet_rows_linking(vault: Vault, rel: str, target_rel: str) -> list[str]:
    """Sheet-row notes whose links a move of rel would rewrite."""
    return [r for _p, r, *_ in _plan_rewrites(vault, rel, target_rel)
            if _in_sheet_folder(vault, r)]


def _wants_organizing(vault: Vault, rel: str) -> bool:
    text = vault.read(rel)
    return parse_frontmatter(text).get("organize") is not False


def iter_organized(vault: Vault) -> list[str]:
    """Rel paths of .md notes currently inside taxonomy folders."""
    out: list[str] = []
    for folder in discover_taxonomy(vault):
        for p in sorted((vault.root / folder).rglob("*.md")):
            out.append("/".join(p.relative_to(vault.root).parts))
    return out


def iter_candidates(vault: Vault, organized: list[str] | None = None) -> list[str]:
    """Notes the organizer may classify: vault-root .md files plus
    already-organized notes (re-checkable), minus organize: false.
    Pass a precomputed iter_organized() result to skip the second walk."""
    if organized is None:
        organized = iter_organized(vault)
    root_notes = sorted(
        p.name for p in vault.root.iterdir()
        if p.is_file() and p.suffix == ".md" and p.name not in EXCLUDED_ROOT_FILES
    )
    return [
        rel for rel in root_notes + organized
        if _wants_organizing(vault, rel) and not _in_sheet_folder(vault, rel)
    ]


@dataclass
class Classification:
    folder: str | None
    share: float
    neighbors: list[str]


def classify(
    rel: str,
    vectors: dict[str, list[float]],
    labeled: list[str],
    k: int = VOTE_K,
) -> Classification:
    """Cosine-weighted K-nearest-neighbor vote among labeled notes.
    share = winning folder's similarity mass / total mass of the top K."""
    vec = vectors.get(rel)
    if vec is None:
        return Classification(None, 0.0, [])
    scored = [
        (other, cosine(vec, vectors[other]))
        for other in labeled
        if other != rel and other in vectors
    ]
    scored = [(p, s) for p, s in scored if s > 0]
    if not scored:
        return Classification(None, 0.0, [])
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:k]
    votes: dict[str, float] = {}
    for path, sim in top:
        votes[path.split("/")[0]] = votes.get(path.split("/")[0], 0.0) + sim
    winner = max(votes, key=votes.get)
    share = votes[winner] / sum(votes.values())
    return Classification(winner, share, [p for p, _ in top])


def journal_path(vault: Vault) -> Path:
    return indexer.state_dir(vault.root) / "organizer_journal.jsonl"


def _ensure_note(vault: Vault) -> None:
    try:
        vault.read(ORGANIZER_NOTE)
    except VaultError:
        vault.write(ORGANIZER_NOTE, _NOTE_SEED)


def _open_journal(vault: Vault) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | platforms.current.O_BINARY
    return _retry_locked(lambda: os.open(journal_path(vault), flags, 0o666))


def _append_journal(vault: Vault, entry: dict) -> None:
    """One raw write per entry. A failed write is truncated away (best
    effort); if a killed process left a torn last line anyway, the new entry
    starts on a fresh line and undo skips the torn one."""
    data = (json.dumps(entry) + "\n").encode("utf-8")
    jp = journal_path(vault)
    if jp.is_file() and jp.stat().st_size:
        with jp.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
    fd = _open_journal(vault)
    try:
        size = os.fstat(fd).st_size
        try:
            _write_all(fd, data)
        except BaseException:
            try:
                os.ftruncate(fd, size)
            except OSError:
                pass
            raise
    finally:
        os.close(fd)


def record_move(
    vault: Vault, record: dict, share: float, neighbors: list[str]
) -> None:
    entry = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from": record["from"],
        "to": record["to"],
        "share": share,
        "neighbors": neighbors,
        "rewrites": record["rewrites"],
        "undone": False,
    }
    try:
        _append_journal(vault, entry)
    except BaseException as e:
        # an unjournaled move can never be undone: put everything back.
        # Not covered: a kill between move_note and this append, or a
        # rollback that itself fails, leaves the move unjournaled.
        try:
            _reverse_move(vault, entry)
        except BaseException as rollback_error:
            e.add_note(f"rollback also failed: {rollback_error!r}")
        raise
    _ensure_note(vault)
    stem = record["to"].rsplit("/", 1)[-1][:-3]
    vault.append(
        ORGANIZER_NOTE,
        f"- {entry['ts']} — moved [[{record['to'][:-3]}|{stem}]] "
        f"from `{record['from']}` (share {share:.2f})\n",
    )


def _reverse_move(vault: Vault, entry: dict) -> None:
    _move_file(vault.resolve(entry["to"]), vault.resolve(entry["from"]))
    reverse_rewrites(
        vault, entry["from"], entry["to"], [r["path"] for r in entry["rewrites"]]
    )
    indexer.rename_manifest_entry(vault.root, entry["to"], entry["from"])
    rename_vector_cache_entry(vault, entry["to"], entry["from"])


def moves_today(vault: Vault) -> int:
    """Journaled moves dated today (local time), undone ones included."""
    jp = journal_path(vault)
    if not jp.exists():
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    count = 0
    for line in jp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn line, skipped as undo_move does
        if isinstance(entry, dict) and str(entry.get("ts", "")).startswith(today):
            count += 1
    return count


def undo_move(vault: Vault, note_rel: str) -> dict:
    jp = journal_path(vault)
    entries = []
    if jp.exists():
        for line in jp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn line from a killed append; never a valid entry
    target_idx = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i]["to"] == note_rel and not entries[i].get("undone"):
            target_idx = i
            break
    if target_idx is None:
        raise VaultError(f"No undoable move found for: {note_rel}")
    entry = entries[target_idx]
    src = vault.resolve(entry["to"])
    dst = vault.resolve(entry["from"])
    if not src.is_file():
        raise VaultError(f"Cannot undo: file no longer at {entry['to']}")
    if dst.exists():
        raise VaultError(f"Cannot undo: original location occupied: {entry['from']}")
    _reverse_move(vault, entry)  # rides out transient sync/AV locks
    entries[target_idx]["undone"] = True
    write_text_atomic(jp, "".join(json.dumps(e) + "\n" for e in entries))
    _ensure_note(vault)
    vault.append(
        ORGANIZER_NOTE,
        f"- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — UNDID move of "
        f"`{entry['to']}` back to `{entry['from']}`\n",
    )
    try:  # graph mentions follow the reversed link rewrite; best effort
        cache.rebuild(vault, indexer.db_path(vault.root))
    except (OSError, sqlite3.Error):
        pass  # the next sweep or index run rebuilds it
    return {"restored": entry["from"], "was": entry["to"]}


def run_sweep(
    vault: Vault, embedder=None, apply: bool = False, max_moves: int | None = None
) -> dict:
    """max_moves caps moves per sweep (None = unlimited); confident moves past
    the cap are skipped and re-evaluated next sweep."""
    if apply:  # fail before any move if the journal cannot be opened
        os.close(_open_journal(vault))
    if embedder is None:
        embedder = embeddings_mod.SentenceTransformerEmbedder()
    vectors = embeddings_mod.get_note_vectors(
        vault, indexer.state_dir(vault.root), embedder, persist=apply
    )
    taxonomy = set(discover_taxonomy(vault))
    labeled = iter_organized(vault)
    moved: list[dict] = []
    proposals: list[dict] = []
    skipped: list[dict] = []

    for rel in iter_candidates(vault, organized=labeled):
        current = rel.split("/")[0] if "/" in rel and rel.split("/")[0] in taxonomy else None
        cls = classify(rel, vectors, labeled)
        if cls.folder is None:
            skipped.append({"path": rel, "reason": "no vector or no labeled neighbors"})
            continue
        if current == cls.folder:
            skipped.append({"path": rel, "reason": "already correctly filed"})
            continue
        if cls.share < VOTE_THRESHOLD:
            if current is None:  # root notes queue for a human; filed notes rest
                proposals.append({
                    "path": rel, "suggested": cls.folder,
                    "share": round(cls.share, 3),
                    "neighbors": cls.neighbors[:3],
                    "reason": "low confidence",
                })
            else:
                skipped.append({"path": rel, "reason": "low-confidence disagreement"})
            continue
        if duplicate_stem_exists(vault, rel):
            proposals.append({
                "path": rel, "suggested": cls.folder,
                "share": round(cls.share, 3),
                "neighbors": cls.neighbors[:3],
                "reason": "duplicate stem — bare links would become ambiguous",
            })
            continue
        if max_moves is not None and len(moved) >= max_moves:
            skipped.append({"path": rel, "reason": "move cap reached"})
            continue
        stem_name = rel.rsplit("/", 1)[-1]
        target_rel = f"{cls.folder}/{stem_name}"
        if not apply:
            # dry-run must report what apply would do, protection included
            rows = sheet_rows_linking(vault, rel, target_rel)
            if rows:
                proposals.append({
                    "path": rel, "suggested": cls.folder,
                    "share": round(cls.share, 3),
                    "neighbors": cls.neighbors[:3],
                    "reason": "linked from sheet rows: " + ", ".join(rows[:3]),
                })
                continue
        if apply:
            try:
                # sheet rows change only through sheet_upsert (constitution →
                # Sheets; roadmap: sheets are invisible to the mover), so a
                # move that would rewrite a link inside one goes to a human
                record = move_note(vault, rel, target_rel,
                                   protect=lambda r: _in_sheet_folder(vault, r))
            except ProtectedLinkError as e:
                proposals.append({
                    "path": rel, "suggested": cls.folder,
                    "share": round(cls.share, 3),
                    "neighbors": cls.neighbors[:3],
                    "reason": "linked from sheet rows: " + ", ".join(e.paths[:3]),
                })
                continue
            record_move(vault, record, share=cls.share, neighbors=cls.neighbors[:3])
        moved.append({
            "from": rel, "to_folder": cls.folder,
            "share": round(cls.share, 3), "neighbors": cls.neighbors[:3],
        })

    cache_rebuilt = False
    if apply and moved:
        cache.rebuild(vault, indexer.db_path(vault.root))
        cache_rebuilt = True
    if apply and proposals:
        _ensure_note(vault)
        lines = [f"\n### Proposals {datetime.now().strftime('%Y-%m-%d')}\n"]
        for p in proposals:
            lines.append(
                f"- `{p['path']}` → **{p['suggested']}** "
                f"(share {p['share']}; {p['reason']})\n"
            )
        vault.append(ORGANIZER_NOTE, "".join(lines))
    return {
        "moved": moved, "proposals": proposals,
        "skipped": skipped, "cache_rebuilt": cache_rebuilt,
    }
