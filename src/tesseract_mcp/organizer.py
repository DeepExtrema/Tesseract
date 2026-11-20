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
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
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

PROPOSAL_STALE_DAYS = 30
_BLOCK_START = "<!-- tesseract:proposals:start -->"
_BLOCK_END = "<!-- tesseract:proposals:end -->"
_BLOCK_HINT = (
    "_Maintained by the sweep. Mark a box (any mark) to dismiss that proposal "
    "(this path to this folder only). To undo a dismissal, delete its line. "
    "Other text typed inside this block is not kept._"
)
_SECTIONS = (
    ("needs-resolution", "Needs resolution"),
    ("pending", "Pending"),
    ("dismissed", "Dismissed"),
)
_LINE_RE = re.compile(
    r"^- \[(.)\] `(.+?)` → \*\*(.+?)\*\* "
    r"\(share ([0-9.]+); (.*); proposed (\d{4}-\d{2}-\d{2})"
    r"(?:; stale since (\d{4}-\d{2}-\d{2}))?\)(?:\s.*)?$"  # Tasks may append " ✅ date"
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


# --- proposal lifecycle -------------------------------------------------
# One managed block in Claude/Organizer.md holds every open proposal; the
# note syncs, so any machine reads the same state. Identity = (path,
# suggested folder). Only an applying sweep writes it.


def render_proposal_block(entries: list[dict]) -> str:
    out = [_BLOCK_START, _BLOCK_HINT]
    for state, title in _SECTIONS:
        out.append(f"#### {title}")
        group = [e for e in entries if e["state"] == state]
        for e in group:
            box = "x" if state == "dismissed" else " "
            stale = f"; stale since {e['stale_since']}" if e["stale_since"] else ""
            out.append(
                f"- [{box}] `{e['path']}` → **{e['suggested']}** "
                f"(share {e['share']}; {e['reason']}; proposed {e['proposed']}{stale})"
            )
        if not group:
            out.append("_none_")
    out.append(_BLOCK_END)
    return "\n".join(out)


def parse_proposal_block(block: str) -> list[dict]:
    """Entries of a managed block. `ticked` marks any owner's mark ([x], [-],
    ...) on a line that is not yet dismissed. A checkbox line that cannot be
    read is an error: dropping it would lose a dismissal or reset its date."""
    entries: list[dict] = []
    state = None
    titles = {title: s for s, title in _SECTIONS}
    for line in block.splitlines():
        if line.startswith("#### "):
            state = titles.get(line[5:].strip())
            continue
        m = _LINE_RE.match(line)
        if not m:
            if state is not None and line.startswith("- ["):
                raise VaultError(
                    f"{ORGANIZER_NOTE}: cannot read proposal line {line!r}; "
                    "fix it by hand or delete the line")
            continue
        if state is None:
            continue
        box, path, suggested, share, reason, proposed, stale = m.groups()
        for d in (proposed, stale):
            try:
                if d:
                    date.fromisoformat(d)
            except ValueError:
                raise VaultError(
                    f"{ORGANIZER_NOTE}: bad date {d!r} in proposal line "
                    f"`{path}`; fix it by hand") from None
        entries.append({
            "path": path, "suggested": suggested, "share": float(share),
            "reason": reason, "proposed": proposed, "stale_since": stale,
            "state": state, "ticked": box != " " and state != "dismissed",
        })
    return entries


def _read_proposals(vault: Vault) -> tuple[str | None, list[dict]]:
    """(note text or None, stored entries). A damaged block is an error, not
    a silent reset: it fails before any move or write."""
    try:
        text = vault.read(ORGANIZER_NOTE)
    except VaultError:
        return None, []
    starts, ends = text.count(_BLOCK_START), text.count(_BLOCK_END)
    start, end = text.find(_BLOCK_START), text.find(_BLOCK_END)
    if starts != ends or starts > 1 or end < start:
        raise VaultError(
            f"{ORGANIZER_NOTE}: proposals block markers are damaged; fix by hand")
    if not starts:
        return text, []
    return text, parse_proposal_block(text[start:end + len(_BLOCK_END)])


def _age(proposed: str, today: date) -> int:
    return (today - date.fromisoformat(proposed)).days


def stale_proposals(vault: Vault, today: date | None = None) -> list[dict]:
    """Open proposals past PROPOSAL_STALE_DAYS the owner has not ticked,
    read from the synced block (works on any machine, writes nothing)."""
    today = today or date.today()
    _, entries = _read_proposals(vault)
    out = []
    for e in entries:
        age = _age(e["proposed"], today)
        if e["ticked"] or e["state"] == "dismissed":
            continue
        if e["state"] == "needs-resolution" or age >= PROPOSAL_STALE_DAYS:
            out.append({
                "path": e["path"], "suggested": e["suggested"],
                "proposed": e["proposed"], "stale_since": e["stale_since"],
                "age_days": age,
            })
    return out


def _advance(vault: Vault, stored: list[dict], proposed: list[dict],
             today: date, held: frozenset = frozenset(),
             ) -> tuple[list[dict], list[str]]:
    """Next block entries and log lines for one sweep's fresh proposals.
    `held` = (path, folder) pairs a move cap deferred: still proposed."""
    iso = today.isoformat()
    current = {(p["path"], p["suggested"]): p for p in proposed}
    out: list[dict] = []
    log: list[str] = []
    seen: set[tuple[str, str]] = set()

    def note(what: str, e: dict) -> None:
        log.append(f"- {iso} — proposal {what}: `{e['path']}` → **{e['suggested']}**")

    def leave(e: dict) -> None:
        gone = not (vault.root / e["path"]).is_file()
        filed = (vault.root / e["suggested"] / e["path"].rsplit("/", 1)[-1]).is_file()
        note("applied" if gone and filed else "withdrawn", e)

    for e in stored:
        e = dict(e)
        key = (e["path"], e["suggested"])
        seen.add(key)
        if e["ticked"]:
            e.update(state="dismissed", ticked=False)
            note("dismissed", e)
        elif e["state"] == "dismissed":
            if not (vault.root / e["path"]).is_file():
                leave(e)
                continue
        elif key not in current and key not in held:
            leave(e)
            continue
        elif e["state"] == "pending" and _age(e["proposed"], today) >= PROPOSAL_STALE_DAYS:
            e.update(state="needs-resolution", stale_since=iso)
            note("needs-resolution", e)
        out.append(e)
    for key, p in current.items():
        if key not in seen:
            out.append({
                "path": p["path"], "suggested": p["suggested"],
                "share": p["share"], "reason": p["reason"], "proposed": iso,
                "stale_since": None, "state": "pending", "ticked": False,
            })
    return out, log


def _store_proposals(vault: Vault, entries: list[dict], log: list[str]) -> None:
    """Rewrite the block (and append log lines) only when the text changes."""
    text, stored = _read_proposals(vault)
    if text is None:
        text = _NOTE_SEED
    block = render_proposal_block(entries)
    start = text.find(_BLOCK_START)
    if start >= 0:
        end = text.index(_BLOCK_END, start) + len(_BLOCK_END)
        new = text[:start] + block + text[end:]
    elif entries:
        new = text + ("" if text.endswith("\n") else "\n") + "\n" + block + "\n"
    else:
        new = text
    if log:
        new += ("" if new.endswith("\n") else "\n") + "\n".join(log) + "\n"
    if new != text:
        vault.write(ORGANIZER_NOTE, new, overwrite=True)


def run_sweep(
    vault: Vault, embedder=None, apply: bool = False,
    max_moves: int | None = None, today: date | None = None,
) -> dict:
    """max_moves caps moves per sweep (None = unlimited); confident moves past
    the cap are skipped and re-evaluated next sweep. Proposals follow the
    lifecycle above: an applying sweep updates the managed block in
    Claude/Organizer.md (only when its text changes); a dry-run reads it and
    writes nothing. `today` is injectable for tests."""
    today = today or date.today()
    _, stored = _read_proposals(vault)
    dismissed_before = {(e["path"], e["suggested"]) for e in stored
                        if e["state"] == "dismissed" or e["ticked"]}
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
        if (rel, cls.folder) in dismissed_before:
            # the owner said no to this folder: never move there, however
            # confident the vote later becomes
            skipped.append({"path": rel, "reason": "proposal dismissed"})
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
            skipped.append({"path": rel, "reason": "move cap reached",
                            "suggested": cls.folder})
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
    if apply:  # re-read: keep a box the owner ticked while this sweep ran
        _, stored = _read_proposals(vault)
    dismissed = {(e["path"], e["suggested"]) for e in stored
                 if e["state"] == "dismissed" or e["ticked"]}
    open_now = []
    for p in proposals:
        if (p["path"], p["suggested"]) in dismissed:
            skipped.append({"path": p["path"], "reason": "proposal dismissed"})
        else:
            open_now.append(p)
    proposals = open_now
    held = frozenset((s["path"], s["suggested"]) for s in skipped
                     if s["reason"] == "move cap reached")
    entries, log = _advance(vault, stored, proposals, today, held)
    by_key = {(e["path"], e["suggested"]): e for e in entries}
    for p in proposals:
        e = by_key[(p["path"], p["suggested"])]
        p["state"], p["proposed"] = e["state"], e["proposed"]
        p["age_days"] = _age(e["proposed"], today)
    if apply:
        _store_proposals(vault, entries, log)
    return {
        "moved": moved, "proposals": proposals,
        "skipped": skipped, "cache_rebuilt": cache_rebuilt,
    }
