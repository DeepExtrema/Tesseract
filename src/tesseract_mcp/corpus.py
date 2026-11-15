"""Process-wide note index: every markdown note in a vault, read once.

A search must see the vault as it is on disk now, and a vault is thousands of
notes, so each refresh walks the tree with ``os.scandir`` (one ``stat`` per
file) and re-reads only the notes whose file stamp changed. Parsed frontmatter,
BM25 term counts and the content hash ride along with the text, so no search
ever re-parses an unchanged note.

The stamp is ``(mtime_ns, ctime_ns, size, inode)``. A file rewritten twice
inside the filesystem's timestamp granularity can keep the same stamp, so a
note whose mtime is within ``RACY_NS`` of the moment it was read is not
trusted: it is read again on the next refresh until the clock has moved on.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import bm25
from .frontmatter import frontmatter_tags, meta_is_superseded, parse_frontmatter
from .vault import Vault

RACY_NS = 2_000_000_000

# A note's file name counts this many extra times in its term counts, so a
# query that names a note scores it above notes that merely repeat the words.
TITLE_WEIGHT = 3


def file_stamp(st: os.stat_result) -> tuple[int, int, int, int]:
    return (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino)


def is_racy(st: os.stat_result, read_ns: int) -> bool:
    """True when the file was modified so close to ``read_ns`` that a later
    rewrite could keep an identical stamp."""
    return abs(read_ns - st.st_mtime_ns) < RACY_NS


@dataclass(frozen=True, slots=True, eq=False)
class Note:
    rel: str
    parts: tuple[str, ...]
    sort_key: tuple[str, ...]
    stamp: tuple[int, int, int, int]
    mtime: float
    racy: bool
    text: str
    meta: dict
    superseded: bool
    tags: frozenset[str]
    counts: Counter
    total: int
    names: tuple[tuple[str, ...], ...]
    sha256: str


@dataclass(frozen=True, eq=False)
class Snapshot:
    """The notes of one refresh. Immutable: searches read it without locks."""

    generation: int
    by_rel: dict[str, Note]
    live: tuple[Note, ...]
    position: dict[str, int]
    totals: np.ndarray
    named: dict[tuple[str, ...], tuple[int, ...]]

    def select(
        self, tags: list[str] | None, folder_parts: tuple[str, ...]
    ) -> np.ndarray:
        """Positions in ``live`` of the notes passing the tag and folder
        filters, in note order."""
        depth = len(folder_parts)
        want = {t.casefold() for t in tags} if tags else None
        if not depth and want is None:
            return np.arange(len(self.live))
        picked = [
            i
            for i, note in enumerate(self.live)
            if (not depth or (len(note.parts) > depth and note.parts[:depth] == folder_parts))
            and (want is None or want <= note.tags)
        ]
        return np.array(picked, dtype=np.intp)


def name_tokens(rel: str, meta: dict) -> tuple[tuple[str, ...], ...]:
    """The distinct names of a note as token tuples: its file name and its
    ``aliases`` frontmatter. Every character notes.safe_filename strips from
    an entity name is a character the tokenizer drops too, so
    "Project X: Phase 2" is the name of the "Project X Phase 2.md" it was
    written to."""
    names = [rel.rsplit("/", 1)[-1][:-3]]
    aliases = meta.get("aliases") or []
    names.extend(str(a) for a in (aliases if isinstance(aliases, list) else [aliases]))
    return tuple(dict.fromkeys(t for t in map(tuple, map(bm25.tokenize, names)) if t))


def _sort_key(parts: tuple[str, ...]) -> tuple[str, ...]:
    # Path ordering compares components (case-folded where the platform folds
    # case), not the joined string: "Work/y.md" sorts before "Work Log/x.md".
    return tuple(os.path.normcase(p) for p in parts)


def _build_note(
    rel: str, path: str, st: os.stat_result, read_ns: int, prev: Note | None
) -> Note:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    stamp = file_stamp(st)
    racy = is_racy(st, read_ns)
    if prev is not None and prev.text == text:
        return replace(prev, stamp=stamp, mtime=st.st_mtime, racy=racy)
    meta = parse_frontmatter(text)
    tokens = bm25.tokenize(text)
    counts = Counter(tokens)
    title = bm25.tokenize(rel.rsplit("/", 1)[-1][:-3])
    for token in title:
        counts[token] += TITLE_WEIGHT
    parts = tuple(rel.split("/"))
    return Note(
        rel=rel,
        parts=parts,
        sort_key=_sort_key(parts),
        stamp=stamp,
        mtime=st.st_mtime,
        racy=racy,
        text=text,
        meta=meta,
        superseded=meta_is_superseded(meta),
        tags=frozenset(t.casefold() for t in frontmatter_tags(meta)),
        counts=counts,
        total=len(tokens) + TITLE_WEIGHT * len(title),
        names=name_tokens(rel, meta),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


_FOLDS_CASE = os.path.normcase("A") != "A"


def _scan(directory: str, prefix: str, found: list) -> None:
    """Append (rel, path, stat) for every markdown note below ``directory``.

    Dot-entries are never content (see ``search.is_skipped``) and symlinked
    folders are not descended into, as ``Path.rglob`` does not either.
    """
    try:
        scan = os.scandir(directory)
    except PermissionError:
        return
    with scan as entries:
        for entry in entries:
            name = entry.name
            if name[0] == ".":
                continue
            if entry.is_dir(follow_symlinks=False):
                _scan(entry.path, f"{prefix}{name}/", found)
            elif (name.lower() if _FOLDS_CASE else name).endswith(".md") and entry.is_file():
                try:
                    st = entry.stat()
                except FileNotFoundError:
                    continue
                found.append((f"{prefix}{name}", entry.path, st))


class NoteIndex:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._notes: dict[str, Note] = {}
        self._snapshot: Snapshot | None = None
        self._generation = 0

    def refresh(self) -> Snapshot:
        with self._lock:
            read_ns = time.time_ns()
            seen: dict[str, Note] = {}
            changed = self._snapshot is None
            found: list = []
            _scan(str(self.root), "", found)
            for rel, path, st in found:
                prev = self._notes.get(rel)
                if prev is not None and not prev.racy and prev.stamp == file_stamp(st):
                    seen[rel] = prev
                    continue
                try:
                    note = _build_note(rel, path, st, read_ns, prev)
                except FileNotFoundError:
                    continue
                seen[rel] = note
                if prev is None or note.stamp != prev.stamp or note.text != prev.text:
                    changed = True
            if seen.keys() != self._notes.keys():
                changed = True
            self._notes = seen
            if changed or self._snapshot is None:
                self._generation += 1
                self._snapshot = self._build_snapshot()
            return self._snapshot  # type: ignore[return-value]

    def _build_snapshot(self) -> Snapshot:
        live = tuple(
            sorted(
                (n for n in self._notes.values() if not n.superseded),
                key=lambda n: n.sort_key,
            )
        )
        named: dict[tuple[str, ...], list[int]] = {}
        for i, note in enumerate(live):
            for name in note.names:
                named.setdefault(name, []).append(i)
        return Snapshot(
            generation=self._generation,
            by_rel=self._notes,
            live=live,
            position={n.rel: i for i, n in enumerate(live)},
            totals=np.array([n.total for n in live], dtype=np.float64),
            named={name: tuple(ix) for name, ix in named.items()},
        )


_registry_lock = threading.Lock()
_indexes: dict[Path, NoteIndex] = {}


def index_for(vault: Vault) -> NoteIndex:
    with _registry_lock:
        index = _indexes.get(vault.root)
        if index is None:
            index = _indexes[vault.root] = NoteIndex(vault.root)
        return index


def refresh(vault: Vault) -> Snapshot:
    """Bring the vault's index up to date with the disk and return it."""
    return index_for(vault).refresh()


def folder_parts(vault: Vault, folder: str | None) -> tuple[str, ...]:
    """Vault-relative path components of a folder filter; raises VaultError
    when the folder escapes the vault."""
    base = vault.resolve(folder) if folder else vault.root
    return base.relative_to(vault.root).parts


def reset() -> None:
    """Forget every index. Tests use this to stay isolated from each other."""
    with _registry_lock:
        _indexes.clear()
