"""Shared note-parsing and vault-scanning helpers (frontmatter, note walks)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from .vault import Vault

# Named only for documentation and tests; the rule below is the whole class,
# not this list.
KNOWN_SKIP_DIRS = {".obsidian", ".trash", ".git", ".smart-env"}


def is_skipped(rel_parts: Sequence[str]) -> bool:
    """True for a vault-relative path under, or named as, a dot-entry.

    Dot-entries in a vault are tool state or stashed copies, never content.
    An allow-list of three known names let anything else that writes a
    dot-directory leak into search, embeddings, tasks, the graph and the mover
    at once, so the whole class is hidden instead of named cases.
    """
    return any(part.startswith(".") for part in rel_parts)


@dataclass
class Hit:
    path: str
    excerpt: str


def parse_frontmatter(text: str) -> dict:
    """Parse the leading YAML frontmatter block; {} on any failure."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        meta = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    return meta if isinstance(meta, dict) else {}


def is_superseded(text: str) -> bool:
    """True for a note whose frontmatter declares it superseded: a retired
    entity tombstone (``retired``) or a merge stub (``merged_into``).

    Retirement is the graph's answer to an entity with no remaining source.
    The tombstone deliberately keeps the aliases on disk so graphstore can
    revive the entity if extraction sees it again — but a tombstone is not an
    answer to a query. Left in the corpus it still ranks, and BM25 scores it
    *above* the note it was derived from, because the tombstone is shorter.
    Retiring without this check moves nothing out of search.
    """
    meta = parse_frontmatter(text)
    return bool(meta.get("retired") or meta.get("merged_into"))


def body_text(text: str) -> str:
    """Note content with the leading YAML frontmatter block removed."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:]
    return text


def as_str_list(value) -> list[str]:
    """Normalize a frontmatter value to a list of strings: None -> [],
    a bare scalar -> a 1-item list."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(v) for v in value]


def iter_note_files(
    vault: Vault, folder: str | None = None
) -> Iterator[tuple[Path, str]]:
    """(absolute path, vault-relative path) for every .md note under
    `folder` (default: the whole vault), skipping dot-entries (is_skipped)."""
    base = vault.resolve(folder) if folder else vault.root
    for path in sorted(base.rglob("*.md")):
        rel_parts = path.relative_to(vault.root).parts
        if is_skipped(rel_parts):
            continue
        yield path, "/".join(rel_parts)


def iter_candidate_notes(
    vault: Vault, tags: list[str] | None = None, folder: str | None = None
) -> list[tuple[str, str]]:
    """(rel_path, text) for every note passing the tag/folder filters."""
    wanted = {t.casefold() for t in tags} if tags else None
    out: list[tuple[str, str]] = []
    for path, rel in iter_note_files(vault, folder):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if is_superseded(text):
            continue
        if wanted and not wanted <= {
            t.casefold()
            for t in as_str_list(parse_frontmatter(text).get("tags"))
        }:
            continue
        out.append((rel, text))
    return out
