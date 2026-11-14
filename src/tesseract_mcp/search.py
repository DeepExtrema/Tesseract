"""Shared note-parsing and vault-scanning helpers (frontmatter, note walks)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import corpus
from .frontmatter import (  # noqa: F401
    as_str_list,
    body_text,
    is_superseded,
    parse_frontmatter,
)
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
    folder_parts = corpus.folder_parts(vault, folder)
    snap = corpus.refresh(vault)
    return [
        (snap.live[i].rel, snap.live[i].text) for i in snap.select(tags, folder_parts)
    ]
