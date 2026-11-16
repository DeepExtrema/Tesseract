"""Reading a note's YAML frontmatter and body.

Lives below both ``search`` and ``corpus``: the note index parses each note once
with these helpers, and ``search`` re-exports them for its many importers.
"""

from __future__ import annotations

import yaml


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


def meta_is_superseded(meta: dict) -> bool:
    """True when parsed frontmatter declares the note superseded: a retired
    entity tombstone (``retired``) or a merge stub (``merged_into``)."""
    return bool(meta.get("retired") or meta.get("merged_into"))


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
    return meta_is_superseded(parse_frontmatter(text))


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


def frontmatter_tags(meta: dict) -> list[str]:
    return as_str_list(meta.get("tags"))
