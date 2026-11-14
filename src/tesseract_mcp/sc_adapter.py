"""Reads Smart Connections' local embeddings directly from disk.

.smart-env/multi/*.ajson is Smart Connections' own "append-only JSON"
format: one `"key": {...},` fragment per line, with later lines for the
same key superseding earlier ones. Wrapping the stripped lines in braces
and parsing as one JSON object gives last-occurrence-wins for free, since
Python's dict construction from duplicate keys keeps the last value.

Only whole-note entries (`smart_sources:<path>`) are used — block-level
`smart_blocks:<path>#chunk` entries are Smart Connections' finer-grained
index and are out of scope for note-level ranking.

There is one .ajson file per note, thousands of them, so each file's parse is
memoised by its file stamp (see ``corpus``) and a refresh re-parses only the
files Smart Connections touched since the last one.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import Note, file_stamp, is_racy, refresh as refresh_notes
from .vault import Vault

SMART_ENV_DIR = ".smart-env"
MODEL_KEY = "TaylorAI/bge-micro-v2"
_SOURCE_PREFIX = "smart_sources:"

# In-process memo of parsed .ajson files keyed by (mtime_ns, size): Smart
# Connections' files are re-read on every search in a long-running server
# but only change when Obsidian re-embeds something.
_parsed_ajson: dict[str, tuple[tuple[int, int], dict]] = {}


def _parse_ajson_file(path: Path) -> dict[str, dict]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    stamp = (stat.st_mtime_ns, stat.st_size)
    memo = _parsed_ajson.get(str(path))
    if memo and memo[0] == stamp:
        return memo[1]
    lines = [
        line.strip().rstrip(",")
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    blob = "{" + ",".join(lines) + "}" if lines else "{}"
    try:
        entries = json.loads(blob)
    except json.JSONDecodeError:
        entries = {}
    _parsed_ajson[str(path)] = (stamp, entries)
    return entries


@dataclass(frozen=True, slots=True)
class SourceVector:
    vec: np.ndarray
    embedded_at_ms: float

    def is_fresh_for(self, note: Note) -> bool:
        """Smart Connections embedded the note after its last edit."""
        return self.embedded_at_ms >= note.mtime * 1000


@dataclass(frozen=True, slots=True)
class _ParsedFile:
    stamp: tuple[int, int, int, int]
    racy: bool
    sources: tuple[tuple[str, SourceVector], ...]


def _parse_sources(path: Path, model_key: str) -> tuple[tuple[str, SourceVector], ...]:
    sources: list[tuple[str, SourceVector]] = []
    for key, entry in _parse_ajson_file(path).items():
        if not key.startswith(_SOURCE_PREFIX):
            continue
        note_path = entry.get("path")
        embeddings = entry.get("embeddings") or {}
        model_entry = embeddings.get(model_key)
        if not note_path or not model_entry or "vec" not in model_entry:
            continue
        embedded_at_ms = (entry.get("last_embed") or {}).get("at", 0)
        sources.append(
            (note_path, SourceVector(np.asarray(model_entry["vec"], dtype=np.float64), embedded_at_ms))
        )
    return tuple(sources)


class SmartEnv:
    """Every whole-note vector Smart Connections has stored for one vault."""

    def __init__(self, root: Path, model_key: str):
        self._root = root
        self._model_key = model_key
        self._lock = threading.Lock()
        self._files: dict[str, _ParsedFile] = {}
        self.generation = 0
        self.vectors: dict[str, SourceVector] = {}

    def refresh(self) -> "SmartEnv":
        """Re-parse the .ajson files whose stamp changed. ``generation``
        advances only when the stored vectors may have changed."""
        multi_dir = self._root / SMART_ENV_DIR / "multi"
        with self._lock:
            read_ns = time.time_ns()
            found: dict[str, _ParsedFile] = {}
            changed = False
            if multi_dir.is_dir():
                with os.scandir(multi_dir) as entries:
                    for entry in entries:
                        if not entry.name.endswith(".ajson") or not entry.is_file():
                            continue
                        st = entry.stat()
                        prev = self._files.get(entry.name)
                        if prev is not None and not prev.racy and prev.stamp == file_stamp(st):
                            found[entry.name] = prev
                            continue
                        found[entry.name] = _ParsedFile(
                            file_stamp(st),
                            is_racy(st, read_ns),
                            _parse_sources(Path(entry.path), self._model_key),
                        )
                        changed = True
            if found.keys() != self._files.keys():
                changed = True
            self._files = found
            if changed:
                self.generation += 1
                vectors: dict[str, SourceVector] = {}
                for name in sorted(found):
                    for note_path, source in found[name].sources:
                        vectors[note_path] = source
                self.vectors = vectors
            return self


_registry_lock = threading.Lock()
_envs: dict[tuple[Path, str], SmartEnv] = {}


def smart_env(vault: Vault, model_key: str = MODEL_KEY) -> SmartEnv:
    """The vault's Smart Connections vectors, refreshed from disk."""
    with _registry_lock:
        env = _envs.get((vault.root, model_key))
        if env is None:
            env = _envs[(vault.root, model_key)] = SmartEnv(vault.root, model_key)
    return env.refresh()


def reset() -> None:
    with _registry_lock:
        _envs.clear()


def load_note_vectors(vault: Vault, model_key: str = MODEL_KEY) -> dict[str, dict]:
    """{note path: {"vec": ndarray, "fresh": bool}} for every indexed note
    Smart Connections has embedded."""
    env = smart_env(vault, model_key)
    notes = refresh_notes(vault).by_rel
    return {
        path: {"vec": source.vec, "fresh": source.is_fresh_for(notes[path])}
        for path, source in env.vectors.items()
        if path in notes
    }
