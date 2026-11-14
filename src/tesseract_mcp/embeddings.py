"""Vector source for hybrid search: Smart Connections' embeddings where
fresh, a same-model local fallback (cached) where stale or missing.

The fallback MUST use the identical model Smart Connections uses
(TaylorAI/bge-micro-v2) — vectors from a different model live in an
unrelated space and would silently corrupt similarity ranking if mixed in.

The fallback store is one binary file, ``note_vectors.npz``, in the state
folder: note paths, content hashes and a float64 matrix. Loading is memoised by
file stamp, so another process's save is picked up and an unchanged store is
never re-read. A ``fallback_embeddings.json`` left by an older version is read
when the binary file is absent, converted by the next call that may write, and
left in place.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import corpus, sc_adapter
from .corpus import Note, Snapshot, file_stamp, is_racy
from .vault import Vault

STORE_FILE = "note_vectors.npz"
LEGACY_JSON_FILE = "fallback_embeddings.json"


class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_key: str = sc_adapter.MODEL_KEY):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_key)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# In-process memo of parsed vector-cache files, keyed by (mtime_ns, size).
# The files grow with the vault (megabytes of JSON) and are read on every
# search in a long-running server; the stat key re-parses only after an
# on-disk change, so out-of-process writers are still picked up.
_parsed_caches: dict[str, tuple[tuple[int, int], dict]] = {}


def load_vector_cache(path: Path) -> dict[str, dict]:
    """JSON vector cache keyed by path: {key: {"hash": ..., "vec": [...]}}.
    Returns a fresh top-level dict: callers mutate it before saving, and an
    unsaved mutation must not leak into the memo."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return {}
    stamp = (stat.st_mtime_ns, stat.st_size)
    memo = _parsed_caches.get(str(path))
    if memo and memo[0] == stamp:
        return dict(memo[1])
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # corrupt/truncated cache self-heals: treat as empty and rewrite
    _parsed_caches[str(path)] = (stamp, cache)
    return dict(cache)


def save_vector_cache(path: Path, cache: dict[str, dict]) -> None:
    # unique temp file + atomic replace: an interrupted write must never
    # leave a truncated cache, and the server, indexer and librarian may
    # save the same cache concurrently from separate processes.
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(cache))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    stat = path.stat()
    _parsed_caches[str(path)] = ((stat.st_mtime_ns, stat.st_size), dict(cache))


class VectorStore:
    """Fallback vectors keyed by note path, valid while the stored content hash
    matches the note's current hash."""

    def __init__(self, state_root: Path):
        self._root = state_root
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._racy = False
        self._legacy_stamp: tuple | None = None
        self.generation = 0
        self.rows: dict[str, tuple[str, np.ndarray]] = {}

    @property
    def _path(self) -> Path:
        return self._root / STORE_FILE

    def refresh(self, *, persist: bool = True) -> "VectorStore":
        """Bring the rows up to date with the files. With ``persist`` False
        nothing is written: an old JSON store is read and its conversion is
        left to the next call that may write."""
        with self._lock:
            self._refresh(persist)
            return self

    def _refresh(self, persist: bool) -> None:
        try:
            st = self._path.stat()
        except FileNotFoundError:
            try:
                legacy_st = (self._root / LEGACY_JSON_FILE).stat()
            except FileNotFoundError:
                if self._stamp is not None or self._legacy_stamp is not None:
                    self._stamp, self._legacy_stamp, self.rows = None, None, {}
                    self.generation += 1
                return
            self._read_legacy(legacy_st)
            if persist:
                self._write()
            return
        if self._stamp == file_stamp(st) and not self._racy:
            return
        read_ns = time.time_ns()
        with np.load(self._path, allow_pickle=False) as z:
            paths = z["paths"].tolist()
            hashes = z["hashes"].tolist()
            lens = z["lens"].tolist()
            matrix = z["matrix"]
        self.rows = {
            path: (digest, matrix[i, :n])
            for i, (path, digest, n) in enumerate(zip(paths, hashes, lens))
        }
        self._stamp, self._racy = file_stamp(st), is_racy(st, read_ns)
        self.generation += 1

    def _read_legacy(self, st: os.stat_result) -> None:
        if self._legacy_stamp == file_stamp(st):
            return
        cache = json.loads((self._root / LEGACY_JSON_FILE).read_text(encoding="utf-8"))
        self.rows = {
            rel: (entry["hash"], np.asarray(entry["vec"], dtype=np.float64))
            for rel, entry in cache.items()
        }
        self._legacy_stamp = file_stamp(st)
        self.generation += 1

    def save(self, entries: dict[str, tuple[str, np.ndarray]], existing: set[str]) -> None:
        """Add ``entries`` and drop every note that is no longer in the vault."""
        with self._lock:
            self._refresh(persist=False)
            self.rows.update(entries)
            self.rows = {rel: row for rel, row in self.rows.items() if rel in existing}
            self._write()

    def _write(self) -> None:
        paths = list(self.rows)
        lens = np.array([len(self.rows[p][1]) for p in paths], dtype=np.int64)
        matrix = np.zeros((len(paths), int(lens.max(initial=0))), dtype=np.float64)
        for i, p in enumerate(paths):
            matrix[i, : lens[i]] = self.rows[p][1]
        fd, tmp = tempfile.mkstemp(dir=self._root, prefix=".note_vectors-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    paths=np.array(paths, dtype=str),
                    hashes=np.array([self.rows[p][0] for p in paths], dtype=str),
                    lens=lens,
                    matrix=matrix,
                )
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        st = self._path.stat()
        self._stamp, self._racy = file_stamp(st), is_racy(st, time.time_ns())
        self.rows = {p: (self.rows[p][0], matrix[i, : lens[i]]) for i, p in enumerate(paths)}
        self.generation += 1


@dataclass(eq=False)
class NoteVectors:
    """The vector chosen for every live note, in note order.

    ``vectors[i]`` is None for a note with neither a fresh Smart Connections
    vector nor a stored vector for its current text (``stale``).
    """

    key: tuple[int, int, int]
    notes: tuple[Note, ...]
    vectors: list[np.ndarray | None]
    stale: list[int]
    _matrices: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def matrix(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """(matrix, row norms) over the vectors of length ``dim``; a note whose
        vector has another length (a different model) gets a zero row."""
        built = self._matrices.get(dim)
        if built is None:
            m = np.zeros((len(self.vectors), dim), dtype=np.float64)
            for i, v in enumerate(self.vectors):
                if v is not None and len(v) == dim:
                    m[i] = v
            built = self._matrices[dim] = (m, np.sqrt(np.einsum("ij,ij->i", m, m)))
        return built


class _Source:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current: NoteVectors | None = None


_registry_lock = threading.Lock()
_stores: dict[Path, VectorStore] = {}
_sources: dict[tuple[Path, Path, str], _Source] = {}


def _store_for(state_root: Path) -> VectorStore:
    with _registry_lock:
        key = Path(state_root)
        store = _stores.get(key)
        if store is None:
            store = _stores[key] = VectorStore(key)
        return store


def _source_for(vault: Vault, state_root: Path) -> _Source:
    with _registry_lock:
        key = (vault.root, Path(state_root), sc_adapter.MODEL_KEY)
        source = _sources.get(key)
        if source is None:
            source = _sources[key] = _Source()
        return source


def reset() -> None:
    """Forget every memoised store and assembly. Tests use this to stay
    isolated from each other, or to stand in for a freshly started process."""
    with _registry_lock:
        _stores.clear()
        _sources.clear()


def _choose(
    snap: Snapshot, env: sc_adapter.SmartEnv, store: VectorStore
) -> tuple[list[np.ndarray | None], list[int]]:
    vectors: list[np.ndarray | None] = []
    stale: list[int] = []
    for i, note in enumerate(snap.live):
        source = env.vectors.get(note.rel)
        if source is not None and source.is_fresh_for(note):
            vectors.append(source.vec)
            continue
        stored = store.rows.get(note.rel)
        if stored is not None and stored[0] == note.sha256:
            vectors.append(stored[1])
            continue
        vectors.append(None)
        stale.append(i)
    return vectors, stale


def resolve_vectors(
    vault: Vault,
    state_root: Path,
    embedder: Embedder | None,
    snap: Snapshot,
    persist: bool = True,
) -> NoteVectors:
    """Choose each live note's vector: a fresh Smart Connections vector, else
    the stored one whose hash matches the text, else embed now (one batch for
    every such note) and store it. With ``embedder`` None nothing is computed
    or saved and stale notes are only reported. With ``persist`` False (a
    dry-run) stale notes are embedded in memory and never stored."""
    source = _source_for(vault, state_root)
    with source.lock:
        env = sc_adapter.smart_env(vault)
        store = _store_for(state_root).refresh(persist=embedder is not None and persist)
        key = (snap.generation, env.generation, store.generation)
        current = source.current
        if current is None or current.key != key:
            vectors, stale = _choose(snap, env, store)
            current = NoteVectors(key, snap.live, vectors, stale)
        if current.stale and embedder is not None and not persist:
            embedded = embedder.embed_batch([snap.live[i].text for i in current.stale])
            vectors = list(current.vectors)
            for i, v in zip(current.stale, embedded):
                vectors[i] = np.asarray(v, dtype=np.float64)
            source.current = current
            return NoteVectors(key, snap.live, vectors, [])
        if current.stale and embedder is not None:
            notes = [snap.live[i] for i in current.stale]
            embedded = embedder.embed_batch([n.text for n in notes])
            store.save(
                {n.rel: (n.sha256, np.asarray(v, dtype=np.float64)) for n, v in zip(notes, embedded)},
                set(snap.by_rel),
            )
            key = (snap.generation, env.generation, store.generation)
            vectors, stale = _choose(snap, env, store)
            current = NoteVectors(key, snap.live, vectors, stale)
        source.current = current
        return current


def get_note_vectors(
    vault: Vault, state_root: Path, embedder: Embedder, persist: bool = True
) -> dict[str, list[float]]:
    """persist=False (dry-runs) embeds stale notes in memory but never
    writes the vector store."""
    resolved = resolve_vectors(
        vault, state_root, embedder, corpus.refresh(vault), persist=persist
    )
    return {
        note.rel: vec.tolist()
        for note, vec in zip(resolved.notes, resolved.vectors)
        if vec is not None
    }


def stale_notes(vault: Vault, state_root: Path) -> list[str]:
    """Rel paths of notes with no fresh Smart Connections vector AND no
    matching fallback-cache entry — the notes a search would embed inline.
    Read-only: never computes or caches anything."""
    resolved = resolve_vectors(vault, state_root, None, corpus.refresh(vault))
    return [resolved.notes[i].rel for i in resolved.stale]
