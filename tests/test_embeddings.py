import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tesseract_mcp import embeddings
from tesseract_mcp.embeddings import get_note_vectors, stale_notes
from tesseract_mcp.hybrid import hybrid_search
from tesseract_mcp.sc_adapter import MODEL_KEY


class FakeEmbedder:
    """Deterministic stand-in — no model download in tests."""

    def __init__(self):
        self.calls = []

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]


def _write_sc_vector(vault_dir, note_rel, vec, fresh=True):
    note = vault_dir / note_rel
    env_dir = vault_dir / ".smart-env" / "multi"
    env_dir.mkdir(parents=True, exist_ok=True)
    offset = 3600 if fresh else -3600
    at_ms = int((note.stat().st_mtime + offset) * 1000)
    entry = {
        "path": note_rel,
        "last_embed": {"at": at_ms},
        "embeddings": {MODEL_KEY: {"vec": vec}},
    }
    (env_dir / f"{note_rel.replace('/', '_')}.ajson").write_text(
        f'"smart_sources:{note_rel}": {json.dumps(entry)},', encoding="utf-8"
    )


def test_uses_smart_connections_vector_when_fresh(vault, vault_dir):
    _write_sc_vector(vault_dir, "Daily.md", [1.0, 2.0], fresh=True)
    for path in sorted(vault_dir.rglob("*.md")):
        rel = "/".join(path.relative_to(vault_dir).parts)
        if rel != "Daily.md":
            _write_sc_vector(vault_dir, rel, [0.0, 0.0], fresh=True)
    embedder = FakeEmbedder()
    got = get_note_vectors(vault, vault.root, embedder)
    assert got["Daily.md"] == [1.0, 2.0]
    assert embedder.calls == []  # never fell back for this note


def test_falls_back_when_stale(vault, vault_dir):
    _write_sc_vector(vault_dir, "Daily.md", [1.0, 2.0], fresh=False)
    embedder = FakeEmbedder()
    got = get_note_vectors(vault, vault.root, embedder)
    assert got["Daily.md"] != [1.0, 2.0]
    assert embedder.calls  # fallback was used


def test_falls_back_when_missing(vault):
    embedder = FakeEmbedder()
    got = get_note_vectors(vault, vault.root, embedder)
    assert "Daily.md" in got
    assert embedder.calls


def test_fallback_cached_across_calls(vault):
    embedder = FakeEmbedder()
    get_note_vectors(vault, vault.root, embedder)
    call_count_after_first = len(embedder.calls)
    get_note_vectors(vault, vault.root, embedder)
    assert len(embedder.calls) == call_count_after_first  # no re-embedding


def test_stale_notes_lists_only_uncached_edits(vault, vault_dir, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    get_note_vectors(vault, state, FakeEmbedder())  # warm the fallback cache
    assert stale_notes(vault, state) == []

    (vault_dir / "Daily.md").write_text("edited content\n", encoding="utf-8")
    assert stale_notes(vault, state) == ["Daily.md"]


def test_stale_notes_does_not_write(vault, vault_dir, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    stale_notes(vault, state)
    assert not list(state.iterdir())


def test_load_vector_cache_memo_tracks_disk_changes(tmp_path):
    from tesseract_mcp.embeddings import load_vector_cache, save_vector_cache

    p = tmp_path / "vecs.json"
    save_vector_cache(p, {"a": {"hash": "h1", "vec": [1.0]}})
    assert load_vector_cache(p) == {"a": {"hash": "h1", "vec": [1.0]}}
    # a repeat load returns the memoized parse (same content)
    assert load_vector_cache(p)["a"]["hash"] == "h1"
    # an out-of-process rewrite (new stamp) must be picked up. The rewrite is
    # the same size, and on Windows it can land in the same mtime tick as the
    # save above, so move mtime forward to give it the new stamp the memo keys on.
    before = p.stat()
    p.write_text('{"a": {"hash": "h2", "vec": [2.0]}}', encoding="utf-8")
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert load_vector_cache(p)["a"]["hash"] == "h2"


def test_load_vector_cache_corrupt_returns_empty(tmp_path):
    from tesseract_mcp.embeddings import load_vector_cache

    p = tmp_path / "vecs.json"
    p.write_text("{ truncated", encoding="utf-8")
    assert load_vector_cache(p) == {}


def test_load_vector_cache_mutation_does_not_leak_into_memo(tmp_path):
    from tesseract_mcp.embeddings import load_vector_cache, save_vector_cache

    p = tmp_path / "vecs.json"
    save_vector_cache(p, {"a": {"hash": "h1", "vec": [1.0]}})
    # callers mutate the loaded dict before saving; if that save never lands
    # (crash, locked file), the next load must still reflect what's on disk
    loaded = load_vector_cache(p)
    loaded["b"] = {"hash": "h2", "vec": [2.0]}
    del loaded["a"]
    assert load_vector_cache(p) == {"a": {"hash": "h1", "vec": [1.0]}}


def test_save_vector_cache_failed_replace_keeps_old_file_and_no_temp(tmp_path, monkeypatch):
    import pytest
    from tesseract_mcp import embeddings as emb

    p = tmp_path / "vecs.json"
    emb.save_vector_cache(p, {"a": {"hash": "h1", "vec": [1.0]}})

    def locked(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(emb.os, "replace", locked)
    with pytest.raises(PermissionError):
        emb.save_vector_cache(p, {"b": {"hash": "h2", "vec": [2.0]}})
    assert [f.name for f in tmp_path.iterdir()] == ["vecs.json"]
    assert emb.load_vector_cache(p) == {"a": {"hash": "h1", "vec": [1.0]}}


def test_persist_false_embeds_without_writing_cache(vault, tmp_path):
    state = tmp_path / "state-dry"
    state.mkdir()
    got = get_note_vectors(vault, state, FakeEmbedder(), persist=False)
    assert got and not list(state.iterdir())


def test_retired_notes_are_never_embedded(vault, vault_dir):
    # Retirement rewrites the tombstone, which would invalidate its cache
    # entry and re-embed the whole retired set for vectors fusion discards.
    graph = vault_dir / "Claude" / "Graph" / "Topics"
    graph.mkdir(parents=True)
    (graph / "Logistics.md").write_text(
        "---\nentity: topic\nretired: 2026-10-19 21:00\n---\n\n# Logistics\n",
        encoding="utf-8",
    )
    embedder = FakeEmbedder()
    vectors = get_note_vectors(vault, vault_dir, embedder)
    assert "Claude/Graph/Topics/Logistics.md" not in vectors
    assert "Claude/Graph/Topics/Logistics.md" not in stale_notes(vault, vault_dir)


def _all_notes_get_a_fresh_sc_vector(vault_dir, vec):
    for path in sorted(vault_dir.rglob("*.md")):
        rel = "/".join(path.relative_to(vault_dir).parts)
        if not rel.startswith("."):
            _write_sc_vector(vault_dir, rel, vec, fresh=True)


def test_smart_connections_vector_goes_stale_when_the_note_is_edited_later(vault, vault_dir):
    _all_notes_get_a_fresh_sc_vector(vault_dir, [1.0, 2.0])
    embedder = FakeEmbedder()
    assert get_note_vectors(vault, vault.root, embedder)["Daily.md"] == [1.0, 2.0]
    assert embedder.calls == []

    note = vault_dir / "Daily.md"
    note.write_text("# Daily\n\nrewritten after the embedding\n", encoding="utf-8")
    later = note.stat().st_mtime + 7200  # after Smart Connections embedded it
    os.utime(note, (later, later))
    got = get_note_vectors(vault, vault.root, embedder)
    assert embedder.calls == [["# Daily\n\nrewritten after the embedding\n"]]
    assert got["Daily.md"] != [1.0, 2.0]


def test_second_process_sees_vectors_the_first_saved(vault, vault_dir, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    assert "Daily.md" in stale_notes(vault, state)  # this process now holds an empty store

    src = Path(embeddings.__file__).resolve().parent.parent
    script = (
        "from tesseract_mcp.embeddings import get_note_vectors\n"
        "from tesseract_mcp.vault import Vault\n"
        "import sys\n"
        "class E:\n"
        "    def embed_batch(self, texts):\n"
        "        return [[float(len(t)), 1.0] for t in texts]\n"
        "get_note_vectors(Vault(sys.argv[1]), sys.argv[2], E())\n"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(vault_dir), str(state)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )

    embedder = FakeEmbedder()
    assert stale_notes(vault, state) == []
    get_note_vectors(vault, state, embedder)
    assert embedder.calls == []


def test_vectors_survive_a_restart(vault, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    first = get_note_vectors(vault, state, FakeEmbedder())
    embeddings.reset()
    embedder = FakeEmbedder()
    assert get_note_vectors(vault, state, embedder) == first
    assert embedder.calls == []


def test_old_json_cache_is_imported_and_left_in_place(vault, vault_dir, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    text = (vault_dir / "Daily.md").read_text(encoding="utf-8")
    legacy = state / "fallback_embeddings.json"
    legacy.write_text(
        json.dumps({
            "Daily.md": {
                "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "vec": [7.0, 8.0],
            }
        }),
        encoding="utf-8",
    )
    embedder = FakeEmbedder()
    got = get_note_vectors(vault, state, embedder)
    assert got["Daily.md"] == [7.0, 8.0]
    assert all("Daily" not in t for call in embedder.calls for t in call)
    assert legacy.exists()

    embeddings.reset()
    legacy.unlink()
    assert get_note_vectors(vault, state, FakeEmbedder())["Daily.md"] == [7.0, 8.0]


def test_stale_notes_reads_an_old_json_cache_without_writing(vault, vault_dir, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    text = (vault_dir / "Daily.md").read_text(encoding="utf-8")
    (state / "fallback_embeddings.json").write_text(
        json.dumps({
            "Daily.md": {
                "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "vec": [7.0, 8.0],
            }
        }),
        encoding="utf-8",
    )
    assert "Daily.md" not in stale_notes(vault, state)
    assert [p.name for p in state.iterdir()] == ["fallback_embeddings.json"]


def test_vector_of_a_deleted_note_is_not_kept(vault, vault_dir, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    get_note_vectors(vault, state, FakeEmbedder())
    daily = vault_dir / "Daily.md"
    text = daily.read_text(encoding="utf-8")
    daily.unlink()
    (vault_dir / "Other.md").write_text("something new\n", encoding="utf-8")
    get_note_vectors(vault, state, FakeEmbedder())  # embeds Other.md, saves

    daily.write_text(text, encoding="utf-8")
    embedder = FakeEmbedder()
    get_note_vectors(vault, state, embedder)
    assert embedder.calls == [[text]]


def test_vectors_from_another_model_never_match(vault, vault_dir, tmp_path):
    class OtherModel:
        def embed_batch(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    state = tmp_path / "state"
    state.mkdir()
    (vault_dir / "Contractors.md").write_text("Outstanding invoices.\n", encoding="utf-8")
    get_note_vectors(vault, state, FakeEmbedder())  # two-dimensional vectors, from the old model
    # No shared words with the query: a hit could only come from the vectors.
    assert hybrid_search(vault, state, OtherModel(), "zzz qqq") == []
