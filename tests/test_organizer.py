import asyncio
import os
from pathlib import Path

import json
import threading

import pytest

from tesseract_mcp import indexer
from tesseract_mcp.mover import move_note
from tesseract_mcp.organizer import (
    ORGANIZER_NOTE,
    VOTE_K,
    VOTE_THRESHOLD,
    Classification,
    classify,
    discover_taxonomy,
    iter_candidates,
    iter_organized,
    journal_path,
    record_move,
    run_sweep,
    undo_move,
)
from tesseract_mcp.vault import Vault, VaultError


@pytest.fixture
def org_vault(tmp_path):
    """A vault with two topical folders, excluded dirs, and loose notes."""
    for d in (".obsidian", ".smart-env", ".trash", "00 - Maps of Content",
              "Claude/Inbox", "02 - Space", "05 - Cooking"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "02 - Space" / "NASA JPL.md").write_text(
        "space telemetry anomaly research\n", encoding="utf-8")
    (tmp_path / "02 - Space" / "SmallSat.md").write_text(
        "space conference smallsat\n", encoding="utf-8")
    (tmp_path / "02 - Space" / "Telemanom.md").write_text(
        "space lstm telemetry\n", encoding="utf-8")
    (tmp_path / "05 - Cooking" / "Sourdough.md").write_text(
        "recipe starter flour\n", encoding="utf-8")
    (tmp_path / "05 - Cooking" / "Ramen.md").write_text(
        "recipe broth noodles\n", encoding="utf-8")
    (tmp_path / "00 - Maps of Content" / "Home.md").write_text(
        "moc\n", encoding="utf-8")
    (tmp_path / "Claude" / "Inbox" / "capture.md").write_text(
        "agent capture\n", encoding="utf-8")
    (tmp_path / "Loose Space Note.md").write_text(
        "space orbital telemetry note\n", encoding="utf-8")
    (tmp_path / "Pinned.md").write_text(
        "---\norganize: false\n---\n\nspace note that must stay put\n",
        encoding="utf-8")
    return Vault(tmp_path)


def test_constants_match_spec():
    assert VOTE_K == 10
    assert VOTE_THRESHOLD == 0.7


def test_discover_taxonomy_excludes_hard_exclusions(org_vault):
    assert discover_taxonomy(org_vault) == ["02 - Space", "05 - Cooking"]


def test_discover_taxonomy_picks_up_new_human_folder(org_vault):
    (org_vault.root / "07 - Finance").mkdir()
    assert "07 - Finance" in discover_taxonomy(org_vault)


def test_iter_organized_lists_taxonomy_notes_only(org_vault):
    organized = iter_organized(org_vault)
    assert "02 - Space/NASA JPL.md" in organized
    assert "05 - Cooking/Ramen.md" in organized
    assert not any(p.startswith("Claude/") for p in organized)
    assert not any(p.startswith("00 - Maps of Content") for p in organized)


def test_iter_candidates_root_and_organized_minus_pinned(org_vault):
    candidates = iter_candidates(org_vault)
    assert "Loose Space Note.md" in candidates
    assert "02 - Space/NASA JPL.md" in candidates       # filed notes are re-checkable
    assert "Pinned.md" not in candidates                # organize: false
    assert "Claude/Inbox/capture.md" not in candidates  # excluded dir


SPACE = [1.0, 0.0]
COOK = [0.0, 1.0]
MIXED = [0.7, 0.7]

LABELED_VECS = {
    "02 - Space/NASA JPL.md": SPACE,
    "02 - Space/SmallSat.md": SPACE,
    "02 - Space/Telemanom.md": SPACE,
    "05 - Cooking/Sourdough.md": COOK,
    "05 - Cooking/Ramen.md": COOK,
}
LABELED = list(LABELED_VECS)


def test_classify_clear_majority():
    vectors = {**LABELED_VECS, "Loose Space Note.md": [0.9, 0.1]}
    got = classify("Loose Space Note.md", vectors, LABELED)
    assert got.folder == "02 - Space"
    assert got.share >= 0.7
    assert "02 - Space/NASA JPL.md" in got.neighbors


def test_classify_split_vote_low_share():
    vectors = {**LABELED_VECS, "Ambiguous.md": MIXED}
    got = classify("Ambiguous.md", vectors, LABELED)
    assert got.share < 0.7


def test_classify_candidate_never_votes_for_itself():
    vectors = {**LABELED_VECS, "02 - Space/NASA JPL.md": SPACE}
    got = classify("02 - Space/NASA JPL.md", vectors, LABELED)
    assert "02 - Space/NASA JPL.md" not in got.neighbors


def test_classify_no_vector_or_no_labeled_returns_none():
    got = classify("Unknown.md", LABELED_VECS, LABELED)  # no vector for it
    assert got.folder is None and got.share == 0.0
    got2 = classify("X.md", {"X.md": SPACE}, [])          # nothing labeled
    assert got2.folder is None


@pytest.fixture
def moved(org_vault):
    record = move_note(org_vault, "Loose Space Note.md", "02 - Space/Loose Space Note.md")
    record_move(org_vault, record, share=0.85,
                neighbors=["02 - Space/NASA JPL.md"])
    return record


def test_record_move_writes_jsonl_and_note(org_vault, moved):
    lines = journal_path(org_vault).read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["from"] == "Loose Space Note.md"
    assert entry["to"] == "02 - Space/Loose Space Note.md"
    assert entry["share"] == 0.85
    note = org_vault.read(ORGANIZER_NOTE)
    assert "Loose Space Note" in note and "0.85" in note


def test_undo_restores_location_and_journal(org_vault, moved):
    result = undo_move(org_vault, "02 - Space/Loose Space Note.md")
    assert result["restored"] == "Loose Space Note.md"
    assert (org_vault.root / "Loose Space Note.md").is_file()
    assert not (org_vault.root / "02 - Space" / "Loose Space Note.md").exists()


def test_undo_twice_raises(org_vault, moved):
    undo_move(org_vault, "02 - Space/Loose Space Note.md")
    with pytest.raises(VaultError, match="No undoable move"):
        undo_move(org_vault, "02 - Space/Loose Space Note.md")


def test_undo_unknown_path_raises(org_vault):
    with pytest.raises(VaultError, match="No undoable move"):
        undo_move(org_vault, "02 - Space/Never Moved.md")


def test_undo_transfers_failure_record_back(org_vault, moved):
    """Undo restores a failing note's retry count to its original path."""
    manifest = indexer.load_manifest(org_vault.root)
    manifest["failures"]["02 - Space/Loose Space Note.md"] = {
        "error": "boom", "attempts": 2}
    indexer.save_manifest(manifest, org_vault.root)
    undo_move(org_vault, "02 - Space/Loose Space Note.md")
    manifest = indexer.load_manifest(org_vault.root)
    assert "02 - Space/Loose Space Note.md" not in manifest["failures"]
    assert manifest["failures"]["Loose Space Note.md"] == {
        "error": "boom", "attempts": 2}


class ClusterEmbedder:
    """space→[1,0], recipe→[0,1], both/neither→[0.7,0.7]. Deterministic."""

    def embed_batch(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            has_space, has_recipe = "space" in low, "recipe" in low
            if has_space and not has_recipe:
                out.append([1.0, 0.0])
            elif has_recipe and not has_space:
                out.append([0.0, 1.0])
            else:
                out.append([0.7, 0.7])
        return out


def test_sweep_dry_run_reports_but_touches_nothing(org_vault):
    report = run_sweep(org_vault, ClusterEmbedder(), apply=False)
    moves = {m["from"]: m["to_folder"] for m in report["moved"]}
    assert moves.get("Loose Space Note.md") == "02 - Space"
    assert (org_vault.root / "Loose Space Note.md").is_file()  # not actually moved
    assert not journal_path(org_vault).exists()


def test_sweep_apply_moves_and_journals(org_vault):
    report = run_sweep(org_vault, ClusterEmbedder(), apply=True)
    assert any(m["from"] == "Loose Space Note.md" for m in report["moved"])
    assert (org_vault.root / "02 - Space" / "Loose Space Note.md").is_file()
    assert not (org_vault.root / "Loose Space Note.md").exists()
    assert journal_path(org_vault).exists()
    assert report["cache_rebuilt"] is True


def test_sweep_correctly_filed_note_skipped(org_vault):
    report = run_sweep(org_vault, ClusterEmbedder(), apply=False)
    moved_from = [m["from"] for m in report["moved"]]
    assert "02 - Space/NASA JPL.md" not in moved_from  # already in the right place


def test_sweep_ambiguous_note_becomes_proposal(org_vault):
    (org_vault.root / "Fusion Cuisine In Space.md").write_text(
        "space station recipe experiments\n", encoding="utf-8")  # mixed → [0.7, 0.7]
    report = run_sweep(org_vault, ClusterEmbedder(), apply=True)
    props = [p["path"] for p in report["proposals"]]
    assert "Fusion Cuisine In Space.md" in props
    assert (org_vault.root / "Fusion Cuisine In Space.md").is_file()  # not moved
    assert "tesseract:proposals:start" in org_vault.read(ORGANIZER_NOTE)


def test_sweep_duplicate_stem_becomes_proposal(org_vault):
    (org_vault.root / "05 - Cooking" / "Loose Space Note.md").write_text(
        "recipe named confusingly\n", encoding="utf-8")
    report = run_sweep(org_vault, ClusterEmbedder(), apply=True)
    props = {p["path"]: p for p in report["proposals"]}
    assert "Loose Space Note.md" in props
    assert "duplicate" in props["Loose Space Note.md"]["reason"]
    assert (org_vault.root / "Loose Space Note.md").is_file()  # not moved


def test_sweep_move_cap_defers_confident_moves(org_vault):
    (org_vault.root / "Another Space Note.md").write_text(
        "space mission telemetry\n", encoding="utf-8")
    report = run_sweep(org_vault, ClusterEmbedder(), apply=True, max_moves=1)
    assert len(report["moved"]) == 1
    capped = [s["path"] for s in report["skipped"] if s["reason"] == "move cap reached"]
    assert len(capped) == 1
    assert (org_vault.root / capped[0]).is_file()  # deferred note stays put

    zero = run_sweep(org_vault, ClusterEmbedder(), apply=True, max_moves=0)
    assert zero["moved"] == [] and (org_vault.root / capped[0]).is_file()


def test_root_agent_guides_are_never_candidates(org_vault):
    (org_vault.root / "CLAUDE.md").write_text(
        "space vault guide for agents\n", encoding="utf-8")
    (org_vault.root / "AGENTS.md").write_text(
        "space agent instructions\n", encoding="utf-8")
    candidates = iter_candidates(org_vault)
    assert "CLAUDE.md" not in candidates
    assert "AGENTS.md" not in candidates
    # and a full sweep never proposes or moves them either
    report = run_sweep(org_vault, ClusterEmbedder(), apply=True)
    touched = [m["from"] for m in report["moved"]] + [p["path"] for p in report["proposals"]]
    assert "CLAUDE.md" not in touched and "AGENTS.md" not in touched
    assert (org_vault.root / "CLAUDE.md").is_file()


def test_dot_directories_never_taxonomy(org_vault):
    (org_vault.root / ".claude" / "commands").mkdir(parents=True)
    (org_vault.root / ".claude" / "commands" / "day.md").write_text(
        "space daily note command\n", encoding="utf-8")
    assert not any(f.startswith(".") for f in discover_taxonomy(org_vault))
    assert ".claude/commands/day.md" not in iter_candidates(org_vault)
    report = run_sweep(org_vault, ClusterEmbedder(), apply=True)
    touched = [m["from"] for m in report["moved"]] + [p["path"] for p in report["proposals"]]
    assert not any(t.startswith(".claude/") for t in touched)
    assert (org_vault.root / ".claude" / "commands" / "day.md").is_file()


def test_organizer_skips_sheet_folders(org_vault, tmp_path):
    folder = tmp_path / "Records"
    folder.mkdir()
    (folder / "_schema.md").write_text(
        "---\nsheet: things\nfilename: \"{name}\"\nkey: [name]\n"
        "columns:\n  name: {type: string}\n---\n", encoding="utf-8")
    (folder / "Row.md").write_text("---\nname: Row\n---\n", encoding="utf-8")
    vault = Vault(tmp_path)
    assert all("Records/" not in c for c in iter_candidates(vault))
    assert "Records" not in discover_taxonomy(vault)


def _sheet_linking_loose_note(org_vault):
    folder = org_vault.root / "Records"
    folder.mkdir()
    (folder / "_schema.md").write_text(
        "---\nsheet: things\nfilename: \"{name}\"\nkey: [name]\n"
        "columns:\n  name: {type: string}\n---\n", encoding="utf-8")
    row = folder / "Row.md"
    row.write_text("---\nname: Row\n---\n\nSee [[Loose Space Note]].\n",
                   encoding="utf-8")
    return row


@pytest.mark.parametrize("apply", [False, True])
def test_move_that_would_edit_a_sheet_row_becomes_a_proposal(org_vault, apply):
    """Sheet rows change only through sheet_upsert; the mover must not
    rewrite links inside them."""
    row = _sheet_linking_loose_note(org_vault)
    before = row.read_bytes()
    report = run_sweep(org_vault, ClusterEmbedder(), apply=apply)
    assert not any(m["from"] == "Loose Space Note.md" for m in report["moved"])
    prop = [p for p in report["proposals"] if p["path"] == "Loose Space Note.md"]
    assert prop and prop[0]["reason"].startswith("linked from sheet rows")
    assert (org_vault.root / "Loose Space Note.md").is_file()
    assert row.read_bytes() == before


def test_server_undo_refused_while_sweep_holds_caretaker_lock(
        org_vault, tmp_path, monkeypatch):
    """A tool-level undo must not interleave with a running apply sweep."""
    from mcp.server.fastmcp.exceptions import ToolError

    from tesseract_mcp import organizer as organizer_mod
    from tesseract_mcp import server
    from tesseract_mcp.locking import caretaker_lock

    monkeypatch.setenv("TESSERACT_VAULT_PATH", str(org_vault.root))
    monkeypatch.setenv("TESSERACT_STATE_DIR", str(tmp_path / "state"))
    server._vault = None
    record = move_note(org_vault, "02 - Space/SmallSat.md",
                       "02 - Space/SmallSat Renamed.md")
    record_move(org_vault, record, share=0.9, neighbors=[])

    in_sweep, release = threading.Event(), threading.Event()
    real_move = organizer_mod.move_note

    def paused_move(*args, **kwargs):
        in_sweep.set()
        assert release.wait(10)
        return real_move(*args, **kwargs)

    monkeypatch.setattr(organizer_mod, "move_note", paused_move)
    errors = []

    def sweep():
        try:
            with caretaker_lock(indexer.state_dir(org_vault.root)):
                run_sweep(org_vault, ClusterEmbedder(), apply=True)
        except Exception as e:  # surfaced below
            errors.append(e)

    t = threading.Thread(target=sweep)
    t.start()
    try:
        assert in_sweep.wait(10)
        snap = lambda: (
            sorted(p.relative_to(org_vault.root).as_posix()
                   for p in org_vault.root.rglob("*") if p.is_file()),
            journal_path(org_vault).read_bytes(),
            json.dumps(indexer.load_manifest(org_vault.root), sort_keys=True),
        )
        before = snap()
        with pytest.raises(ToolError, match="another caretaker"):
            asyncio.run(server.mcp.call_tool(
                "undo_move", {"path": "02 - Space/SmallSat Renamed.md"}))
        assert snap() == before
    finally:
        release.set()
        t.join(15)
    assert not t.is_alive()
    assert not errors, errors

    result = server.undo_move("02 - Space/SmallSat Renamed.md")
    assert result["restored"] == "02 - Space/SmallSat.md"
    server._vault = None


def _journal_state(org_vault):
    jp = journal_path(org_vault)
    return jp.read_bytes(), sorted(p.name for p in jp.parent.iterdir())


def test_undo_failed_journal_write_keeps_journal(org_vault, moved, monkeypatch):
    before = _journal_state(org_vault)
    real = Path.write_text

    def broken(self, data, *a, **kw):
        real(self, data[: len(data) // 2], *a, **kw)
        raise OSError("disk full")

    with monkeypatch.context() as mp:
        mp.setattr(Path, "write_text", broken)
        with pytest.raises(OSError):
            undo_move(org_vault, "02 - Space/Loose Space Note.md")
    assert _journal_state(org_vault) == before
    entries = [json.loads(l) for l in before[0].decode().splitlines()]
    assert entries[-1]["to"] == "02 - Space/Loose Space Note.md"
    assert not entries[-1].get("undone")


def test_undo_failed_journal_replace_keeps_journal(org_vault, moved, monkeypatch):
    before = _journal_state(org_vault)
    jp = journal_path(org_vault)
    real_replace = os.replace

    def locked(src, dst):
        if Path(dst) == jp:
            raise PermissionError("sharing violation")
        return real_replace(src, dst)

    with monkeypatch.context() as mp:
        mp.setattr("os.replace", locked)
        mp.setattr("time.sleep", lambda s: None)
        with pytest.raises(PermissionError):
            undo_move(org_vault, "02 - Space/Loose Space Note.md")
    assert _journal_state(org_vault) == before


def test_undo_success_marks_entry_undone_without_temp(org_vault, moved):
    undo_move(org_vault, "02 - Space/Loose Space Note.md")
    data, names = _journal_state(org_vault)
    assert json.loads(data.decode().splitlines()[-1])["undone"] is True
    assert not [n for n in names if n.endswith(".tmp-write")]


def _snapshot(org_vault):
    root = org_vault.root
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*.md") if "Organizer" not in p.name}


def _linking_vault(org_vault):
    (org_vault.root / "Linker.md").write_text(
        "see [[Loose Space Note]] and [[Loose Space Note|x]]\n", encoding="utf-8")


def test_sweep_journal_append_failure_leaves_no_unjournaled_move(
        org_vault, monkeypatch):
    from tesseract_mcp import organizer
    _linking_vault(org_vault)
    before = _snapshot(org_vault)
    manifest_before = indexer.load_manifest(org_vault.root)

    def broken(fd, data):
        raise OSError("disk full")
    monkeypatch.setattr(organizer, "_write_all", broken)
    with pytest.raises(OSError):
        run_sweep(org_vault, ClusterEmbedder(), apply=True)
    assert _snapshot(org_vault) == before
    assert indexer.load_manifest(org_vault.root) == manifest_before
    jp = journal_path(org_vault)
    assert not jp.exists() or jp.read_text(encoding="utf-8").strip() == ""
    cache = _fallback_cache(org_vault)  # rollback moved the vector key back too
    assert cache and all((org_vault.root / rel).is_file() for rel in cache)


def test_sweep_unwritable_journal_moves_nothing(org_vault):
    journal_path(org_vault).mkdir(parents=True)  # cannot be opened for append
    before = _snapshot(org_vault)
    emb = CountingEmbedder()
    with pytest.raises(OSError):
        run_sweep(org_vault, emb, apply=True)
    assert _snapshot(org_vault) == before
    assert emb.embedded == []  # refused before any work, not rolled back after


def test_torn_journal_line_does_not_block_undo(org_vault):
    record = move_note(org_vault, LOOSE, FILED)
    record_move(org_vault, record, share=0.9, neighbors=[])
    with journal_path(org_vault).open("ab") as f:
        f.write(b'{"ts": "2026-11-16 02:00:00", "from": "x.md", "to')  # killed mid-append
    record = move_note(org_vault, "02 - Space/SmallSat.md", "SmallSat.md")
    record_move(org_vault, record, share=0.9, neighbors=[])
    assert undo_move(org_vault, "SmallSat.md")["restored"] == "02 - Space/SmallSat.md"
    assert undo_move(org_vault, FILED)["restored"] == LOOSE


def test_move_with_unsavable_manifest_is_rolled_back(org_vault):
    _linking_vault(org_vault)
    before = _snapshot(org_vault)
    indexer._manifest_path(org_vault.root).write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        move_note(org_vault, LOOSE, FILED)
    assert _snapshot(org_vault) == before


def test_sweep_note_append_failure_keeps_move_undoable(org_vault, monkeypatch):
    from tesseract_mcp import vault as vault_mod
    _linking_vault(org_vault)
    before = _snapshot(org_vault)
    real = vault_mod._write_all

    def flaky(fd, data):
        if b"moved [[" in data:
            raise OSError("note locked")
        real(fd, data)
    monkeypatch.setattr(vault_mod, "_write_all", flaky)
    with pytest.raises(OSError, match="note locked"):
        run_sweep(org_vault, ClusterEmbedder(), apply=True)
    entries = [json.loads(l) for l in
               journal_path(org_vault).read_text(encoding="utf-8").splitlines()]
    assert entries and _snapshot(org_vault) != before
    for e in entries:
        undo_move(org_vault, e["to"])
    assert _snapshot(org_vault) == before


# ---- path-keyed search state follows moves and undos -----------------------

class CountingEmbedder(ClusterEmbedder):
    def __init__(self):
        self.embedded: list[str] = []

    def embed_batch(self, texts):
        self.embedded.extend(texts)
        return super().embed_batch(texts)


LOOSE = "Loose Space Note.md"
FILED = "02 - Space/Loose Space Note.md"


def _fallback_cache(vault):
    """Stored vectors as they are on disk, read by a fresh store."""
    from tesseract_mcp import embeddings
    return embeddings.VectorStore(
        indexer.state_dir(vault.root)).refresh(persist=False).rows


def _search(vault, embedder):
    from tesseract_mcp.hybrid import hybrid_search
    return [h.path for h in hybrid_search(
        vault, indexer.state_dir(vault.root), embedder, "orbital telemetry")]


def _warm(vault):
    from tesseract_mcp import embeddings
    embeddings.get_note_vectors(vault, indexer.state_dir(vault.root),
                                ClusterEmbedder())


def _note_text_embedded(embedder):
    return "space orbital telemetry note\n" in embedder.embedded


def test_move_carries_vector_instead_of_reembedding(org_vault):
    _warm(org_vault)
    move_note(org_vault, LOOSE, FILED)
    emb = CountingEmbedder()
    paths = _search(org_vault, emb)
    assert FILED in paths and LOOSE not in paths
    assert not _note_text_embedded(emb)
    cache = _fallback_cache(org_vault)
    assert FILED in cache and LOOSE not in cache


def test_undo_carries_vector_back(org_vault, moved):
    _warm(org_vault)  # warmed at the moved location
    undo_move(org_vault, FILED)
    emb = CountingEmbedder()
    paths = _search(org_vault, emb)
    assert LOOSE in paths and FILED not in paths
    assert not _note_text_embedded(emb)
    cache = _fallback_cache(org_vault)
    assert LOOSE in cache and FILED not in cache


def test_undo_completes_when_the_vector_store_is_unreadable(org_vault, moved):
    from tesseract_mcp import embeddings
    _warm(org_vault)
    state = indexer.state_dir(org_vault.root)
    (state / embeddings.STORE_FILE).write_bytes(b"not an npz file")
    embeddings.reset()
    undo_move(org_vault, FILED)
    assert (org_vault.root / LOOSE).is_file()
    entries = [json.loads(line) for line in
               journal_path(org_vault).read_text(encoding="utf-8").splitlines()]
    assert [e["undone"] for e in entries] == [True]


def test_move_carries_vector_from_the_old_json_store(org_vault):
    from tesseract_mcp import embeddings
    _warm(org_vault)
    state = indexer.state_dir(org_vault.root)
    legacy = {rel: {"hash": digest, "vec": vec.tolist()}
              for rel, (digest, vec) in _fallback_cache(org_vault).items()}
    (state / embeddings.LEGACY_JSON_FILE).write_text(json.dumps(legacy), encoding="utf-8")
    (state / embeddings.STORE_FILE).unlink()
    embeddings.reset()
    move_note(org_vault, LOOSE, FILED)
    emb = CountingEmbedder()
    paths = _search(org_vault, emb)
    assert FILED in paths and LOOSE not in paths
    assert not _note_text_embedded(emb)


def test_note_with_rewritten_backlink_is_still_found(org_vault):
    """A linker's content changes with the rewrite, so it is re-embedded (its
    hash no longer matches); it must still be searchable at its own path."""
    (org_vault.root / "Linker.md").write_text(
        "space orbital see [[Loose Space Note]]\n", encoding="utf-8")
    _warm(org_vault)
    move_note(org_vault, LOOSE, FILED)
    emb = CountingEmbedder()
    paths = _search(org_vault, emb)
    assert "Linker.md" in paths and FILED in paths


def _graph_vault(org_vault):
    from tesseract_mcp import cache
    (org_vault.root / "Claude" / "Graph" / "Topics").mkdir(parents=True)
    (org_vault.root / "Claude" / "Graph" / "Topics" / "Space.md").write_text(
        "# Space\n\nspace stuff\n\n## Mentions\n\n"
        "- [[Loose Space Note|Loose]] — e\n"
        "- [[02 - Space/NASA JPL|NASA]] — e\n", encoding="utf-8")
    cache.rebuild(org_vault, indexer.db_path(org_vault.root))


def test_related_notes_follow_undo_without_manual_rebuild(org_vault):
    from tesseract_mcp import cache
    _graph_vault(org_vault)
    db = indexer.db_path(org_vault.root)
    record = move_note(org_vault, LOOSE, FILED)
    record_move(org_vault, record, share=0.9, neighbors=[])
    cache.rebuild(org_vault, db)  # what run_sweep does
    got = [r["path"] for r in cache.related_notes(
        db, org_vault, "02 - Space/NASA JPL.md")]
    assert got == [FILED]
    undo_move(org_vault, FILED)
    got = [r["path"] for r in cache.related_notes(
        db, org_vault, "02 - Space/NASA JPL.md")]
    assert got == [LOOSE]


# --- agent move budget (organize_vault(apply=True)) ---------------------

BUDGET_ENV = "TESSERACT_AGENT_MOVES_PER_DAY"


@pytest.fixture
def budget_srv(org_vault, tmp_path, monkeypatch):
    """The server tool bound to org_vault with two movable loose notes."""
    from tesseract_mcp import server

    (org_vault.root / "Loose Cooking Note.md").write_text(
        "recipe broth noodles note\n", encoding="utf-8")
    monkeypatch.setenv("TESSERACT_VAULT_PATH", str(org_vault.root))
    monkeypatch.setenv("TESSERACT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("TESSERACT_NO_TOOL_APPLY", raising=False)
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    server._vault = None
    server._embedder = ClusterEmbedder()
    yield server
    server._vault = None
    server._embedder = None


def _loose(vault):
    return sorted(p.name for p in vault.root.glob("*.md")
                  if p.name.startswith("Loose "))


def _journal_line(ts, undone=False):
    return json.dumps({"ts": ts, "from": "x.md", "to": "A/x.md", "share": 1.0,
                       "neighbors": [], "rewrites": [], "undone": undone}) + "\n"


def _seed_journal(vault, *lines):
    jp = journal_path(vault)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text("".join(lines), encoding="utf-8")


def _now(days_ago=0):
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def test_agent_apply_default_budget_moves_nothing(org_vault, budget_srv):
    before = _loose(org_vault)
    result = budget_srv.organize_vault(apply=True)
    assert result["moved"] == []
    assert _loose(org_vault) == before
    assert not journal_path(org_vault).read_text(encoding="utf-8")
    assert result["move_budget"] == {
        "per_day": 0, "used_today": 0, "remaining": 0, "setting": BUDGET_ENV}
    assert BUDGET_ENV in result["note"]
    assert "proposals" in result
    assert {s["path"] for s in result["skipped"]
            if s["reason"] == "move cap reached"} == set(before)


def test_agent_apply_budget_one_moves_once_per_day(org_vault, budget_srv, monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "1")
    first = budget_srv.organize_vault(apply=True)
    assert len(first["moved"]) == 1
    assert first["move_budget"] == {
        "per_day": 1, "used_today": 1, "remaining": 0, "setting": BUDGET_ENV}
    assert len(_loose(org_vault)) == 1
    second = budget_srv.organize_vault(apply=True)
    assert second["moved"] == []
    assert second["move_budget"] == {
        "per_day": 1, "used_today": 1, "remaining": 0, "setting": BUDGET_ENV}
    assert len(_loose(org_vault)) == 1
    assert len(journal_path(org_vault).read_text(encoding="utf-8").splitlines()) == 1


def test_undone_move_still_counts_against_budget(org_vault, budget_srv, monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "1")
    _seed_journal(org_vault, _journal_line(_now(), undone=True))
    result = budget_srv.organize_vault(apply=True)
    assert result["moved"] == []
    assert result["move_budget"]["used_today"] == 1
    assert len(_loose(org_vault)) == 2


def test_yesterdays_and_torn_journal_lines_do_not_count(
        org_vault, budget_srv, monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "1")
    _seed_journal(org_vault, _journal_line(_now(2)), '{"ts": "torn\n',
                  _journal_line(_now(1)))
    result = budget_srv.organize_vault(apply=True)
    assert len(result["moved"]) == 1  # the budget of 1 was still whole
    assert result["move_budget"]["used_today"] == 1  # only this call's move


def test_budget_subtracts_moves_already_made_today(org_vault, budget_srv, monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "2")
    _seed_journal(org_vault, _journal_line(_now()))
    result = budget_srv.organize_vault(apply=True)
    assert result["move_budget"]["used_today"] == 2
    assert result["move_budget"]["remaining"] == 0
    assert len(result["moved"]) == 1
    assert len(_loose(org_vault)) == 1


def test_no_tool_apply_refusal_precedes_the_budget(org_vault, budget_srv, monkeypatch):
    monkeypatch.setenv("TESSERACT_NO_TOOL_APPLY", "1")
    monkeypatch.setenv(BUDGET_ENV, "abc")  # would raise its own error if read first
    with pytest.raises(VaultError, match="TESSERACT_NO_TOOL_APPLY"):
        budget_srv.organize_vault(apply=True)
    assert len(_loose(org_vault)) == 2


def test_dry_run_with_zero_budget_is_not_truncated(org_vault, budget_srv):
    uncapped = run_sweep(org_vault, ClusterEmbedder(), apply=False)
    preview = budget_srv.organize_vault(apply=False)
    assert len(uncapped["moved"]) == 2
    assert preview["moved"] == uncapped["moved"]
    assert preview["proposals"] == uncapped["proposals"]
    assert preview["skipped"] == uncapped["skipped"]


@pytest.mark.parametrize("bad", ["abc", "-1", "1.5"])
def test_invalid_budget_names_the_variable(org_vault, budget_srv, monkeypatch, bad):
    monkeypatch.setenv(BUDGET_ENV, bad)
    before = _loose(org_vault)
    with pytest.raises(VaultError, match=BUDGET_ENV):
        budget_srv.organize_vault(apply=True)
    assert _loose(org_vault) == before


# --- proposal lifecycle (managed block in Claude/Organizer.md) -----------

from datetime import date, datetime, timedelta  # noqa: E402

from tesseract_mcp import librarian, recall  # noqa: E402
from tesseract_mcp.organizer import (  # noqa: E402
    PROPOSAL_STALE_DAYS,
    parse_proposal_block,
    render_proposal_block,
    stale_proposals,
)

D0 = date(2026, 1, 1)
FUSION = "Fusion Cuisine In Space.md"
BLOCK_START = "<!-- tesseract:proposals:start -->"


def _day(n):
    return D0 + timedelta(days=n)


@pytest.fixture
def prop_vault(org_vault):
    """org_vault plus one mixed note the sweep can only propose."""
    (org_vault.root / FUSION).write_text(
        "space station recipe experiments\n", encoding="utf-8")
    return org_vault


def _sweep(vault, n=0, apply=True, **kw):
    return run_sweep(vault, ClusterEmbedder(), apply=apply, today=_day(n), **kw)


def _section(note, heading):
    """Lines of one checkbox list inside the managed block."""
    block = note.split(BLOCK_START, 1)[1]
    parts = block.split(f"#### {heading}\n", 1)[1]
    return [ln for ln in parts.split("#### ")[0].splitlines() if ln.startswith("- [")]


def _bytes_and_mtimes(*roots):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for r in roots for p in sorted(Path(r).rglob("*")) if p.is_file()}


def test_stale_days_is_thirty():
    assert PROPOSAL_STALE_DAYS == 30


def test_block_round_trips_every_field():
    entries = [
        {"path": "A; proposed x.md", "suggested": "02 - Space", "share": 0.55,
         "reason": "duplicate stem — bare links; would become ambiguous",
         "proposed": "2026-01-01", "stale_since": "2026-01-31",
         "state": "needs-resolution", "ticked": False},
        {"path": "B.md", "suggested": "05 - Cooking", "share": 0.0,
         "reason": "linked from sheet rows: Jobs/a.md, Jobs/b.md",
         "proposed": "2026-01-02", "stale_since": None,
         "state": "pending", "ticked": False},
        {"path": "C.md", "suggested": "02 - Space", "share": 0.667,
         "reason": "low confidence", "proposed": "2026-01-03",
         "stale_since": None, "state": "dismissed", "ticked": False},
    ]
    assert parse_proposal_block(render_proposal_block(entries)) == entries


def test_proposal_listed_once_and_second_sweep_writes_nothing(prop_vault):
    _sweep(prop_vault, 0)
    _sweep(prop_vault, 0)  # settle: the move line from sweep 1 is in the note
    first = prop_vault.read(ORGANIZER_NOTE)
    note_path = prop_vault.root / ORGANIZER_NOTE
    stamp = note_path.stat().st_mtime_ns
    report = _sweep(prop_vault, 1)
    assert prop_vault.read(ORGANIZER_NOTE) == first
    assert note_path.stat().st_mtime_ns == stamp  # not rewritten, not even identically
    pending = _section(first, "Pending")
    assert len(pending) == 1 and FUSION in pending[0]
    assert "proposed 2026-01-01" in pending[0]
    prop = [p for p in report["proposals"] if p["path"] == FUSION][0]
    assert (prop["state"], prop["proposed"], prop["age_days"]) == (
        "pending", "2026-01-01", 1)


@pytest.mark.parametrize("tick", ["- [x]", "- [X]", "- [x]+✅", "- [-]"])
def test_ticked_line_is_dismissed_and_never_reproposed(prop_vault, tick):
    _sweep(prop_vault, 0)
    note = prop_vault.read(ORGANIZER_NOTE)
    box, _, suffix = tick.partition("+")
    lines = []
    for ln in note.splitlines():
        if ln.startswith(f"- [ ] `{FUSION}`"):
            ln = box + ln[5:] + (f" {suffix} 2026-01-01" if suffix else "")
        lines.append(ln)
    prop_vault.write(ORGANIZER_NOTE, "\n".join(lines) + "\n", overwrite=True)
    report = _sweep(prop_vault, 1)
    text = prop_vault.read(ORGANIZER_NOTE)
    assert len(_section(text, "Dismissed")) == 1
    assert _section(text, "Pending") == []
    assert "proposal dismissed" in text
    assert (prop_vault.root / FUSION).is_file()
    for later in (_sweep(prop_vault, 2), _sweep(prop_vault, 3, apply=False)):
        assert FUSION not in [p["path"] for p in later["proposals"]]
        assert {"path": FUSION, "reason": "proposal dismissed"} in later["skipped"]
    assert {"path": FUSION, "reason": "proposal dismissed"} in report["skipped"]


def test_dismissed_pair_is_not_moved_when_the_vote_turns_confident(prop_vault):
    _sweep(prop_vault, 0)
    note = prop_vault.read(ORGANIZER_NOTE)
    prop_vault.write(ORGANIZER_NOTE, note.replace(
        f"- [ ] `{FUSION}`", f"- [x] `{FUSION}`"), overwrite=True)
    # the note now votes for the folder the owner turned down
    (prop_vault.root / FUSION).write_text("space orbital telemetry note\n",
                                          encoding="utf-8")
    for day in (1, 2):
        report = _sweep(prop_vault, day)
        assert FUSION not in [m["from"] for m in report["moved"]]
        assert {"path": FUSION, "reason": "proposal dismissed"} in report["skipped"]
    assert (prop_vault.root / FUSION).is_file()
    text = prop_vault.read(ORGANIZER_NOTE)
    assert len(_section(text, "Dismissed")) == 1
    assert "proposal applied" not in text


def test_capped_move_keeps_its_stored_proposal(prop_vault):
    _sweep(prop_vault, 0, max_moves=0)
    assert len(_section(prop_vault.read(ORGANIZER_NOTE), "Pending")) == 1
    # the vote now clears the threshold, but a cap holds the move back
    (prop_vault.root / FUSION).write_text("space orbital telemetry note\n",
                                          encoding="utf-8")
    report = _sweep(prop_vault, 1, max_moves=0)
    text = prop_vault.read(ORGANIZER_NOTE)
    assert any(s["path"] == FUSION and s["reason"] == "move cap reached"
               for s in report["skipped"])
    assert len(_section(text, "Pending")) == 1
    assert "proposal withdrawn" not in text


def test_box_ticked_during_a_sweep_is_kept(prop_vault, monkeypatch):
    from tesseract_mcp import organizer
    _sweep(prop_vault, 0, max_moves=0)
    real = organizer.embeddings_mod.get_note_vectors

    def tick_then_embed(*a, **kw):
        note = prop_vault.read(ORGANIZER_NOTE)
        prop_vault.write(ORGANIZER_NOTE, note.replace(
            f"- [ ] `{FUSION}`", f"- [x] `{FUSION}`"), overwrite=True)
        return real(*a, **kw)

    monkeypatch.setattr(organizer.embeddings_mod, "get_note_vectors", tick_then_embed)
    _sweep(prop_vault, 1, max_moves=0)
    text = prop_vault.read(ORGANIZER_NOTE)
    assert len(_section(text, "Dismissed")) == 1
    assert _section(text, "Pending") == []


@pytest.mark.parametrize(
    "damage", ["bad-date", "lost-end-marker", "lost-start-marker", "unreadable-line"])
def test_damaged_block_fails_before_any_move(prop_vault, damage):
    line = (f"- [ ] `{FUSION}` → **02 - Space** (share 0.6; low confidence; "
            f"proposed {'2026-02-30' if damage == 'bad-date' else '2026-01-01'})")
    if damage == "unreadable-line":
        line = f"- [ ] {FUSION} to 02 - Space, maybe"
    start = "" if damage == "lost-start-marker" else BLOCK_START
    end = "" if damage == "lost-end-marker" else "<!-- tesseract:proposals:end -->\n"
    prop_vault.write(ORGANIZER_NOTE, "# Organizer\n\n" + start
                     + "\n#### Pending\n" + line + "\n" + end, overwrite=True)
    before = _snapshot(prop_vault)
    with pytest.raises(VaultError):
        _sweep(prop_vault, 0)
    assert _snapshot(prop_vault) == before


def test_pending_turns_stale_at_threshold_and_logs_once(prop_vault):
    _sweep(prop_vault, 0)
    _sweep(prop_vault, PROPOSAL_STALE_DAYS - 1)
    text = prop_vault.read(ORGANIZER_NOTE)
    assert len(_section(text, "Pending")) == 1
    assert stale_proposals(prop_vault, _day(PROPOSAL_STALE_DAYS - 1)) == []

    report = _sweep(prop_vault, PROPOSAL_STALE_DAYS)
    text = prop_vault.read(ORGANIZER_NOTE)
    stale = _section(text, "Needs resolution")
    assert len(stale) == 1 and "proposed 2026-01-01" in stale[0]
    assert _section(text, "Pending") == []
    assert text.count("proposal needs-resolution") == 1
    assert [p["state"] for p in report["proposals"] if p["path"] == FUSION] == [
        "needs-resolution"]
    health = librarian.run_health(prop_vault, {}, None, None, {})
    assert health["stale_proposals"] == 1

    again = _sweep(prop_vault, PROPOSAL_STALE_DAYS + 5)
    assert prop_vault.read(ORGANIZER_NOTE) == text
    assert [p["age_days"] for p in again["proposals"] if p["path"] == FUSION] == [
        PROPOSAL_STALE_DAYS + 5]


def test_stale_note_filed_by_owner_leaves_as_applied(prop_vault):
    _sweep(prop_vault, 0)
    _sweep(prop_vault, PROPOSAL_STALE_DAYS)
    move_note(prop_vault, FUSION, f"02 - Space/{FUSION}")
    _sweep(prop_vault, PROPOSAL_STALE_DAYS + 1)
    text = prop_vault.read(ORGANIZER_NOTE)
    assert _section(text, "Needs resolution") == []
    assert "proposal applied" in text
    assert stale_proposals(prop_vault, _day(PROPOSAL_STALE_DAYS + 1)) == []
    assert librarian.run_health(prop_vault, {}, None, None, {})[
        "stale_proposals"] == 0


@pytest.mark.parametrize("how", ["delete", "organize-false"])
def test_pending_note_gone_or_pinned_is_withdrawn_never_stale(prop_vault, how):
    _sweep(prop_vault, 0)
    path = prop_vault.root / FUSION
    if how == "delete":
        path.unlink()
    else:
        path.write_text("---\norganize: false\n---\n\nspace station recipe\n",
                        encoding="utf-8")
    _sweep(prop_vault, PROPOSAL_STALE_DAYS + 10)
    text = prop_vault.read(ORGANIZER_NOTE)
    assert _section(text, "Pending") == []
    assert _section(text, "Needs resolution") == []
    assert "proposal withdrawn" in text
    assert stale_proposals(prop_vault, _day(PROPOSAL_STALE_DAYS + 10)) == []


def test_synced_stale_item_is_reported_without_any_write(prop_vault):
    """A machine that never ran an applying sweep still sees the item."""
    line = (f"- [ ] `{FUSION}` → **02 - Space** (share 0.6; low confidence; "
            f"proposed 2026-01-01; stale since 2026-01-31)")
    prop_vault.write(
        ORGANIZER_NOTE,
        "# Organizer\n\n## Log\n\n" + BLOCK_START + "\n#### Needs resolution\n"
        + line + "\n#### Pending\n#### Dismissed\n<!-- tesseract:proposals:end -->\n",
        overwrite=True)
    state = indexer.state_dir(prop_vault.root)
    state.mkdir(parents=True, exist_ok=True)
    before = _bytes_and_mtimes(prop_vault.root, state)

    report = _sweep(prop_vault, 45, apply=False)
    digest = recall.digest_bundle(prop_vault, now=datetime(2026, 2, 15, 12, 0))

    assert [(p["state"], p["age_days"]) for p in report["proposals"]
            if p["path"] == FUSION] == [("needs-resolution", 45)]
    props = digest["proposals"]
    assert props["stale"] == 1
    assert props["stale_items"][0]["path"] == FUSION
    assert props["stale_items"][0]["age_days"] == 45
    assert _bytes_and_mtimes(prop_vault.root, state) == before


def test_old_proposal_sections_stay_byte_identical_and_block_added_once(prop_vault):
    old = ("# Organizer\n\nAutonomous move log and proposals.\n\n## Log\n"
           + "".join(f"\n### Proposals 2025-12-0{i}\n"
                     f"- `Old {i}.md` → **02 - Space** (share 0.5; low confidence)\n"
                     for i in (1, 2, 3)))
    prop_vault.write(ORGANIZER_NOTE, old, overwrite=True)
    _sweep(prop_vault, 0, max_moves=0)
    first = prop_vault.read(ORGANIZER_NOTE)
    assert first.startswith(old)
    assert first.count(BLOCK_START) == 1
    _sweep(prop_vault, 1, max_moves=0)
    assert prop_vault.read(ORGANIZER_NOTE) == first


def test_agent_apply_with_unchanged_proposals_writes_nothing(
        budget_srv, org_vault):
    (org_vault.root / FUSION).write_text(
        "space station recipe experiments\n", encoding="utf-8")
    budget_srv.organize_vault(apply=True)
    first = org_vault.read(ORGANIZER_NOTE)
    note_path = org_vault.root / ORGANIZER_NOTE
    stamp = note_path.stat().st_mtime_ns
    result = budget_srv.organize_vault(apply=True)
    assert result["moved"] == []
    assert org_vault.read(ORGANIZER_NOTE) == first
    assert note_path.stat().st_mtime_ns == stamp
