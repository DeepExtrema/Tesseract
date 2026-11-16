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
    assert "Proposals" in org_vault.read(ORGANIZER_NOTE)


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
