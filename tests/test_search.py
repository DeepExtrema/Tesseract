from tesseract_mcp.search import (
    as_str_list,
    body_text,
    iter_candidate_notes,
    is_skipped,
    iter_note_files,
)


def _paths(pairs):
    return [rel for rel, _text in pairs]


def test_iter_note_files_skips_obsidian_dir(vault):
    rels = [rel for _p, rel in iter_note_files(vault)]
    assert rels
    assert not any(rel.startswith(".obsidian") for rel in rels)


def test_iter_note_files_folder_scope(vault):
    rels = [rel for _p, rel in iter_note_files(vault, "Claude")]
    assert rels
    assert all(rel.startswith("Claude/") for rel in rels)


def test_candidates_unfiltered_list_all_notes(vault):
    rels = _paths(iter_candidate_notes(vault))
    assert "Projects/Sentinel ESG.md" in rels
    assert "Daily.md" in rels


def test_candidates_tag_filter(vault):
    got = _paths(iter_candidate_notes(vault, tags=["esg"]))
    assert got == ["Projects/Sentinel ESG.md"]


def test_candidates_tag_filter_case_insensitive(vault):
    got = _paths(iter_candidate_notes(vault, tags=["ESG"]))
    assert got == ["Projects/Sentinel ESG.md"]


def test_candidates_multi_tag_filter_requires_all(vault):
    assert _paths(iter_candidate_notes(vault, tags=["esg", "project"]))
    assert iter_candidate_notes(vault, tags=["esg", "nope"]) == []


def test_candidates_scalar_tag_frontmatter(vault, vault_dir):
    (vault_dir / "Scalar.md").write_text(
        "---\ntags: solo\n---\n\nBody.\n", encoding="utf-8"
    )
    assert _paths(iter_candidate_notes(vault, tags=["solo"])) == ["Scalar.md"]


def test_candidates_folder_filter(vault):
    rels = _paths(iter_candidate_notes(vault, folder="Claude"))
    assert rels
    assert all(rel.startswith("Claude/") for rel in rels)


def test_as_str_list_normalizes():
    assert as_str_list(None) == []
    assert as_str_list("one") == ["one"]
    assert as_str_list(["a", 2]) == ["a", "2"]


def test_body_text_strips_frontmatter():
    assert body_text("---\ntags: [x]\n---\n\n# T\n\nBody.\n") == "\n\n# T\n\nBody.\n"


def test_body_text_without_frontmatter_is_passthrough():
    assert body_text("# T\n\nBody.\n") == "# T\n\nBody.\n"


def test_body_text_unclosed_frontmatter_is_passthrough():
    assert body_text("---\ntags: [x]\nno closing fence\n") == "---\ntags: [x]\nno closing fence\n"


def test_skips_every_dot_directory_not_only_known_names(vault, vault_dir):
    # A three-name allow-list let a stray backup folder reach the index; the
    # rule is the whole dot-entry class now.
    backup = vault_dir / ".notes-backup-2026"
    backup.mkdir()
    (backup / "Sentinel ESG.md").write_text(
        "---\ntags: [project, esg]\n---\n\nESG incident ingestion pipeline copy.\n",
        encoding="utf-8",
    )
    assert _paths(iter_candidate_notes(vault, tags=["esg"])) == [
        "Projects/Sentinel ESG.md"
    ]
    assert not any(rel.startswith(".") for _p, rel in iter_note_files(vault))


def test_skips_nested_dot_directory(vault, vault_dir):
    nested = vault_dir / "Projects" / ".archive"
    nested.mkdir()
    (nested / "Old.md").write_text("ingestion pipeline archived\n", encoding="utf-8")
    assert "Projects/.archive/Old.md" not in _paths(iter_candidate_notes(vault))
    assert "Projects/.archive/Old.md" not in [
        rel for _p, rel in iter_note_files(vault)
    ]


def test_is_skipped_rule():
    assert is_skipped((".obsidian", "app.json"))
    assert is_skipped(("Claude", ".trash", "Note.md"))
    assert is_skipped((".smart-env", "x", "y.md"))
    assert is_skipped((".hidden note.md",))
    assert not is_skipped(("Projects", "Sentinel ESG.md"))
    assert not is_skipped(("_Attachments", "img.png"))
    assert not is_skipped(("00 - Maps of Content", "Home.md"))
    assert not is_skipped(())
