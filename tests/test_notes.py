from datetime import datetime

import yaml

from tesseract_mcp import notes

NOW = datetime(2026, 7, 5, 14, 30)


def test_safe_filename_strips_illegal_chars():
    assert notes.safe_filename('a/b\\c:d*e?f"g<h>i|j') == "abcdefghij"


def test_safe_filename_empty_falls_back():
    assert notes.safe_filename("///") == "untitled"


def test_safe_filename_strips_brackets_and_newlines():
    assert notes.safe_filename("Fix ]] the [[bug") == "Fix the bug"
    assert notes.safe_filename("line1\nline2\ttab") == "line1 line2 tab"


def test_make_frontmatter_fields():
    fm = notes.make_frontmatter(project="sentinel", tags=["esg"], created=NOW)
    assert fm.startswith("---\n")
    assert "created: 2026-07-05 14:30" in fm
    assert "agent: claude" in fm
    assert "project: sentinel" in fm
    assert "- esg" in fm
    assert fm.endswith("---\n\n")


def test_log_session_creates_note_and_updates_index(vault):
    rel = notes.log_session(
        vault, "LiveSync setup", "We configured CouchDB.",
        project="tesseract", tags=["infra"], now=NOW,
    )
    assert rel == "Claude/Sessions/2026-07-05 LiveSync setup.md"
    body = vault.read(rel)
    assert "agent: claude" in body
    assert "We configured CouchDB." in body
    index = vault.read("Claude/Index.md")
    assert "[[2026-07-05 LiveSync setup]]" in index
    assert "tesseract" in index


def test_capture_appends_timestamped_bullet(vault):
    rel = notes.capture(vault, "check R2 pricing", now=NOW)
    assert rel == "Claude/Inbox/2026-07-05.md"
    assert vault.read(rel) == "- 14:30 check R2 pricing\n"
    notes.capture(vault, "second thought", now=NOW)
    assert vault.read(rel).count("- 14:30") == 2


def test_upsert_concept_creates_new(vault):
    rel = notes.upsert_concept(vault, "DuckDNS", "Free dynamic DNS.", now=NOW)
    assert rel == "Claude/Concepts/DuckDNS.md"
    body = vault.read(rel)
    assert body.startswith("---\n")
    assert "# DuckDNS" in body
    assert "Free dynamic DNS." in body


def test_upsert_concept_appends_to_existing(vault):
    notes.upsert_concept(vault, "CouchDB", "Used by LiveSync.", now=NOW)
    body = vault.read("Claude/Concepts/CouchDB.md")
    assert "Document database used for LiveSync." in body  # original preserved
    assert "## Update 2026-07-05" in body
    assert "Used by LiveSync." in body


def test_log_session_same_title_same_day_numbers_the_note(vault):
    first = notes.log_session(
        vault, "Standup", "First.", project="p", tags=[], now=NOW
    )
    second = notes.log_session(
        vault, "Standup", "Second.", project="p", tags=[], now=NOW
    )
    assert first == "Claude/Sessions/2026-07-05 Standup.md"
    assert second == "Claude/Sessions/2026-07-05 Standup 2.md"
    assert "First." in vault.read(first)
    assert "Second." in vault.read(second)
    index = vault.read("Claude/Index.md")
    assert "[[2026-07-05 Standup]]" in index
    assert "[[2026-07-05 Standup 2]]" in index


def test_upsert_concept_matches_name_case_insensitively(vault):
    rel = notes.upsert_concept(vault, "couchdb", "Case-folded note.", now=NOW)
    assert rel == "Claude/Concepts/CouchDB.md"
    body = vault.read("Claude/Concepts/CouchDB.md")
    assert "Document database used for LiveSync." in body
    assert "## Update 2026-07-05" in body
    assert "Case-folded note." in body
    concepts = [
        p.name for p in vault.resolve("Claude/Concepts").iterdir() if p.suffix == ".md"
    ]
    assert len([n for n in concepts if n.casefold() == "couchdb.md"]) == 1


def test_make_frontmatter_yaml_roundtrip_with_special_chars():
    fm = notes.make_frontmatter(
        project="esg: incident [pipeline]", tags=['odd"tag', "b:c"], created=NOW
    )
    block = fm.removeprefix("---\n").rsplit("---\n\n", 1)[0]
    meta = yaml.safe_load(block)
    assert meta["project"] == "esg: incident [pipeline]"
    assert meta["tags"] == ['odd"tag', "b:c"]


ORIGIN = "t3-thread://v2/env/thread:abc"


def _fm(text):
    from tesseract_mcp.search import parse_frontmatter

    return parse_frontmatter(text)


def test_log_session_stamps_origin_exactly(vault):
    rel = notes.log_session(
        vault, "O", "body", project="p", tags=[], now=NOW, origin=ORIGIN
    )
    assert _fm(vault.read(rel))["origin"] == ORIGIN


def test_log_session_without_origin_has_no_key(vault):
    rel = notes.log_session(vault, "N", "body", project="p", tags=[], now=NOW)
    text = vault.read(rel)
    assert "origin" not in _fm(text)
    assert text.startswith(notes.make_frontmatter(project="p", created=NOW))


def test_upsert_concept_new_is_stamped(vault):
    rel = notes.upsert_concept(vault, "Fresh", "x", now=NOW, origin=ORIGIN)
    assert _fm(vault.read(rel))["origin"] == ORIGIN


def test_upsert_concept_existing_keeps_frontmatter(vault):
    rel = notes.upsert_concept(vault, "Old", "x", now=NOW)
    before = vault.read(rel).split("---\n")[1]
    notes.upsert_concept(vault, "Old", "y", now=NOW, origin="other:1")
    text = vault.read(rel)
    assert text.split("---\n")[1] == before
    assert "origin" not in _fm(text)
    assert "other:1" not in text
