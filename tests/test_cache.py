import sqlite3

import pytest

from tesseract_mcp import cache
from tesseract_mcp.extractor import Extraction
from tesseract_mcp.graphstore import GraphStore

ACME = {"name": "Acme Corp", "type": "organization", "aliases": ["ACME"], "summary": "A company."}
CHAIN = {"name": "Supply Chain", "type": "domain", "aliases": [], "summary": "Logistics."}
REL = {"from": "Acme Corp", "from_type": "organization", "rel": "operates_in",
       "to": "Supply Chain", "to_type": "domain", "evidence": "logistics"}


@pytest.fixture
def populated(vault, tmp_path):
    store = GraphStore(vault)
    store.apply("Projects/Sentinel ESG.md", Extraction([ACME, CHAIN], [REL]))
    vault.write("Claude/Inbox/interview.md", "Talked to [[Acme Corp]] folks.\n")
    store.apply("Claude/Inbox/interview.md", Extraction([ACME], []))
    db = tmp_path / "graph.db"
    cache.rebuild(vault, db)
    return db


def test_rebuild_creates_tables(populated):
    con = sqlite3.connect(populated)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"entities", "edges", "mentions"} <= names


def test_find_entity_by_name_and_alias(populated):
    got = cache.find_entity(populated, "acme")
    assert got and got[0]["name"] == "Acme Corp" and got[0]["type"] == "organization"
    assert cache.find_entity(populated, "ACME")  # alias
    assert got[0]["mention_count"] == 2
    assert {"rel": "operates_in", "to": "Supply Chain"} in [
        {"rel": e["rel"], "to": e["to"]} for e in got[0]["relations"]
    ]


def test_find_entity_type_filter(populated):
    assert cache.find_entity(populated, "supply", type="domain")
    assert cache.find_entity(populated, "supply", type="person") == []


def test_find_entity_limit_ranks_before_cut(vault, tmp_path):
    store = GraphStore(vault)
    ents = [
        {"name": n, "type": "topic", "aliases": [], "summary": ""}
        for n in ("Blazed", "Zedcorp", "Zed")
    ]
    store.apply("Src/a.md", Extraction(ents, []))
    db = tmp_path / "graph.db"
    cache.rebuild(vault, db)
    names = [r["name"] for r in cache.find_entity(db, "zed")]
    assert names == ["Zed", "Zedcorp", "Blazed"]
    assert [r["name"] for r in cache.find_entity(db, "zed", limit=1)] == ["Zed"]
    assert cache.find_entity(db, "zed", limit=0) == []
    assert cache.find_entity(db, "zed", limit=-3) == []


def test_related_notes_shared_entity(populated, vault):
    got = cache.related_notes(populated, vault, "Claude/Inbox/interview.md", hops=1)
    paths = [r["path"] for r in got]
    assert "Projects/Sentinel ESG.md" in paths
    assert any("Acme Corp" in r["via"] for r in got)


def test_related_notes_excludes_self_and_graph_notes(populated, vault):
    got = cache.related_notes(populated, vault, "Claude/Inbox/interview.md", hops=2)
    paths = [r["path"] for r in got]
    assert "Claude/Inbox/interview.md" not in paths
    assert not any(p.startswith("Claude/Graph/") for p in paths)


def test_stats(populated):
    s = cache.stats(populated)
    assert s["entities"]["organization"] == 1
    assert s["entities"]["domain"] == 1
    assert s["edges"] == 1
    assert s["mentions"] == 3


def test_rebuild_atomic_replaces(populated, vault, tmp_path):
    db = populated
    cache.rebuild(vault, db)  # second rebuild over existing db must not error
    assert cache.find_entity(db, "acme")


def test_same_name_different_type_not_conflated(vault, tmp_path):
    from tesseract_mcp.graphstore import GraphStore

    store = GraphStore(vault)
    store.apply("A.md", Extraction(
        [{"name": "Ops", "type": "domain", "aliases": [], "summary": "domain ops"}],
        [{"from": "Globex", "from_type": "organization", "rel": "operates_in",
          "to": "Ops", "to_type": "domain", "evidence": ""}],
    ))
    store.apply("B.md", Extraction(
        [{"name": "Ops", "type": "topic", "aliases": [], "summary": "topic ops"}], []
    ))
    db = tmp_path / "g.db"
    cache.rebuild(vault, db)
    dom = cache.find_entity(db, "Ops", type="domain")
    top = cache.find_entity(db, "Ops", type="topic")
    assert dom and top
    assert dom[0]["mention_count"] == 1   # only A.md, not conflated with topic
    assert top[0]["mention_count"] == 1   # only B.md
    # the domain has an inbound operates_in edge; the topic has none
    assert top[0]["relations"] == []


def test_related_notes_excludes_self_without_md_suffix(populated, vault):
    got = cache.related_notes(populated, vault, "Claude/Inbox/interview", hops=1)  # no .md
    assert "Claude/Inbox/interview.md" not in [r["path"] for r in got]


def test_note_entity_paths(populated, vault):
    got = cache.note_entity_paths(populated, "Claude/Inbox/interview.md")
    assert got == ["Claude/Graph/Organizations/Acme Corp"]
    assert cache.note_entity_paths(populated, "Nope.md") == []


@pytest.fixture
def hub(vault, tmp_path):
    """A hub graph: `Claude/Inbox/seed.md` shares Acme directly with
    `Projects/Direct.md`; a hop-2 hub domain is mentioned by many notes."""
    store = GraphStore(vault)
    hub_ent = {"name": "Everything", "type": "domain", "aliases": [], "summary": "Hub."}
    rel = {"from": "Acme Corp", "from_type": "organization", "rel": "operates_in",
           "to": "Everything", "to_type": "domain", "evidence": ""}
    vault.write("Claude/Inbox/seed.md", "Seed note about [[Acme Corp]].\n")
    store.apply("Claude/Inbox/seed.md", Extraction([ACME], [rel]))
    vault.write("Projects/Direct.md", "Also about Acme.\n",
                confirm_outside_claude=True)
    store.apply("Projects/Direct.md", Extraction([ACME], []))
    for i in range(6):
        p = f"Projects/Hubbed {i}.md"
        vault.write(p, "About the hub only.\n", confirm_outside_claude=True)
        store.apply(p, Extraction([hub_ent], []))
    db = tmp_path / "graph.db"
    cache.rebuild(vault, db)
    return db


def test_related_notes_ranks_direct_share_above_hub_reach(hub, vault):
    got = cache.related_notes(hub, vault, "Claude/Inbox/seed.md", hops=2)
    # the note sharing Acme directly (hop 1) must outrank every note that is
    # only reachable through the hop-2 hub domain
    assert got[0]["path"] == "Projects/Direct.md"
    assert "Acme Corp" in got[0]["via"]


def test_related_notes_limit_caps_hub_flood(hub, vault):
    got = cache.related_notes(hub, vault, "Claude/Inbox/seed.md", hops=2, limit=3)
    assert len(got) == 3
    assert got[0]["path"] == "Projects/Direct.md"  # strongest survives the cap


def test_related_notes_rejects_negative_limit(hub, vault):
    # a negative slice would silently return all-but-N instead of a cap
    with pytest.raises(ValueError, match="limit"):
        cache.related_notes(hub, vault, "Claude/Inbox/seed.md", limit=-1)


def _write_outside_claude(vault, rel: str, text: str) -> None:
    """Human notes live outside Claude/, which Vault.write guards."""
    path = vault.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


BRACKETED = "Applications/Initech - [2027] Software Engineer.md"
TRAILING_BRACKET = "Applications/Globex - [role not specified].md"


@pytest.mark.parametrize("note_path", [BRACKETED, TRAILING_BRACKET])
def test_mention_of_note_whose_name_contains_brackets(vault, tmp_path, note_path):
    # A "]" in the note name used to end the mention target early, so the row
    # was never written and related_notes reported "no connections".
    store = GraphStore(vault)
    store.apply("Projects/Sentinel ESG.md", Extraction([ACME, CHAIN], [REL]))
    _write_outside_claude(vault, note_path, "Applied via [[Acme Corp]].\n")
    store.apply(note_path, Extraction([ACME], []))
    db = tmp_path / "graph.db"
    cache.rebuild(vault, db)

    assert cache.note_entity_paths(db, note_path) == [
        "Claude/Graph/Organizations/Acme Corp"
    ]
    related = cache.related_notes(db, vault, note_path, hops=1)
    assert "Projects/Sentinel ESG.md" in [r["path"] for r in related]
    assert cache.find_entity(db, "acme")[0]["mention_count"] == 2


def test_bracketed_mention_keeps_its_evidence(vault, tmp_path):
    store = GraphStore(vault)
    _write_outside_claude(vault, BRACKETED, "Applied.\n")
    store.apply(BRACKETED, Extraction([dict(ACME, summary="A company.")], []))
    entity = "Claude/Graph/Organizations/Acme Corp.md"
    store.remove_mention(entity, BRACKETED)
    store.add_mention(entity, BRACKETED, "Recruiter contact for the 2027 req.")
    db = tmp_path / "graph.db"
    cache.rebuild(vault, db)
    con = sqlite3.connect(db)
    rows = con.execute("SELECT note_path, evidence FROM mentions").fetchall()
    con.close()
    assert rows == [(BRACKETED[:-3], "Recruiter contact for the 2027 req.")]
