import os
import sqlite3
from pathlib import Path

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


@pytest.fixture
def cyclic(vault, tmp_path):
    """Entity cycle A->B->C->A plus a tail C->D. `seed.md` mentions A; one
    note mentions each of B, C, D. From seed: A is hop 1, B and C hop 2
    (the cycle closes back on A), D hop 3; after that the graph is exhausted."""
    store = GraphStore(vault)

    def ent(name):
        return {"name": name, "type": "domain", "aliases": [], "summary": ""}

    def rel(a, b):
        return {"from": a, "from_type": "domain", "rel": "links",
                "to": b, "to_type": "domain", "evidence": ""}

    # edges come from a Claude/Graph/ source, which related_notes never returns
    store.apply("Claude/Graph/cycle-source.md", Extraction(
        [ent("A"), ent("B"), ent("C"), ent("D")],
        [rel("A", "B"), rel("B", "C"), rel("C", "A"), rel("C", "D")]))
    vault.write("Claude/Inbox/seed.md", "About A.\n")
    store.apply("Claude/Inbox/seed.md", Extraction([ent("A")], []))
    for name in ("B", "C", "D"):
        p = f"Projects/Note {name}.md"
        vault.write(p, f"About {name}.\n", confirm_outside_claude=True)
        store.apply(p, Extraction([ent(name)], []))
    db = tmp_path / "graph.db"
    cache.rebuild(vault, db)
    return db


def test_related_notes_huge_hops_terminates(cyclic):
    import subprocess
    import sys
    code = (
        "from pathlib import Path; from tesseract_mcp import cache; "
        f"r = cache.related_notes(Path({str(cyclic)!r}), None, "
        "'Claude/Inbox/seed.md', hops=10**12); print(len(r))"
    )
    # run the module under test, not whatever copy the venv has installed
    src = str(Path(cache.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=10, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "3"


def test_related_notes_cycle_scores_by_hop(cyclic, vault):
    def scores(hops):
        got = cache.related_notes(cyclic, vault, "Claude/Inbox/seed.md", hops=hops)
        return [r["path"] for r in got]

    assert scores(1) == []  # only A is reached; no other note mentions A
    # B and C at hop 2 tie (1/2 each), ordered by path
    assert scores(2) == ["Projects/Note B.md", "Projects/Note C.md"]
    # D joins at hop 3 (1/3), ranked last
    assert scores(3) == ["Projects/Note B.md", "Projects/Note C.md", "Projects/Note D.md"]
    assert scores(50) == scores(3)
