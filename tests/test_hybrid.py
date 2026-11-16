import pytest

from tesseract_mcp.hybrid import _excerpt, hybrid_search, is_derived


class FakeEmbedder:
    """Maps note text to a deterministic vector by simple keyword presence,
    so 'semantic' matches can be tested without a real model."""

    VOCAB = ["logistics", "cooking", "finance"]

    def embed_batch(self, texts):
        return [
            [1.0 if word in t.lower() else 0.0 for word in self.VOCAB]
            for t in texts
        ]


class FakeSemanticEmbedder:
    """Embeds 'owe money' queries and 'invoice' notes into the same region
    of vector space despite zero shared tokens — the paraphrase case the
    vector half of hybrid search exists for."""

    def embed_batch(self, texts):
        out = []
        for t in texts:
            lower = t.lower()
            if "owe" in lower or "invoice" in lower:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


def test_hybrid_search_semantic_match_without_shared_tokens(vault, vault_dir):
    (vault_dir / "Contractors.md").write_text(
        "Outstanding invoices from contractors need review.\n", encoding="utf-8"
    )
    hits = hybrid_search(
        vault, vault.root, FakeSemanticEmbedder(), "who do I owe money to"
    )
    assert "Contractors.md" in [h.path for h in hits]


def test_hybrid_search_exact_keyword_match(vault, vault_dir):
    (vault_dir / "Logistics.md").write_text(
        "This note is about supply chain logistics operations.\n",
        encoding="utf-8",
    )
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "logistics")
    assert "Logistics.md" in [h.path for h in hits]


def test_hybrid_search_respects_tag_filter(vault):
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "e", tags=["esg"])
    assert [h.path for h in hits] == ["Projects/Sentinel ESG.md"]


def test_hybrid_search_respects_limit(vault, vault_dir):
    for i in range(5):
        (vault_dir / f"Note{i}.md").write_text("shared keyword here", encoding="utf-8")
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "shared", limit=2)
    assert len(hits) == 2


def test_hybrid_search_no_match_returns_empty(vault):
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "zzzznomatch")
    assert hits == []


def test_substring_signal_only_when_bm25_empty(vault, vault_dir):
    # "aaa.md" contains the query only inside another word (substring match,
    # not a token match); "zzz.md" contains it as a real token. When BM25 has
    # results, substring matching stays out of the ranking, so aaa.md must not
    # appear at all.
    (vault_dir / "aaa.md").write_text(
        "an important announcement was made\n", encoding="utf-8"
    )
    (vault_dir / "zzz.md").write_text(
        "the port of hamburg is busy\n", encoding="utf-8"
    )
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "port")
    paths = [h.path for h in hits]
    assert "zzz.md" in paths       # real BM25 token match
    assert "aaa.md" not in paths   # substring-only; signal gated off


FRONTMATTER_NOTE = "---\ntags: [x]\n---\n\n# Title Line\n\nReal content here.\n"


def test_excerpt_semantic_only_hit_returns_first_body_line_not_delimiter():
    assert _excerpt(FRONTMATTER_NOTE, "Notes/N.md", "unrelated-query") == "# Title Line"


def test_excerpt_line_match_never_returns_frontmatter_line():
    text = "---\ntags: [billing]\n---\n\nInvoice for the billing cycle.\n"
    assert _excerpt(text, "Notes/N.md", "billing") == "Invoice for the billing cycle."


# Notes imported from a database tool keep every field in frontmatter and have
# an empty body. BM25 and the embeddings rank the whole file, so these are
# ranked hits with nothing in the body to preview.
FRONTMATTER_ONLY_NOTE = (
    "---\nCategory: Space\nEntity Type: Customer\n"
    "Notes: Smallsat constellation operator with 200+ satellites.\n---\n"
)


def test_excerpt_frontmatter_field_match_when_body_is_empty():
    assert _excerpt(FRONTMATTER_ONLY_NOTE, "Companies/Planet Labs.md", "satellites") == (
        "Notes: Smallsat constellation operator with 200+ satellites."
    )


def test_excerpt_frontmatter_only_note_falls_back_to_first_field():
    assert _excerpt(FRONTMATTER_ONLY_NOTE, "Companies/Planet Labs.md", "zzz") == (
        "Category: Space"
    )


def test_excerpt_body_match_beats_frontmatter_match():
    text = "---\nCategory: billing\n---\n\nInvoice for the billing cycle.\n"
    assert _excerpt(text, "Notes/N.md", "billing") == "Invoice for the billing cycle."


def test_excerpt_empty_note_returns_empty():
    assert _excerpt("", "Notes/N.md", "zzz") == ""
    assert _excerpt("---\n---\n", "Notes/N.md", "zzz") == ""


def test_excerpt_caps_long_frontmatter_field():
    text = "---\nNotes: " + "satellite " * 60 + "\n---\n"
    got = _excerpt(text, "Companies/X.md", "satellite")
    assert got.startswith("Notes: satellite") and len(got) == 200


def test_excerpt_title_match_unchanged():
    assert _excerpt(FRONTMATTER_NOTE, "Claude/Sessions/Weekly Review.md", "weekly") == "(title match)"


def test_hybrid_search_semantic_hit_excerpt_is_not_frontmatter(vault, vault_dir):
    (vault_dir / "Contractors.md").write_text(
        "---\ntags: [money]\n---\n\nOutstanding invoices from contractors.\n",
        encoding="utf-8",
    )
    hits = hybrid_search(
        vault, vault.root, FakeSemanticEmbedder(), "who do I owe money to"
    )
    by_path = {h.path: h.excerpt for h in hits}
    assert by_path["Contractors.md"] == "Outstanding invoices from contractors."


def test_substring_fallback_still_works_when_bm25_empty(vault):
    # Single-character query: BM25's [a-z0-9]+ tokenizer yields no token
    # matches, so the substring fallback must still return results.
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "e", tags=["esg"])
    assert [h.path for h in hits] == ["Projects/Sentinel ESG.md"]


def test_folder_scoped_hit_on_a_frontmatter_only_note_has_an_excerpt(vault, vault_dir):
    # Reported as "folder-scoped search returns empty excerpts": the folders
    # that were scoped into happen to hold imported, body-less notes.
    (vault_dir / "Companies").mkdir()
    (vault_dir / "Companies" / "Planet Labs.md").write_text(
        FRONTMATTER_ONLY_NOTE, encoding="utf-8"
    )
    hits = hybrid_search(
        vault, vault.root, FakeEmbedder(), "satellites", folder="Companies"
    )
    by_path = {h.path: h.excerpt for h in hits}
    assert by_path["Companies/Planet Labs.md"] == (
        "Notes: Smallsat constellation operator with 200+ satellites."
    )


def test_hybrid_search_excludes_a_retired_tombstone(vault, vault_dir):
    # A tombstone is short, so BM25 would rank it above the note it was
    # derived from. It must not be a candidate at all.
    graph = vault_dir / "Claude" / "Graph" / "Topics"
    graph.mkdir(parents=True)
    (graph / "Logistics.md").write_text(
        "---\nentity: topic\nretired: 2026-10-19 21:00\n---\n\n"
        "# Logistics\n\nRetired: orphaned — no mentions or relations.\n",
        encoding="utf-8",
    )
    (vault_dir / "Freight.md").write_text(
        "# Freight\n\nLogistics planning for inbound freight.\n", encoding="utf-8"
    )
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "logistics")]
    assert "Claude/Graph/Topics/Logistics.md" not in paths
    assert "Freight.md" in paths


def _graph_vault(vault_dir):
    """A vault where derived entity notes outnumber the one source note and
    would otherwise win on BM25: they are shorter and name the term twice."""
    graph = vault_dir / "Claude" / "Graph" / "Organizations"
    graph.mkdir(parents=True)
    for n in range(5):
        (graph / f"Logistics Co {n}.md").write_text(
            f"---\nentity: organization\naliases: [LC{n}]\n---\n\n"
            f"# Logistics Co {n}\n\nA logistics company.\n",
            encoding="utf-8",
        )
    (vault_dir / "Freight Review.md").write_text(
        "# Freight Review\n\nNotes on our logistics spend this quarter, "
        "covering carriers, lanes and warehouse logistics.\n",
        encoding="utf-8",
    )
    return vault_dir


def test_source_notes_rank_above_derived_entity_notes_by_default(vault, vault_dir):
    # On a mature vault most notes are derived entity notes, so they take most of
    # the unscoped top-10 slots and crowd out the sources that answer the query.
    _graph_vault(vault_dir)
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "logistics", limit=6)]
    assert paths[0] == "Freight Review.md"
    assert all(p.startswith("Claude/Graph/") for p in paths[1:])


def test_prefer_source_false_restores_the_mixed_ranking(vault, vault_dir):
    _graph_vault(vault_dir)
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "logistics", limit=6,
        prefer_source=False)]
    assert "Freight Review.md" in paths
    assert paths[0].startswith("Claude/Graph/")


def test_a_query_that_is_an_entity_name_keeps_the_entity_note_on_top(vault, vault_dir):
    # Demoting every derived note cost entity-by-name lookup outright. A
    # query that *is* the name is asking for the definition.
    _graph_vault(vault_dir)
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "Logistics Co 3", limit=6)]
    assert paths[0] == "Claude/Graph/Organizations/Logistics Co 3.md"


def test_an_entity_alias_also_exempts_the_note_from_demotion(vault, vault_dir):
    _graph_vault(vault_dir)
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "LC3", limit=6)]
    assert paths[0] == "Claude/Graph/Organizations/Logistics Co 3.md"


def test_a_question_about_an_entity_still_prefers_the_source_notes(vault, vault_dir):
    # Equality, not containment: the name appears inside the query, but the
    # notes that answer a question about it are the sources.
    _graph_vault(vault_dir)
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(),
        "what do we spend on logistics each quarter", limit=6)]
    assert paths[0] == "Freight Review.md"


def test_graph_scoped_search_ranking_is_untouched(vault, vault_dir):
    # Folder scoping into the graph is a deliberate ask for entity notes.
    # A homogeneous candidate set must come back in relevance order.
    _graph_vault(vault_dir)
    args = (vault, vault.root, FakeEmbedder(), "logistics")
    scoped = [h.path for h in hybrid_search(
        *args, folder="Claude/Graph", limit=6)]
    mixed = [h.path for h in hybrid_search(
        *args, folder="Claude/Graph", limit=6, prefer_source=False)]
    assert scoped == mixed
    assert len(scoped) == 5


def test_is_derived_does_not_match_a_sibling_of_the_graph_folder():
    assert not is_derived("Claude/Graphics/Chart.md")
    assert not is_derived("Claude/Sessions/2026-10-19 Work.md")
    assert is_derived("Claude/Graph/People/Alice Zhang.md")


def test_a_source_note_outranks_an_equally_matching_entity_note(vault, vault_dir):
    text = "Ledger reconciliation steps for the month end close.\n"
    graph = vault_dir / "Claude" / "Graph" / "Topics"
    graph.mkdir(parents=True)
    (graph / "Ledger Reconciliation.md").write_text(text, encoding="utf-8")
    (vault_dir / "Close Checklist.md").write_text(text, encoding="utf-8")
    query = "ledger reconciliation month end"
    paths = [h.path for h in hybrid_search(vault, vault.root, FakeEmbedder(), query)]
    assert paths[:2] == ["Close Checklist.md", "Claude/Graph/Topics/Ledger Reconciliation.md"]
    mixed = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), query, prefer_source=False)]
    assert mixed[0] == "Claude/Graph/Topics/Ledger Reconciliation.md"


def test_an_exact_name_comes_first_and_a_source_note_precedes_an_entity_note(vault, vault_dir):
    graph = vault_dir / "Claude" / "Graph" / "Projects"
    graph.mkdir(parents=True)
    (graph / "Atlas.md").write_text("# Atlas\n\nA project.\n", encoding="utf-8")
    (vault_dir / "Atlas.md").write_text("# Atlas\n\nA project.\n", encoding="utf-8")
    (vault_dir / "Atlas Notes.md").write_text(
        "atlas atlas atlas atlas atlas\n", encoding="utf-8"
    )
    paths = [h.path for h in hybrid_search(vault, vault.root, FakeEmbedder(), "Atlas")]
    assert paths[:3] == ["Atlas.md", "Claude/Graph/Projects/Atlas.md", "Atlas Notes.md"]


def test_a_name_with_punctuation_the_filename_lost_still_matches(vault, vault_dir):
    # notes.safe_filename removes ':' from an entity name, so the note on disk
    # never matches the name the caller types character for character.
    graph = vault_dir / "Claude" / "Graph" / "Projects"
    graph.mkdir(parents=True)
    (graph / "Project X Phase 2.md").write_text(
        "---\nentity: project\n---\n\n# Project X: Phase 2\n", encoding="utf-8"
    )
    (vault_dir / "Status.md").write_text(
        "project x phase 2 project x phase 2 project x phase 2\n", encoding="utf-8"
    )
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "Project X: Phase 2")]
    assert paths[0] == "Claude/Graph/Projects/Project X Phase 2.md"


def test_a_renamed_note_is_found_by_its_new_name(vault, vault_dir):
    (vault_dir / "Old Name.md").write_text("some text\n", encoding="utf-8")
    (vault_dir / "Mentions.md").write_text("new name new name new name\n", encoding="utf-8")
    hybrid_search(vault, vault.root, FakeEmbedder(), "New Name")
    (vault_dir / "Old Name.md").rename(vault_dir / "New Name.md")
    paths = [h.path for h in hybrid_search(vault, vault.root, FakeEmbedder(), "New Name")]
    assert paths[0] == "New Name.md"


def test_a_note_named_in_a_folder_scope_other_than_its_own_is_not_pinned(vault, vault_dir):
    (vault_dir / "Projects" / "Atlas.md").write_text("project\n", encoding="utf-8")
    (vault_dir / "Daily Atlas.md").write_text("atlas\n", encoding="utf-8")
    paths = [h.path for h in hybrid_search(
        vault, vault.root, FakeEmbedder(), "Atlas", folder="Claude")]
    assert "Projects/Atlas.md" not in paths


def test_a_repeated_word_does_not_beat_a_title_match(vault, vault_dir):
    (vault_dir / "Budget Review.md").write_text("Budget.\n", encoding="utf-8")
    (vault_dir / "Scratch.md").write_text("budget " * 5 + "filler " * 30, encoding="utf-8")
    # "budget" is not the note's whole name, so only the keyword score ranks it.
    paths = [h.path for h in hybrid_search(vault, vault.root, FakeEmbedder(), "budget")]
    assert paths[0] == "Budget Review.md"


def test_a_search_scoped_to_one_note_returns_it(vault):
    hits = hybrid_search(vault, vault.root, FakeEmbedder(), "pipeline", tags=["esg"])
    assert [h.path for h in hits] == ["Projects/Sentinel ESG.md"]
