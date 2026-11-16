"""What a user sees when the vault changes between two searches, and which
notes a tag/folder filter selects. The index is process-wide, so every test
here searches twice on the same vault."""

import os

import pytest

from tesseract_mcp import corpus
from tesseract_mcp.hybrid import hybrid_search
from tesseract_mcp.search import Hit, iter_candidate_notes
from tesseract_mcp.vault import VaultError


class WordEmbedder:
    """Vectors that only ever match on the word zebra, so the vector lane stays
    out of the way of keyword expectations."""

    def embed_batch(self, texts):
        return [[1.0 if "zebra" in t.lower() else 0.0, 0.0] for t in texts]


def write(vault_dir, rel, text):
    path = vault_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def paths(hits):
    return [h.path for h in hits]


def search(vault, query, tags=None, folder=None):
    """Plain substring match over the candidate notes, in candidate order."""
    q = query.lower()
    return [
        Hit(rel, "")
        for rel, text in iter_candidate_notes(vault, tags, folder)
        if q in rel.lower() or q in text.lower()
    ]


def hybrid_paths(vault, query, **kwargs):
    return paths(hybrid_search(vault, vault.root, WordEmbedder(), query, **kwargs))


def test_edit_add_delete_and_rename_show_up_in_the_next_search(vault, vault_dir):
    write(vault_dir, "Alpha.md", "walrus colony notes\n")
    assert hybrid_paths(vault, "walrus") == ["Alpha.md"]

    write(vault_dir, "Alpha.md", "now about penguins instead\n")
    assert hybrid_paths(vault, "walrus") == []
    assert hybrid_paths(vault, "penguins") == ["Alpha.md"]

    write(vault_dir, "Beta.md", "walrus again\n")
    assert hybrid_paths(vault, "walrus") == ["Beta.md"]

    (vault_dir / "Beta.md").rename(vault_dir / "Gamma.md")
    assert hybrid_paths(vault, "walrus") == ["Gamma.md"]

    (vault_dir / "Gamma.md").unlink()
    assert hybrid_paths(vault, "walrus") == []


def test_rewrite_with_the_same_size_right_after_a_search_is_seen(
    vault, vault_dir, monkeypatch
):
    # On a filesystem with coarse timestamps two writes in a row can leave the
    # stamp identical. Simulate it by making the stamp blind to everything but
    # the size: a note modified moments ago must be read again regardless.
    monkeypatch.setattr(corpus, "file_stamp", lambda st: (st.st_size,))
    note = write(vault_dir, "Alpha.md", "walrus colony\n")
    assert paths(search(vault, "walrus")) == ["Alpha.md"]

    note.write_text("zebras colony\n", encoding="utf-8")
    assert paths(search(vault, "walrus")) == []
    assert paths(search(vault, "zebras")) == ["Alpha.md"]


def test_tag_and_retired_changes_apply_to_the_next_search(vault, vault_dir):
    note = write(vault_dir, "Tagged.md", "---\ntags: [Wildlife]\n---\n\nwalrus\n")
    assert paths(search(vault, "walrus", tags=["wildlife"])) == ["Tagged.md"]
    assert hybrid_paths(vault, "walrus", tags=["WILDLIFE"]) == ["Tagged.md"]

    note.write_text("---\ntags: [pets]\n---\n\nwalrus\n", encoding="utf-8")
    assert paths(search(vault, "walrus", tags=["wildlife"])) == []
    assert hybrid_paths(vault, "walrus", tags=["pets"]) == ["Tagged.md"]

    note.write_text("---\ntags: [pets]\nretired: 2026-10-19\n---\n\nwalrus\n", encoding="utf-8")
    assert hybrid_paths(vault, "walrus") == []
    assert paths(search(vault, "walrus")) == []


def test_candidates_are_ordered_by_path_component_not_joined_string(vault, vault_dir):
    # "Work Log/x.md" < "Work/y.md" as strings (space < slash) but the
    # components compare "Work" < "Work Log".
    write(vault_dir, "Work Log/x.md", "walrus\n")
    write(vault_dir, "Work/y.md", "walrus\n")
    assert paths(search(vault, "walrus")) == ["Work/y.md", "Work Log/x.md"]
    assert [rel for rel, _ in iter_candidate_notes(vault, folder="Work")] == ["Work/y.md"]


def test_folder_filters(vault, vault_dir):
    write(vault_dir, "Work/y.md", "walrus\n")
    write(vault_dir, "Work Log/x.md", "walrus\n")
    write(vault_dir, "Work/deep/z.md", "walrus\n")
    rels = lambda folder: [r for r, _ in iter_candidate_notes(vault, folder=folder)]

    assert rels("Work") == ["Work/deep/z.md", "Work/y.md"]
    assert rels("Work/") == ["Work/deep/z.md", "Work/y.md"]
    assert rels("Work/deep") == ["Work/deep/z.md"]
    assert rels("Nowhere") == []
    assert rels("Work/y.md") == []  # a file is not a folder
    assert rels(".obsidian") == []
    assert rels("") == rels(None)
    with pytest.raises(VaultError):
        rels("../outside")
    with pytest.raises(VaultError):
        hybrid_search(vault, vault.root, WordEmbedder(), "walrus", folder="../outside")


def test_hidden_symlinked_and_uppercase_extension_files_are_not_candidates(
    vault, vault_dir, tmp_path_factory
):
    write(vault_dir, ".stash/Hidden.md", "walrus\n")
    write(vault_dir, "Projects/.draft.md", "walrus\n")
    if os.path.normcase("A") == "A":  # the extension match follows the platform's case rule
        write(vault_dir, "Upper.MD", "walrus\n")
    outside = tmp_path_factory.mktemp("outside")
    (outside / "Linked.md").write_text("walrus\n", encoding="utf-8")
    try:
        (vault_dir / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert paths(search(vault, "walrus")) == []
