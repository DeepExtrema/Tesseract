# Changelog

## 0.2.0 (unreleased)

### Highlights

- **Search is faster.** The server keeps an index of the vault in memory and re-reads only the notes that changed. Before, it read and tokenized every note on each query. The first search after a start takes about 0.7 s longer, because the server builds the index then.
- **Search puts the right note first more often.** The server now adds the keyword score and the vector score of each note, where it used to merge two ranked lists. A word repeated many times in a note no longer outranks the note that is about the topic. An entity note counts half as much as the notes it was extracted from. On the maintainer's vault, the right note came first for 85% of 300 held-out questions, up from 38%, and was in the top five for 97%, up from 76%. A model wrote and labelled those questions, and no person has audited them.
- **A failed move no longer leaves a note without an undo record.** The organizer checks the undo journal before it moves anything, and puts a move back if the journal write fails. `undo_move` is refused while a sweep is running. The index manifest and the journal are written to a temporary file and swapped in.
- **`consolidate_graph` no longer deletes note text.** Merging aliases into an entity note that had a horizontal rule but no frontmatter used to delete everything above the rule.
- **Agents and the Librarian move notes only within limits.** An agent's `organize_vault(apply=True)` moves nothing unless you set `TESSERACT_AGENT_MOVES_PER_DAY`. The Librarian moves at most 10 notes per run and 20 per day unless you raise its limits.
- **Filing proposals wait for your answer in one list.** `Claude/Organizer.md` holds every open proposal once, in place of a new section per sweep. Tick a proposal to dismiss it, and the organizer never moves that note to that folder. A proposal unanswered for 30 days moves to "Needs resolution" and appears in the digest.
- **An agent can install Tesseract for you.** `docs/AGENT_INSTALL.md` is a procedure for that agent. It asks for the vault path, shows each vault change before it writes, and leaves changes to your agent client's configuration for you to run.

### Upgrade notes

- Two defaults changed: agent moves are off until you set `TESSERACT_AGENT_MOVES_PER_DAY`, and the Librarian is limited to 10 moves per run and 20 per day.
- The `rank-bm25` dependency is removed. Reinstall the package with `pip install -e .`.
- The first search writes fallback vectors to `note_vectors.npz` in the state folder. The older `fallback_embeddings.json` stays in place.

### All changes

- Mentions of notes whose name contains `]` are no longer dropped from the graph, so `related_notes` finds them.
- Every dot-folder in a vault is hidden from search, embeddings, tasks, sheets, the graph and the mover.
- A search hit on a note with an empty body shows an excerpt from its frontmatter.
- Retired and merged entity notes no longer appear in search and are never embedded.
- Derived `Claude/Graph` entity notes score half as much as source notes in search, so a clearly better entity note can still win; a note whose name or alias is the query is listed first; `search_brain` takes `prefer_source` (default true) to turn the weighting off.
- A word that repeats in a note adds less to its keyword score each time instead of growing with its count.
- The words of a note's file name are part of its keyword score, so a note can be found by its name even when its text never repeats it.
- Keyword and vector scores are scaled to 0-1 over the candidates and added with equal weight, replacing rank fusion.
- `find_entity` takes a `limit` and returns the best matches first: exact, then prefix, then substring, then most mentions.
- Search answers from an in-process note index that re-reads only changed notes, scores with numpy BM25L, and keeps fallback vectors in `note_vectors.npz`; the `rank-bm25` dependency is removed.
- The read-only stale-vector check no longer converts the old JSON vector store.
- Added `scripts/bench_mcp.py`, `scripts/bench_workload.py` and `scripts/search_snapshot.py` for measuring search latency and comparing results between two checkouts.
- `python -m tesseract_mcp.evals` also reports success@1, a 95% interval for each success rate and for MRR, and success per `kind` when the golden file gives one.
- `consolidate_graph` no longer deletes text from an entity note that has a horizontal rule but no frontmatter, or keys from one whose frontmatter is not valid YAML.
- `related_notes` stops walking once there is nothing new to reach, so a very large `hops` returns at once.
- An applying sweep checks that the undo journal can be opened before it moves anything, and puts a move back if writing its journal entry fails.
- `undo_move` takes the caretaker lock and is refused while a sweep is running.
- The index manifest and the undo journal are written to a temporary file and swapped in.
- A moved or restored note keeps its stored vector under its new path, and `undo_move` rebuilds `graph.db`.
- The README has a "Provenance and reuse" section.
- **Default changed:** `organize_vault(apply=True)` from an agent moves nothing unless `TESSERACT_AGENT_MOVES_PER_DAY` is set. The budget is counted from this machine's undo journal; a dry-run still lists every candidate.
- **Default changed:** the Librarian moves at most 10 notes per run and 20 per day (`--max-moves`, `--max-moves-per-day`; both accept `unlimited`). The report says how many moves a limit held back.
- Filing proposals live in one managed block in `Claude/Organizer.md` in place of a new section per sweep. Ticking a line dismisses it; a proposal unanswered for 30 days moves to "Needs resolution" and is reported by `organize_vault`, `librarian_status` and the digest.
- `log_session` and `upsert_concept` take an optional `origin`, stored in the frontmatter of new session notes and new concepts and never interpreted by the server.
- Added `docs/AGENT_INSTALL.md`, a procedure for an agent installing Tesseract for a user: vault writes after a yes per step, client configuration changes run by the user.

## 0.1.0

First public version.
