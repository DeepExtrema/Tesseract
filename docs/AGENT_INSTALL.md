# Installing Tesseract for a user (agent procedure)

Read this when asked to set Tesseract up for someone. Code changes follow README "Contributing".

## Consent rule (split by risk)

One "yes" covers one step on one vault.

- **Client config** (`mcp_sync`, `skill_sync`, `hook_sync --install`, any other client config edit, including setting environment variables in a client's settings): run `--check`, show the output, print the exact command, and the **user** runs it. Never run the live step yourself, even after a yes.
- **Vault writes** (`provision`, `scripts/install_conventions.py`): show what the step will write (the `--check` report for `provision`; for `install_conventions.py`, which has no `--check`, the list in step 4), ask, and run only after an explicit yes for that step on that vault.
- **Model calls** (graph build, and any Librarian run once a graph exists): ask, then run.

Exit 1 from a `--check` means work is pending, not failure; exit 2 means the command aborted and changed nothing. In `skill_sync --check` output, `installed` means "would install". Show the user a report, never a config file's contents (they can hold keys). A declined step is skipped, not retried.

`<py>` below is the checkout's venv interpreter: `.venv/bin/python` on macOS and Linux, `.venv\Scripts\python` on Windows. Use it in every command you run and every command you print.

## 1. Ask first

- ASK the absolute vault path. Take it only from the user, never from `TESSERACT_VAULT_PATH`, the hook default, or a path written in `AGENTS.md`.
- ASK which clients will use it (Claude Code, Claude Desktop, Codex, other).
- ASK whether the vault is already synced another way (skip LiveSync if so).

Done when you hold a vault path the user typed and a client list.

## 2. Preflight (read-only)

Python 3.11+, the vault folder exists, `claude` or `codex` on `PATH`, `uvx` if the fetch and arxiv servers are wanted, working directory is a source checkout. Stop with a clear message on any failure; ASK before installing anything missing.

## 3. Install the package

`python3 -m venv .venv` (Windows: `py -m venv .venv`), then `<py> -m pip install -e .` in the checkout. Tell the user first: this pulls torch, and the embedding model downloads on first use.

## 4. Vault provisioning (vault write)

1. `<py> -m tesseract_mcp.provision <vault> --check`; show the report. It lists plugin versions only.
2. Before asking, tell the user what a live run also writes: it installs the 14 pinned plugins into `.obsidian/plugins` and overwrites any installed plugin whose version differs from the pin (this can be a downgrade), adds their ids to `.obsidian/community-plugins.json` (existing entries stay), writes each plugin's `data.json` and `.smart-env/smart_env.json` only where none exists, and adds the `Claude/` tree and vault-root `AGENTS.md` and `CLAUDE.md` where missing.
3. ASK. After a yes, run it without `--check`.

If the user declines plugins: describe what `scripts/install_conventions.py` creates, ASK, then `<py> scripts/install_conventions.py <vault>`. It has no `--check`; it creates whichever of `Claude/{Inbox,Sessions,Concepts,Answers,Digests}`, `Claude/{README,Index,Decisions}.md`, and vault-root `CLAUDE.md` and `AGENTS.md` are missing (see `tesseract_mcp.conventions.install`) and prints a `Created:` line.

## 5. MCP registration (client config: user runs)

1. `<py> -m tesseract_mcp.mcp_sync --check --vault <vault>`; show the report (exit 1 means missing or drifted).
2. Print, for the user to run: `cp ~/.claude.json ~/.claude.json.bak-$(date +%Y%m%d-%H%M%S) && <py> -m tesseract_mcp.mcp_sync --vault <vault>` (Windows: `Copy-Item` the file first, then the same `mcp_sync` command).

Other clients: show the README "MCP client configuration" snippet; the user makes the edit. Use one `TESSERACT_AGENT_NAME` per agent.

## 6. Skills (client config: user runs)

`<py> -m tesseract_mcp.skill_sync --check`; show the output, print `<py> -m tesseract_mcp.skill_sync` for the user. `--force` overwrites drifted skills; print it only if the user separately asks for it.

## 7. Hooks (Claude Code only, optional; client config: user runs)

`<py> -m tesseract_mcp.hook_sync --check`; show the output, print `<py> -m tesseract_mcp.hook_sync --install` for the user. The hooks need `TESSERACT_VAULT_PATH` set where they run: tell the user to set it (shell profile or client settings). Do not set it yourself; list it in the hand-back.

## 8. Build the graph

ASK first: indexing makes one model call per note. After a yes, `<py> -m tesseract_mcp.indexer <vault>` (or the `index_brain` tool).

## 9. Librarian preview

Once a graph exists, even `--dry-run` asks a model about duplicate entities, so ASK first. After a yes, `<py> -m tesseract_mcp.librarian <vault> --dry-run`; show the report. Schedule nothing unless asked.

## 10. Hand back

List steps run, skipped, and declined. Name what only the user can do: run the printed commands, set `TESSERACT_VAULT_PATH` for the hooks, open the vault in Obsidian, turn off Restricted Mode, finish LiveSync, restart agents.

## Trial runs

Use a throwaway vault and a scratch `TESSERACT_STATE_DIR`; an agent shell may carry live values, so set both by hand. Check mode of client-config tools accepts scratch paths: `mcp_sync --config <file>`, `skill_sync --dest <dir>`, `hook_sync` via `CLAUDE_SETTINGS_PATH`.
