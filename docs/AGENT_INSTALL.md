# Installing Tesseract for a user (agent procedure)

Read this when asked to set Tesseract up for someone. Code changes follow README "Contributing". Procedure adapted from Enigma's `AGENTS.md`.

## Consent rule (split by risk)

One "yes" covers one step on one vault.

- **Client config** (`mcp_sync`, `skill_sync`, `hook_sync --install`, any other client config edit): run `--check`, show the output, print the exact command, and the **user** runs it. Run the live step never, even after a yes.
- **Vault writes** (`provision`, `scripts/install_conventions.py`): show the `--check` output, ask, run only after an explicit yes for that step on that vault.
- **Graph build** (costs one model call per note): ask, then run.

Exit 1 from a `--check` means work is pending, not failure. Show the user a report, never a config file's contents (they can hold keys). A declined step is skipped, not retried.

## 1. Ask first

- ASK the absolute vault path. Take it only from the user, never from `TESSERACT_VAULT_PATH`, the hook default, or a path written in `AGENTS.md`.
- ASK which clients will use it (Claude Code, Claude Desktop, Codex, other).
- ASK whether the vault is already synced another way (skip LiveSync if so).

Done when you hold a vault path the user typed and a client list.

## 2. Preflight (read-only)

Python 3.11+, the vault folder exists, `claude` or `codex` on `PATH`, `uvx` if the fetch and arxiv servers are wanted, working directory is a source checkout. Stop with a clear message on any failure; ASK before installing anything missing.

## 3. Install the package

`python -m venv .venv`, then `pip install -e .` in the checkout. Tell the user first: this pulls torch, and the embedding model downloads on first use.

## 4. Vault provisioning (vault write)

1. `python -m tesseract_mcp.provision <vault> --check`; show the report.
2. ASK. After a yes, run it without `--check`.

It writes 14 plugins into `.obsidian/plugins`, settings only where none exist, the `Claude/` tree and vault-root `AGENTS.md` and `CLAUDE.md`; existing files stay untouched.

If the user declines plugins: ASK, then `python scripts/install_conventions.py <vault>`. It has no `--check`; it creates whichever of `Claude/{Inbox,Sessions,Concepts,Answers,Digests}`, `Claude/{README,Index,Decisions}.md`, and vault-root `CLAUDE.md` and `AGENTS.md` are missing (see `tesseract_mcp.conventions.install`) and prints a `Created:` line. Describe that to the user before asking.

## 5. MCP registration (client config: user runs)

1. Back up `~/.claude.json` to `~/.claude.json.bak-<timestamp>`.
2. `python -m tesseract_mcp.mcp_sync --check --vault <vault>`; show the report (exit 1 means missing or drifted).
3. Print `python -m tesseract_mcp.mcp_sync --vault <vault>` for the user to run.

Other clients: show the README "MCP client configuration" snippet; the user makes the edit. Use one `TESSERACT_AGENT_NAME` per agent.

## 6. Skills (client config: user runs)

`python -m tesseract_mcp.skill_sync --check`; show the output, print `python -m tesseract_mcp.skill_sync` for the user. `--force` overwrites drifted skills; print it only if the user separately asks for it.

## 7. Hooks (Claude Code only, optional; client config: user runs)

`python -m tesseract_mcp.hook_sync --check`; show the output, print `python -m tesseract_mcp.hook_sync --install` for the user. The hooks need `TESSERACT_VAULT_PATH` set where they run.

## 8. Build the graph

ASK first: indexing makes one model call per note. After a yes, `python -m tesseract_mcp.indexer <vault>` (or the `index_brain` tool).

## 9. Librarian preview

`python -m tesseract_mcp.librarian <vault> --dry-run`; show the report. Schedule nothing unless asked.

## 10. Hand back

List steps run, skipped, and declined. Name what only the user can do: run the printed commands, open the vault in Obsidian, turn off Restricted Mode, finish LiveSync, restart agents.

## Trial runs

Use a throwaway vault and a scratch `TESSERACT_STATE_DIR`; an agent shell may carry live values, so set both by hand. Check mode of client-config tools accepts scratch paths: `mcp_sync --config <file>`, `skill_sync --dest <dir>`, `hook_sync` via `CLAUDE_SETTINGS_PATH`.
