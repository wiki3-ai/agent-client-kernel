# MCP configuration & slash commands

This document captures how MCP servers and slash commands flow between
the user's notebook, the `agent_client_kernel`, and the ACP agent
(today: `codex-acp`), and the design decisions we've made — including
the constraints `codex-acp` imposes on what we can do client-side.

## Status

| Area | Status |
| --- | --- |
| MCP source merge (user / codex-global / codex-project) | ✅ shipped |
| Per-session enable/disable of preconfigured servers | ✅ shipped |
| `%agent mcp ignore-codex-config on/off` | ✅ shipped |
| `AvailableCommandsUpdate` capture | ✅ shipped |
| `%agent commands` listing | ✅ shipped |
| Client-side slash dispatch / `%agent /name` | ❌ removed — not needed (see below) |
| Bare-cell `/name` invocation | ✅ works — handled server-side by `codex-acp` |
| Session-start MCP probe / tool-failure surfacing | 🟡 not yet |

## Original problem

There were **two MCP configurations in flight** for every session, and
they didn't know about each other:

1. **Agent-side / global.** `~/.codex/config.toml`'s `[mcp_servers.*]`
   (and the project-local equivalent). Loaded by `codex-acp` at startup,
   before the ACP handshake. This is how the `jupyter` MCP server was
   reaching the model in the golden-notebook test even though
   `%agent mcp list` reported nothing.
2. **Kernel-side / per-session.** `KernelState.mcp_servers`, populated by
   `%agent mcp add`, handed to `codex-acp` via the `mcp_servers` arg on
   `new_session` / `resume_session`. `%agent mcp list` only knew about
   these.

The kernel therefore couldn't truthfully answer "which MCP servers does
this session have?" and the user had no way to remove or disable a
globally preconfigured server for a single notebook.

## What we built (MCP)

`MCPServer` now carries `source` and `enabled`:

```python
@dataclass
class MCPServer:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    source: str = "user"        # "user" | "codex-global" | "codex-project"
    enabled: bool = True
```

On `_start_agent()` / session resume, the kernel:

1. Parses `$CODEX_HOME/config.toml` (default `~/.codex/config.toml`) and
   any project-local `.codex/config.toml`, tagging each entry with its
   source.
2. Merges with `state.mcp_servers`. **Precedence: `user` > `codex-project`
   > `codex-global`** (by name).
3. Filters out anything in `state.disabled_preconfigured` or with
   `enabled=False`.
4. Passes the resulting list to `new_session(mcp_servers=...)`.

User-facing surface:

- `%agent mcp list` — shows name, source, enabled.
- `%agent mcp add NAME ...` — adds a `source="user"` entry (overrides
  same-named preconfigured server).
- `%agent mcp remove NAME` — only removes `source="user"` entries; for
  preconfigured ones it suggests `disable`.
- `%agent mcp disable NAME` / `%agent mcp enable NAME` — flip a flag for
  any source.
- `%agent mcp ignore-codex-config on|off` — toggles the global merge for
  the current session (off by default → merge enabled).

There is intentionally **no** path that writes to `~/.codex/config.toml`
from the kernel. Per-session changes live in kernel state only.

## Slash commands — what we learned

ACP has first-class slash commands. The agent sends
`AvailableCommandsUpdate` notifications listing
`AvailableCommand{name, description, input?}` entries. Clients are
expected to surface them so users know what to type.

### How `codex-acp` actually dispatches

After reading `codex-acp/src/thread.rs`, the wire contract is much
simpler than we assumed during design:

- There is **no special invocation primitive** in ACP for slash commands.
  `ContentBlock` is only `Text` / `Image` / `Audio` / `Resource` /
  `ResourceLink`.
- `codex-acp::handle_prompt` calls `extract_slash_command(items)`, which
  inspects **only the first `UserInput::Text`'s first line**, strips a
  leading `/`, and splits on whitespace into `(name, rest)`.
- Recognised built-in names dispatch to dedicated `Op`s:
  `review`, `review-branch <branch>`, `review-commit <sha>`, `init`,
  `compact`, `logout`.
- Anything else (including `/mcp`, `/model`, `/models`) falls through
  to `Op::UserInput` as raw text — the model just sees `/mcp` in chat.
- The list advertised over `AvailableCommandsUpdate` is exactly the six
  built-ins above. The Codex TUI's extra slashes (`/mcp`, `/model`, …)
  are TUI-only and never reach an ACP client.

### Implication for the kernel

The slash command UX is **entirely a function of what the agent
advertises and what it dispatches server-side**. There is nothing useful
for the kernel to do beyond forwarding the cell text.

Concretely:

- A bare cell whose contents start with `/` is sent unchanged through
  `_send_prompt(code)` → `prompt=[text_block(code)]`. `codex-acp` then
  parses the slash exactly as the TUI would. `/init`, `/compact`, etc.
  work this way today.
- The kernel does **not**:
  - intercept `/` in `do_execute` (no warning about "unknown slash"),
  - offer `/` completions in `do_complete`,
  - support `%agent /name` (returns "Unknown subcommand"),
  - try to translate `/mcp` or `/models` into anything — those simply
    aren't codex-acp slash commands.

### What we kept

- `AvailableCommandsUpdate` notifications are captured into
  `state.available_commands` (replace-on-update).
- `%agent commands` lists the latest advertised commands with their
  descriptions, so the user can see what's available without leaving
  the notebook.

### Why `%agent commands` shows fewer entries than the TUI

The TUI implements its own client-side dispatcher with extras like
`/mcp`, `/model`, `/status`, etc. Over ACP, `codex-acp` advertises only
the six it actually handles. Not a kernel bug — an upstream constraint.

## What's still open

- **Session-start MCP probe.** After `new_session`, print one line per
  configured server (`mcp: jupyter ✓ (12 tools)` / `mcp: jupyter ✗ <err>`).
  An earlier prototype emitted unconditional stderr at session start; we
  removed it because it was noisy and unsolicited. The probe should be
  on by default but quiet on success — or gated behind something like
  `ACP_LOG_LEVEL=info`.
- **Tool-failure surface.** Count per-tool failures during a prompt; if
  the same tool name errors ≥ N times in a row, surface a stderr line
  to the cell. This is the signal that would have made the
  `insert_cell` 404 storm in the golden-notebook incident a 30-second
  debug.
- **Upstream: ask codex to advertise more.** If we want `/mcp` and
  friends to "just work" in notebooks, that's a `codex-acp` change
  (add them to `builtin_commands()` and `handle_prompt`'s dispatcher),
  not a kernel change.

## Non-goals

- Writing to `~/.codex/config.toml` from the kernel.
- Reimplementing the Codex TUI's slash-command set client-side. We
  dispatch what the agent advertises; we don't define commands
  ourselves. If a slash isn't recognised by `codex-acp`, the model just
  sees it as text — that's acceptable and predictable.

## Related documents

- [docs/golden-notebook-newlines-postmortem.md](golden-notebook-newlines-postmortem.md)
  — the incident that motivated this work.
- [docs/jupyter-mcp-server-breaks-cell-execution.md](jupyter-mcp-server-breaks-cell-execution.md)
  — why `jupyter-collaboration` is required for cell-write tools.
- [docs/codex-lmstudio-shim.md](codex-lmstudio-shim.md) — the LM Studio
  shim that ran underneath the test.

## References

- `codex-acp` slash dispatch: `extract_slash_command` and
  `handle_prompt` in
  [`zed-industries/codex-acp` `src/thread.rs`](https://github.com/zed-industries/codex-acp/blob/main/src/thread.rs).
- Built-in command list: `ThreadActor::builtin_commands()` in the same
  file — six entries (review, review-branch, review-commit, init,
  compact, logout).
- ACP schema: `acp.schema` has `AvailableCommand`,
  `AvailableCommandsUpdate`, `PromptRequest`, `ContentBlock`
  (`Text` / `Image` / `Audio` / `Resource` / `ResourceLink`) — no
  dedicated slash-command block type.
