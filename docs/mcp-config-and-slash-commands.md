# MCP configuration & slash commands — design

This document captures what we learned about how MCP servers and slash
commands flow between the user's notebook, the `agent_client_kernel`, and
the ACP agent (today: `codex-acp`), and the plan for making that surface
honest and usable.

## Status quo

There are **two MCP configurations in flight** for every session, and they
don't currently know about each other:

1. **Agent-side / global.** For Codex, `~/.codex/config.toml`'s
   `[mcp_servers.*]` (and the project-local equivalent). Loaded by
   `codex-acp` at startup before the ACP handshake. This is how the
   `jupyter` MCP server was reaching the model in our golden-notebook test,
   even though `%agent mcp list` reported nothing.
2. **Kernel-side / per-session.** `KernelState.mcp_servers`, populated by
   `%agent mcp add`, handed to `codex-acp` via the `mcp_servers` argument
   on `new_session` / `resume_session`. `%agent mcp list` only knows about
   these.

The kernel therefore can't truthfully answer "which MCP servers does this
session have?" and the user has no way to remove or disable a globally
preconfigured server for a single notebook.

### Slash commands

ACP has first-class slash commands. The agent sends
`AvailableCommandsUpdate` notifications listing
`AvailableCommand{name, description, input?}` entries (`/mcp`, `/init`,
… for `codex-acp`). Clients are expected to surface them and invoke them
by name.

Our kernel today:

- ignores `AvailableCommandsUpdate` notifications entirely; and
- forwards every cell body straight into `session/prompt` as text.

When a user typed `/mcp` in a cell, `codex-acp` received it as a literal
prompt string — not as a structured slash-command invocation — and either
ignored it or treated it as a chat message. The TUI's `/mcp` works because
the TUI itself implements client-side slash-command dispatch.

## Goals

1. `%agent mcp list` reflects reality — every server the session can reach,
   and where it came from.
2. The user can disable or override a preconfigured server for one
   notebook without editing global config.
3. Slash commands advertised by the agent are discoverable and invokable
   from the notebook (typing `/mcp` should "just work").
4. We surface MCP/tool failures loudly enough that a future
   "golden-notebook newlines"–class incident is a 30-second debug, not a
   multi-hour one.

## Non-goals

- Editing `~/.codex/config.toml` from the kernel. The kernel reads it,
  never writes it. Per-session changes live in kernel state only.
- Reimplementing the Codex TUI's slash-command set in the kernel. We
  dispatch what the agent advertises; we don't define commands ourselves.

## Plan

### 1. Unify the MCP view — sources & enable/disable

Introduce an explicit `source` and `enabled` on `MCPServer`:

```python
@dataclass
class MCPServer:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    source: Literal[
        "user",            # added via %agent mcp add
        "codex-global",    # ~/.codex/config.toml
        "codex-project",   # ./.codex/config.toml
    ] = "user"
    enabled: bool = True
```

On `_start_agent()` (and `_resume_session()`):

1. Parse `$CODEX_HOME/config.toml` (default `~/.codex/config.toml`) and
   any project-local `.codex/config.toml`. Build `MCPServer` entries
   tagged with the appropriate source.
2. Merge with the user's `state.mcp_servers`. Collision rule: **user wins
   by name**, then `codex-project`, then `codex-global`.
3. Filter `enabled=False`.
4. Pass the resulting list to `new_session(mcp_servers=...)`.

UI changes:

- `%agent mcp list` shows a `source` column.
- `%agent mcp disable NAME` flips `enabled=False` (works for any source).
- `%agent mcp enable NAME` flips it back.
- `%agent mcp remove NAME` only removes `source="user"` entries; for
  others it suggests `disable`.
- `%agent mcp add NAME ...` works as today; explicitly overrides a
  same-named preconfigured server.
- New: `%agent mcp ignore-codex-config on|off` toggles the global merge
  for the current session (off by default = merge enabled).

### 2. Wire `AvailableCommandsUpdate` through

Two pieces inside the kernel:

**a. Capture.** Extend the ACP client implementation to handle
`AvailableCommandsUpdate` and store the latest list on
`state.available_commands: list[AvailableCommand]`. (The notification
arrives at any point during a session and replaces the full list each
time — we just overwrite.)

**b. Dispatch.** In `do_execute`, before magic-command processing, check
for a leading `/`. If the first token matches a known
`AvailableCommand.name`, dispatch through the structured slash-command
path rather than as plain prompt text. Otherwise fall through to the
existing prompt path (preserves "I genuinely want to send a message
starting with `/`" behaviour).

The exact wire format for invocation needs experimental confirmation
against `codex-acp` — ACP's prompt object supports an `_meta` field and
slash commands can carry structured `input`. Two viable shapes:

- Send `session/prompt` with the literal command name as the text plus a
  `_meta` marker indicating it's a registered slash command.
- Send `session/prompt` with a structured slash-command content block.

We'll pick whichever `codex-acp` actually recognises and document it in
this file once verified.

**c. Magic surface.** Add:

- `%agent commands` — list everything the agent advertised, with
  description and input hint.
- `%agent commands NAME [args...]` — explicit invocation form for users
  who don't want the leading-`/` interception (or for arguments with
  awkward characters).

Tab-completion is updated so `%agent commands ` and a bare leading `/`
both complete from the cached list.

### 3. Pre-flight visibility & failure surface

- **Session-start MCP probe.** Immediately after `new_session` succeeds,
  emit one line per configured MCP server: `mcp: jupyter ✓ (12 tools)`
  or `mcp: jupyter ✗ failed to start: <error>`. Implementation: call the
  server's `tools/list` (the kernel already proxies MCP for its own
  purposes; if it doesn't, codex-acp can be asked via its standard MCP
  introspection). If we can't probe directly, at minimum print the names
  and sources of the servers we passed in.
- **Tool-failure surface.** Track per-tool-name failure counts in the
  prompt loop. If the same tool returns errors ≥ N times in a row,
  surface a `stderr` line to the cell — exactly the signal that would
  have made the `insert_cell` 404 storm obvious in the golden-notebook
  incident.

## Implementation order

The three areas are largely independent:

1. **MCP source merge.** Largest immediate win for honesty; unblocks
   per-session disable. Land first.
2. **Slash commands.** Same scaffolding the user has been asking for in
   `%agent`. Lands after (1) because some slash commands themselves
   operate on MCP state and we want our list to be canonical.
3. **Health probe & failure surface.** Smallest delta, big debug payoff;
   land alongside or just after (1).

## Open questions

- **Slash-command invocation wire format.** Confirm against
  `codex-acp` 0.43.x by experimentation. Update this doc with the chosen
  form and a link to the codex-acp source if/when it becomes available.
- **Project-local config search path.** Match Codex's own search rules
  (cwd-relative? walk up?) — TBD by inspecting `codex-acp` behaviour.
- **MCP introspection.** Is there a standard ACP-level call to ask the
  agent "which MCP servers do you currently have wired in?" If so, we
  should prefer that over re-parsing `~/.codex/config.toml`. As of ACP
  0.9, there isn't — hence the merge approach above.

## Related documents

- [docs/golden-notebook-newlines-postmortem.md](golden-notebook-newlines-postmortem.md)
  — the incident that motivated this work.
- [docs/jupyter-mcp-server-breaks-cell-execution.md](jupyter-mcp-server-breaks-cell-execution.md)
  — why `jupyter-collaboration` is required for cell-write tools.
- [docs/codex-lmstudio-shim.md](codex-lmstudio-shim.md) — the LM Studio
  shim that ran underneath the test.
