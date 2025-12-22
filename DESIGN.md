# Agent Client Kernel - Design Document

This document explains the internal architecture and key operational details of the ACP (Agent Client Protocol) Jupyter kernel.

## Overview

The Agent Client Kernel acts as a bridge between Jupyter clients (JupyterLab, VS Code, etc.) and ACP-compatible AI agents (Codex, OpenHands, etc.). It translates Jupyter notebook cells into agent prompts and streams the agent's responses back to the notebook.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Jupyter Client │◄───►│  ACPKernel       │◄───►│  ACP Agent      │
│  (JupyterLab)   │     │  (This Kernel)   │     │  (Codex, etc.)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
      ZMQ                   stdio JSON-RPC           Agent-specific
```

## Key Components

### ACPKernel (`kernel.py`)

The main Jupyter kernel class, extending `ipykernel.kernelbase.Kernel`. Handles:

- Jupyter message protocol (execute, complete, inspect, shutdown)
- Agent lifecycle management (start, stop, restart)
- Magic commands (`%agent`, `%mcp`, `%session`, etc.)
- Output streaming to notebook cells

### ACPClientImpl (`kernel.py`)

Implements the ACP `Client` interface to handle callbacks from the agent:

- `session_update()` - Receives streaming updates (text, thoughts, tool calls, plans)
- `request_permission()` - Handles permission requests for agent actions
- File operations (`read_text_file`, `write_text_file`)
- Terminal operations (`create_terminal`, `terminal_output`, etc.)

### SessionState

Maintains per-notebook state:

- Current working directory
- MCP server configurations
- Permission mode and history
- Active terminals
- Current response text and tool calls

## Message Flow

### Cell Execution

1. User runs a cell in Jupyter
2. `do_execute()` receives the code
3. Magic commands (`%agent`, etc.) are handled locally
4. Regular text is sent to the agent via `_send_prompt()`
5. Agent streams responses via `session_update()` callback
6. Responses are forwarded to Jupyter via `send_response()` / `session.send()`

### Output Routing

The kernel must route output to the correct cell, especially for async updates:

```python
# Store parent header when execution starts
self._current_parent = self.get_parent()

# Use parent header when sending async output
self._kernel.session.send(
    self._kernel.iopub_socket,
    "stream",
    {"name": "stdout", "text": text},
    parent=parent,  # Routes to correct cell
)
```

## Jupyter Protocol Notes

### Message Types

The kernel handles these Jupyter message types:

| Message Type | Handler | Description |
|--------------|---------|-------------|
| `execute_request` | `do_execute()` | Run cell code |
| `complete_request` | `do_complete()` | Tab completion |
| `inspect_request` | `do_inspect()` | Object inspection |
| `shutdown_request` | `do_shutdown()` | Kernel shutdown |
| `interrupt_request` | `interrupt_request()` | Cancel execution |
| `comm_open` | `comm_open()` | Widget comm open (stub) |
| `comm_msg` | `comm_msg()` | Widget comm message (stub) |
| `comm_close` | `comm_close()` | Widget comm close (stub) |

### Comm Messages (Widgets)

Jupyter comms are used for interactive widgets (ipywidgets). Some clients send `comm_open` and `comm_msg` messages even when no widgets are used. We provide stub handlers to prevent "Unknown message type" warnings:

```python
async def comm_open(self, stream, ident, msg):
    """Handle comm_open - stub to prevent warning."""
    self._log.debug("comm_open received and ignored (no comm support)")
```

We don't implement full comm support because:
- ACP agents don't use Jupyter widgets
- Adding widget support would require ipywidgets dependency
- The stub handlers are sufficient to suppress warnings

### Output Types

| Output Type | When Used | Notes |
|-------------|-----------|-------|
| `stream` (stdout) | Regular agent responses | Plain text, supports ANSI colors |
| `stream` (stderr) | Thoughts, tool calls, status | Displayed dimmed in most clients |
| `display_data` | Rich HTML (collapsible) | **Does not stream** - see below |

### HTML Display Limitations

`display_data` messages render HTML but have important limitations:

1. **Not streamable** - Each message is a complete discrete block
2. **Message ordering issues** - Can cause subsequent `stream` messages to be delayed or lost in some clients
3. **Workaround** - We send an empty `stream` message after `display_data` to flush the output queue:

```python
# Send HTML
self._kernel.session.send(stream, "display_data", content, parent=parent)

# Flush output queue
self._kernel.session.send(stream, "stream", {"name": "stdout", "text": ""}, parent=parent)
```

## Interrupt Handling

Interrupting execution is critical for long-running agent tasks.

### Signal Handler

```python
def _handle_sigint(self, signum, frame):
    """Handle SIGINT signal (interrupt request)."""
    self._interrupted = True
    if self._current_task and not self._current_task.done():
        self._current_task.cancel()
```

### Task Cancellation

The current prompt task is tracked and can be cancelled:

```python
async def _send_prompt(self, code: str) -> str:
    self._current_task = asyncio.current_task()
    response = await self._conn.prompt(...)
```

When interrupted:
1. SIGINT is received
2. `_handle_sigint()` cancels `_current_task`
3. `do_execute()` catches `CancelledError` and returns `status: "abort"`

### Cleanup on Interrupt

After interrupt, the agent connection may be in an undefined state. The agent process continues running but any in-flight request is abandoned. A subsequent cell execution will work normally.

## Shutdown and Restart

### Shutdown Flow

1. `do_shutdown(restart=False/True)` is called
2. Cancel any running task
3. Stop the agent (`_stop_agent()`):
   - Terminate all managed terminals
   - Close ACP connection
   - Terminate agent process (with timeout, then kill)
   - Clear session state
4. Restore original SIGINT handler

### Agent Restart

When the session needs to be restarted (e.g., after MCP server config change):

```python
async def _restart_session(self) -> None:
    await self._stop_agent()
    await self._start_agent()
```

MCP server changes require a full session restart via `%agent session restart`.

## MCP Server Management

MCP (Model Context Protocol) servers extend agent capabilities with additional tools.

### Configuration

```python
@dataclass
class MCPServer:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
```

### Adding Servers

Via magic command:
```
%mcp add jupyter-server -- uvx mcp-server-jupyter
```

Servers are passed to the agent on session creation/resume.

## Error Handling

### Large Response Buffer Overflow

The asyncio StreamReader has a 64KB line limit by default. We increase it to 1MB, but very large agent responses can still overflow:

```python
buffer_limit = 1 * 1024 * 1024  # 1MB
self._proc = await asyncio.create_subprocess_exec(..., limit=buffer_limit)
```

If overflow occurs, we:
1. Log a warning
2. Stop the agent
3. Notify the user
4. Raise a descriptive error

### Agent Crash

If the agent process exits unexpectedly, `is_connected` will return False and the next `do_execute()` will attempt to restart it.

## Logging

Configurable via environment variables:

- `ACP_LOG_LEVEL` - Set log level (DEBUG, INFO, WARNING, ERROR)
- `ACP_DEBUG_UPDATES` - Comma-separated list of update types to log, or "all"

Example:
```bash
ACP_LOG_LEVEL=DEBUG
ACP_DEBUG_UPDATES=ToolCallStart,ToolCallProgress
```

## Session Update Types

The agent sends various update types during execution:

| Update Type | Displayed As | Description |
|-------------|--------------|-------------|
| `AgentMessageChunk` | stdout | Main response text |
| `AgentThoughtChunk` | stderr (💭) | Agent's thinking process |
| `ToolCallStart` | stderr (🔧) | Tool execution starting |
| `ToolCallProgress` | stderr (✅/❌) | Tool progress/completion |
| `AgentPlanUpdate` | stderr (📋) | Execution plan |
| `AvailableCommandsUpdate` | stderr (📜) | Available commands |
| `CurrentModeUpdate` | stderr (🔄) | Mode change |
| Unknown types | stderr (📨) | Catch-all for debugging |

## File Size Limits

To prevent JSON-RPC buffer overflow, file content is limited:

```python
MAX_FILE_CONTENT_SIZE = 48 * 1024  # 48KB
```

Larger files are truncated with a message suggesting use of `line` and `limit` parameters.

## Future Improvements

1. **Interactive Permission Prompts** - Currently auto-approves; could use `input_request` for manual mode
2. **Widget Support** - Full ipywidgets integration for rich interactive output
3. **Multiple Sessions** - Support multiple agent sessions per kernel
4. **Streaming HTML** - Use `update_display_data` for progressive HTML updates
