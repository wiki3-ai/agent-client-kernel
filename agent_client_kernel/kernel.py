"""
ACP Jupyter Kernel implementation.

This kernel implements a Jupyter kernel that acts as an ACP client,
connecting to external ACP agents like Codex.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from ipykernel.kernelbase import Kernel

from acp import (
    Client,
    PROTOCOL_VERSION,
    RequestError,
    connect_to_agent,
    text_block,
)
from acp.core import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AudioContentBlock,
    AvailableCommandsUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    KillTerminalResponse,
    McpServerStdio,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from . import __version__, KERNEL_NAME, DISPLAY_NAME, LANGUAGE

# Configure logging
log_level = os.environ.get("ACP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Debug logging filter - comma-separated list of update types to log
# e.g., ACP_DEBUG_UPDATES="AgentMessageChunk,ToolCallStart" or "all" for everything
DEBUG_UPDATES = os.environ.get("ACP_DEBUG_UPDATES", "").split(",") if os.environ.get("ACP_DEBUG_UPDATES") else []


@dataclass
class MCPServer:
    """Configuration for an MCP server.

    `source` indicates where the entry came from so we can keep
    `state.mcp_servers` (user-added) separate from servers picked up from
    Codex's own config file, while still presenting a single merged view
    to the user and to the agent.
    """
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # Where the entry originated. "user" means added in-kernel via
    # %agent mcp add; "codex-global" / "codex-project" come from
    # ~/.codex/config.toml or ./.codex/config.toml respectively.
    source: str = "user"
    # Disabling lets a user suppress a preconfigured server for one
    # session without editing the config file.
    enabled: bool = True


@dataclass
class TerminalState:
    """State for a terminal process."""
    terminal_id: str
    process: asyncio.subprocess.Process
    output: str = ""
    output_byte_limit: int | None = None
    exit_code: int | None = None
    cwd: str | None = None


@dataclass
class SessionState:
    """State for an ACP session."""
    session_id: str | None = None
    cwd: str = field(default_factory=os.getcwd)
    mcp_servers: list[MCPServer] = field(default_factory=list)
    # When True, do not merge ~/.codex/config.toml or ./.codex/config.toml
    # MCP servers into the session.  Toggled via
    # %agent mcp ignore-codex-config on|off.
    ignore_codex_config: bool = False
    # Names of preconfigured (codex-global / codex-project) servers that
    # the user has explicitly disabled for this session.  Persists across
    # config reloads so disable-by-name keeps working even if the user
    # restarts the session.
    disabled_preconfigured: set[str] = field(default_factory=set)
    permission_mode: str = "auto"  # auto, manual, deny
    permission_history: list[dict[str, Any]] = field(default_factory=list)
    response_text: str = ""
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    available_commands: list[dict[str, Any]] = field(default_factory=list)
    terminals: dict[str, TerminalState] = field(default_factory=dict)


class ACPClientImpl(Client):
    """ACP Client implementation that handles callbacks from the agent."""

    def __init__(self, kernel: "ACPKernel") -> None:
        self._kernel = kernel
        self._log = logging.getLogger(f"{__name__}.ACPClient")

    def _send_stream(self, name: str, text: str) -> None:
        """Send stream output to the currently active cell."""
        # Use the kernel's current parent header for proper cell association
        parent = getattr(self._kernel, '_current_parent', None)
        
        if parent is not None and hasattr(self._kernel, 'session'):
            # Use session.send with explicit parent for proper cell routing
            self._kernel.session.send(
                self._kernel.iopub_socket,
                "stream",
                {"name": name, "text": text},
                parent=parent,
            )
        else:
            # Fallback to send_response (may not route to correct cell)
            self._kernel.send_response(
                self._kernel.iopub_socket,
                "stream",
                {"name": name, "text": text},
            )

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Handle permission requests from the agent."""
        self._log.info("Permission requested: tool_call=%s", tool_call)

        mode = self._kernel.state.permission_mode

        # Record permission request
        self._kernel.state.permission_history.append({
            "options": [{"kind": o.kind, "name": o.name, "id": o.option_id} for o in options],
            "tool_call": str(tool_call),
            "mode": mode,
        })

        if mode == "deny":
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        elif mode == "manual":
            # TODO: Implement interactive prompting
            # For now, fall back to auto-approve
            pass

        # Auto mode: select first allow option
        for option in options:
            if option.kind in ("allow_once", "allow_always"):
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=option.option_id)
                )

        # Fallback to first option
        if options:
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id=options[0].option_id)
            )

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    def _should_log_update(self, update_type: str) -> bool:
        """Check if this update type should be logged based on ACP_DEBUG_UPDATES."""
        if not DEBUG_UPDATES:
            return False
        return "all" in DEBUG_UPDATES or update_type in DEBUG_UPDATES

    def _get_update_summary(self, update: Any) -> str:
        """Get a concise summary of an update for logging."""
        if isinstance(update, AgentMessageChunk):
            content = update.content
            if isinstance(content, TextContentBlock):
                text = content.text[:100] + "..." if len(content.text) > 100 else content.text
                return f"text={text!r}"
            return f"content_type={type(content).__name__}"
        elif isinstance(update, AgentThoughtChunk):
            content = update.content
            if isinstance(content, TextContentBlock):
                text = content.text[:100] + "..." if len(content.text) > 100 else content.text
                return f"thought={text!r}"
            return f"content_type={type(content).__name__}"
        elif isinstance(update, ToolCallStart):
            return f"tool={update.title}, kind={update.kind}, status={update.status}"
        elif isinstance(update, ToolCallProgress):
            return f"tool_id={update.tool_call_id}, status={update.status}"
        elif isinstance(update, AgentPlanUpdate):
            entries = [e.content[:30] for e in (update.entries or [])]
            return f"entries={entries}"
        elif isinstance(update, AvailableCommandsUpdate):
            cmds = [c.name for c in (update.available_commands or [])]
            return f"commands={cmds}"
        return ""

    async def session_update(
        self,
        session_id: str,
        update: (
            AgentMessageChunk
            | AgentThoughtChunk
            | ToolCallStart
            | ToolCallProgress
            | AgentPlanUpdate
            | AvailableCommandsUpdate
            | CurrentModeUpdate
            | Any
        ),
        **kwargs: Any,
    ) -> None:
        """Handle session updates from the agent."""
        update_type = type(update).__name__
        
        # Only log at DEBUG level if filtering is enabled and type matches
        if self._should_log_update(update_type):
            summary = self._get_update_summary(update)
            self._log.debug("session_update: %s - %s", update_type, summary)

        if isinstance(update, AgentMessageChunk):
            content = update.content
            if isinstance(content, TextContentBlock):
                self._kernel.state.response_text += content.text
                # Stream output to notebook
                self._send_stream("stdout", content.text)
            elif isinstance(content, ImageContentBlock):
                self._kernel.state.response_text += "[image]"
            elif isinstance(content, AudioContentBlock):
                self._kernel.state.response_text += "[audio]"

        elif isinstance(update, AgentThoughtChunk):
            content = update.content
            if isinstance(content, TextContentBlock):
                # Show thoughts as stderr (dimmed in notebooks)
                self._send_stream("stderr", f"💭 {content.text}")

        elif isinstance(update, ToolCallStart):
            self._kernel.state.tool_calls[update.tool_call_id] = {
                "title": update.title,
                "kind": update.kind,
                "status": update.status,
                "started": True,
            }
            self._send_stream("stderr", f"\n🔧 {update.title or 'Tool call'} ({update.status or 'pending'})\n")

        elif isinstance(update, ToolCallProgress):
            tool_state = self._kernel.state.tool_calls.get(update.tool_call_id, {})
            tool_state["status"] = update.status
            if update.status == "completed":
                self._send_stream("stderr", f"✅ Tool {update.tool_call_id} completed\n")

        elif isinstance(update, AgentPlanUpdate):
            plan_text = "\n📋 Plan:\n"
            for entry in update.entries or []:
                status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(entry.status, "•")
                plan_text += f"  {status_emoji} {entry.content}\n"
            self._send_stream("stderr", plan_text)

        elif isinstance(update, AvailableCommandsUpdate):
            # Each notification replaces the full list.  We capture name +
            # description + a flat `input_hint` so the kernel can show
            # users what to type and so `do_complete` can offer them.
            new_cmds: list[dict[str, Any]] = []
            for cmd in (update.available_commands or []):
                hint: str | None = None
                inp = getattr(cmd, "input", None)
                if inp is not None:
                    # ACP UnstructuredCommandInput has a `hint` field;
                    # other input shapes may not.  Be permissive.
                    hint = getattr(inp, "hint", None)
                new_cmds.append(
                    {
                        "name": cmd.name,
                        "description": cmd.description,
                        "input_hint": hint,
                    }
                )
            self._kernel.state.available_commands = new_cmds

    # Maximum size for file content in JSON-RPC responses (must fit in 64KB with JSON overhead)
    MAX_FILE_CONTENT_SIZE = 48 * 1024  # 48KB to leave room for JSON framing

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: Any,
    ) -> WriteTextFileResponse | None:
        """Handle file write requests from the agent."""
        self._log.info("Writing file: %s (%d bytes)", path, len(content))
        try:
            pathlib_path = Path(path)
            pathlib_path.parent.mkdir(parents=True, exist_ok=True)
            pathlib_path.write_text(content)
            self._send_stream("stderr", f"📝 Wrote {path}\n")
            return WriteTextFileResponse()
        except Exception as e:
            self._log.error("Failed to write file %s: %s", path, e)
            raise RequestError.internal_error(str(e))

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        """Handle file read requests from the agent."""
        self._log.info("Reading file: %s", path)
        try:
            pathlib_path = Path(path)
            content = pathlib_path.read_text()
            
            # Apply line and limit if specified
            if line is not None or limit is not None:
                lines = content.splitlines(keepends=True)
                start = (line - 1) if line else 0
                end = (start + limit) if limit else len(lines)
                content = "".join(lines[start:end])
            
            # Enforce size limit to prevent JSON-RPC buffer overflow
            # The asyncio StreamReader has a 64KB line limit
            if len(content.encode('utf-8')) > self.MAX_FILE_CONTENT_SIZE:
                truncated_content = content.encode('utf-8')[:self.MAX_FILE_CONTENT_SIZE].decode('utf-8', errors='ignore')
                # Find last complete line to avoid cutting mid-line
                last_newline = truncated_content.rfind('\n')
                if last_newline > 0:
                    truncated_content = truncated_content[:last_newline + 1]
                truncated_content += f"\n... [truncated: file exceeds {self.MAX_FILE_CONTENT_SIZE // 1024}KB limit, use 'line' and 'limit' params to read sections]"
                self._send_stream("stderr", f"⚠️ File {path} truncated (>{self.MAX_FILE_CONTENT_SIZE // 1024}KB)\n")
                return ReadTextFileResponse(content=truncated_content)
            
            return ReadTextFileResponse(content=content)
        except FileNotFoundError:
            raise RequestError.invalid_params({"path": path, "reason": "file not found"})
        except Exception as e:
            self._log.error("Failed to read file %s: %s", path, e)
            raise RequestError.internal_error(str(e))

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """Create a terminal and run a command."""
        terminal_id = str(uuid.uuid4())
        
        # Build environment
        proc_env = os.environ.copy()
        if env:
            for var in env:
                proc_env[var.name] = var.value
        
        # Use session cwd if not specified
        work_dir = cwd or self._kernel.state.cwd
        
        self._log.info("Creating terminal %s: %s %s (cwd=%s)", terminal_id, command, args or [], work_dir)
        self._send_stream("stderr", f"🖥️ Running: {command} {' '.join(args or [])}\n")
        
        try:
            # Start the process
            process = await asyncio.create_subprocess_exec(
                command,
                *(args or []),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=work_dir,
                env=proc_env,
            )
            
            # Store terminal state
            self._kernel.state.terminals[terminal_id] = TerminalState(
                terminal_id=terminal_id,
                process=process,
                output="",
                output_byte_limit=output_byte_limit,
                cwd=work_dir,
            )
            
            # Start background task to collect output
            asyncio.create_task(self._collect_terminal_output(terminal_id))
            
            return CreateTerminalResponse(terminal_id=terminal_id)
            
        except Exception as e:
            self._log.error("Failed to create terminal: %s", e)
            raise RequestError.internal_error(str(e))

    async def _collect_terminal_output(self, terminal_id: str) -> None:
        """Background task to collect terminal output."""
        terminal = self._kernel.state.terminals.get(terminal_id)
        if not terminal or not terminal.process.stdout:
            return
        
        try:
            while True:
                chunk = await terminal.process.stdout.read(4096)
                if not chunk:
                    break
                
                text = chunk.decode("utf-8", errors="replace")
                terminal.output += text
                
                # Apply byte limit by truncating from beginning
                if terminal.output_byte_limit and len(terminal.output) > terminal.output_byte_limit:
                    terminal.output = terminal.output[-terminal.output_byte_limit:]
                
                # Stream to notebook
                self._send_stream("stdout", text)
            
            # Wait for process to complete and store exit code
            await terminal.process.wait()
            terminal.exit_code = terminal.process.returncode
            
        except Exception as e:
            self._log.error("Error collecting terminal output: %s", e)

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> TerminalOutputResponse:
        """Get terminal output."""
        terminal = self._kernel.state.terminals.get(terminal_id)
        if not terminal:
            raise RequestError.invalid_params({"terminal_id": terminal_id, "reason": "terminal not found"})
        
        from acp.schema import TerminalExitStatus
        exit_status = None
        if terminal.exit_code is not None:
            exit_status = TerminalExitStatus(exit_code=terminal.exit_code)
        
        output = terminal.output
        truncated = bool(terminal.output_byte_limit and len(terminal.output) >= terminal.output_byte_limit)
        
        # Enforce size limit to prevent JSON-RPC buffer overflow
        if len(output.encode('utf-8')) > self.MAX_FILE_CONTENT_SIZE:
            output = output.encode('utf-8')[-self.MAX_FILE_CONTENT_SIZE:].decode('utf-8', errors='ignore')
            truncated = True
        
        return TerminalOutputResponse(
            output=output,
            truncated=truncated,
            exit_status=exit_status,
        )

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> ReleaseTerminalResponse | None:
        """Release a terminal."""
        terminal = self._kernel.state.terminals.pop(terminal_id, None)
        if terminal and terminal.process.returncode is None:
            terminal.process.terminate()
            try:
                await asyncio.wait_for(terminal.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                terminal.process.kill()
        
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal to exit."""
        terminal = self._kernel.state.terminals.get(terminal_id)
        if not terminal:
            raise RequestError.invalid_params({"terminal_id": terminal_id, "reason": "terminal not found"})
        
        # Wait for process to complete
        if terminal.process.returncode is None:
            await terminal.process.wait()
            terminal.exit_code = terminal.process.returncode
        
        return WaitForTerminalExitResponse(exit_code=terminal.exit_code)

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> KillTerminalResponse | None:
        """Kill a terminal."""
        terminal = self._kernel.state.terminals.get(terminal_id)
        if terminal and terminal.process.returncode is None:
            terminal.process.kill()
            await terminal.process.wait()
            terminal.exit_code = terminal.process.returncode
        
        return KillTerminalResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods."""
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications."""
        pass

    def on_connect(self, conn) -> None:
        """Called when connected to the agent."""
        self._log.debug("Connected to agent")


class ACPKernel(Kernel):
    """Jupyter kernel that acts as an ACP client."""

    implementation = KERNEL_NAME
    implementation_version = __version__
    language = LANGUAGE
    language_version = "1.0"
    language_info = {
        "name": LANGUAGE,
        "mimetype": "text/plain",
        "file_extension": ".agent",
    }
    banner = f"Agent Client Protocol Kernel v{__version__}"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log = logging.getLogger(__name__)

        # ACP connection state
        self._proc: aio_subprocess.Process | None = None
        self._conn: ClientSideConnection | None = None
        self._client: ACPClientImpl | None = None

        # Session state (per-kernel instance, so per-notebook)
        self.state = SessionState()

        # Current parent header for proper cell association of async output
        self._current_parent: dict | None = None

        # Agent configuration from environment
        self._agent_command = os.environ.get("ACP_AGENT_COMMAND", "codex-acp")
        self._agent_args = os.environ.get("ACP_AGENT_ARGS", "").split() if os.environ.get("ACP_AGENT_ARGS") else []

        # Magic command patterns
        self._magic_pattern = re.compile(r"^%(\w+)\s*(.*)?$", re.MULTILINE)

    @property
    def is_connected(self) -> bool:
        """Check if connected to an agent."""
        return self._conn is not None and self._proc is not None and self._proc.returncode is None

    # ------------------------------------------------------------------
    # MCP-server discovery & merging.
    #
    # The kernel has two sources of MCP servers:
    #   * `state.mcp_servers` — added in-kernel via %agent mcp add.
    #   * `~/.codex/config.toml` / `./.codex/config.toml` — read by
    #     `codex-acp` itself at startup.  We parse the same files so
    #     `%agent mcp list` can honestly report what the session will
    #     have, and so the user can disable a preconfigured server for
    #     one notebook without editing global config.
    #
    # `_merged_mcp_servers()` is the single source of truth fed to
    # `new_session` / `resume_session`.
    # ------------------------------------------------------------------

    def _codex_config_paths(self) -> list[tuple[Path, str]]:
        """Return (path, source) tuples for known Codex config files."""
        codex_home = Path(
            os.environ.get("CODEX_HOME") or (Path.home() / ".codex")
        ).expanduser()
        return [
            (codex_home / "config.toml", "codex-global"),
            (Path(self.state.cwd) / ".codex" / "config.toml", "codex-project"),
        ]

    def _parse_codex_config(self, path: Path, source: str) -> list[MCPServer]:
        """Parse a Codex config file and return its MCP-server entries."""
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as e:
            self._log.warning("Could not read %s: %s", path, e)
            return []

        servers: list[MCPServer] = []
        for name, entry in (data.get("mcp_servers") or {}).items():
            if not isinstance(entry, dict):
                continue
            command = entry.get("command")
            if not command:
                continue
            args = list(entry.get("args") or [])
            env_block = entry.get("env") or {}
            env = {str(k): str(v) for k, v in env_block.items()} if isinstance(env_block, dict) else {}
            servers.append(
                MCPServer(
                    name=name,
                    command=str(command),
                    args=[str(a) for a in args],
                    env=env,
                    source=source,
                    # `enabled` for preconfigured servers is derived at
                    # merge time from `state.disabled_preconfigured`,
                    # so leave the default here.
                )
            )
        return servers

    def _load_codex_mcp_servers(self) -> list[MCPServer]:
        """Load MCP servers from Codex config files.

        Tests can stub this by setting ``self._codex_mcp_cache = []``.
        """
        cached = getattr(self, "_codex_mcp_cache", None)
        if cached is not None:
            return cached
        if self.state.ignore_codex_config:
            return []
        servers: list[MCPServer] = []
        for path, source in self._codex_config_paths():
            if path.exists():
                servers.extend(self._parse_codex_config(path, source))
        return servers

    def _merged_mcp_servers(self) -> list[MCPServer]:
        """Return the full MCP-server list for this session.

        User entries take precedence over codex-project, which takes
        precedence over codex-global, when names collide.  Disabled
        entries are kept in the result so `%agent mcp list` can show
        them — callers building an ACP `new_session` request must
        filter on ``enabled``.
        """
        codex_servers = self._load_codex_mcp_servers()
        by_name: dict[str, MCPServer] = {}
        # Precedence: codex-global < codex-project < user
        order = {"codex-global": 0, "codex-project": 1, "user": 2}
        for s in codex_servers + list(self.state.mcp_servers):
            existing = by_name.get(s.name)
            if existing is None or order.get(s.source, 0) >= order.get(existing.source, 0):
                by_name[s.name] = s
        # Apply per-session disable overlay for preconfigured entries.
        # User entries carry their own `enabled` flag.  We don't mutate
        # the cached server objects — return a fresh copy so toggling
        # `disabled_preconfigured` between calls is reflected cleanly.
        result = []
        for s in by_name.values():
            if s.source != "user":
                result.append(
                    MCPServer(
                        name=s.name,
                        command=s.command,
                        args=s.args,
                        env=s.env,
                        source=s.source,
                        enabled=s.name not in self.state.disabled_preconfigured,
                    )
                )
            else:
                result.append(s)
        return result

    def _active_mcp_servers(self) -> list[MCPServer]:
        """Merged list filtered to enabled entries."""
        return [s for s in self._merged_mcp_servers() if s.enabled]

    def _announce_mcp_servers(self, servers: list[MCPServer]) -> None:
        """Print a one-line summary of the MCP servers wired into the session.

        This is intentionally cheap — we don't probe each server's
        ``tools/list`` here (that would slow every session start) but
        we *do* surface their names and sources so the user knows what
        the agent has access to.  This is the cheap version of the
        "MCP health probe" in
        ``docs/mcp-config-and-slash-commands.md``.
        """
        try:
            if not servers:
                msg = "mcp: no servers configured\n"
            else:
                parts = [f"{s.name} [{s.source}]" for s in servers]
                msg = f"mcp: {len(servers)} server(s): {', '.join(parts)}\n"
            self.send_response(
                self.iopub_socket,
                "stream",
                {"name": "stderr", "text": msg},
            )
        except Exception:
            # Don't let an iopub hiccup fail session start.
            pass

    async def _start_agent(self) -> None:
        """Start the ACP agent process."""
        if self._proc is not None:
            return

        # Bring up the LM Studio request shim if it isn't already. Safe
        # no-op when the shim is disabled or unused by the configured
        # model provider; see agent_client_kernel.lmstudio_shim for the
        # selection rules. Done here (not at module import) so test
        # imports don't bind ports.
        try:
            from . import lmstudio_shim

            target = lmstudio_shim.ensure_running()
            if target:
                self._log.info("LM Studio shim active -> %s", target)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("LM Studio shim failed to start: %s", exc)

        self._log.info("Starting agent: %s %s", self._agent_command, " ".join(self._agent_args))

        try:
            # Find the agent executable
            program_path = Path(self._agent_command)
            spawn_program = self._agent_command
            spawn_args = self._agent_args

            if program_path.exists() and not os.access(program_path, os.X_OK):
                spawn_program = sys.executable
                spawn_args = [str(program_path), *self._agent_args]

            # Start the agent process with a larger buffer limit to handle large responses.
            # The default asyncio StreamReader limit is 64KB per line, which is too small
            # for large tool outputs (e.g., jupyter/execute_cell results).
            # 
            # NOTE: The proper fix is for codex-acp to truncate large MCP tool outputs
            # before sending them to the client. See end_mcp_tool_call() in conversation.rs
            # which sends CallToolResult content without any size checking.
            # Bug: https://github.com/zed-industries/codex-acp - should truncate large outputs
            buffer_limit = 1 * 1024 * 1024  # 1MB
            self._proc = await asyncio.create_subprocess_exec(
                spawn_program,
                *spawn_args,
                stdin=aio_subprocess.PIPE,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.PIPE,
                limit=buffer_limit,
            )

            if self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("Agent process does not expose stdio pipes")

            # Create client and connection
            self._client = ACPClientImpl(self)
            self._conn = connect_to_agent(self._client, self._proc.stdin, self._proc.stdout)

            # Initialize the agent
            await self._conn.initialize(protocol_version=PROTOCOL_VERSION)

            # Create a new session with MCP servers.  The active list
            # merges user-added entries with whatever's in
            # ~/.codex/config.toml (and the project-local equivalent),
            # minus anything the user has explicitly disabled.
            active_servers = self._active_mcp_servers()
            mcp_servers = [
                McpServerStdio(
                    name=server.name,
                    command=server.command,
                    args=server.args,
                    env=[EnvVariable(name=k, value=v) for k, v in server.env.items()],
                )
                for server in active_servers
            ]

            self._log.info("Creating session with %d MCP servers", len(mcp_servers))
            for server in active_servers:
                self._log.info(
                    "  MCP server: %s (%s) -> %s %s",
                    server.name, server.source, server.command, server.args,
                )

            session = await self._conn.new_session(mcp_servers=mcp_servers, cwd=self.state.cwd)
            self.state.session_id = session.session_id

            self._log.info("Agent started with session ID: %s", self.state.session_id)
            self._announce_mcp_servers(active_servers)

        except Exception as e:
            self._log.error("Failed to start agent: %s", e)
            await self._stop_agent()
            raise

    async def _stop_agent(self) -> None:
        """Stop the ACP agent process."""
        if self._proc is None:
            return

        self._log.info("Stopping agent")

        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()

        self._proc = None
        self.state.session_id = None

    async def _restart_session(self) -> None:
        """Restart the agent session (e.g., after config change)."""
        await self._stop_agent()
        await self._start_agent()

    async def _resume_session(self) -> bool:
        """
        Resume the session with updated MCP servers.
        
        Uses the ACP resume_session method to update the session without
        restarting. Returns True if successful, False if the agent doesn't
        support resume_session (in which case a restart is needed).
        """
        if not self.is_connected or not self._conn or not self.state.session_id:
            return False

        # Build MCP servers list using the merged & enabled view.
        active_servers = self._active_mcp_servers()
        mcp_servers = [
            McpServerStdio(
                name=server.name,
                command=server.command,
                args=server.args,
                env=[EnvVariable(name=k, value=v) for k, v in server.env.items()],
            )
            for server in active_servers
        ]

        try:
            self._log.info("Resuming session with %d MCP servers", len(mcp_servers))
            for server in active_servers:
                self._log.info(
                    "  MCP server: %s (%s) -> %s %s",
                    server.name, server.source, server.command, server.args,
                )

            await self._conn.resume_session(
                session_id=self.state.session_id,
                cwd=self.state.cwd,
                mcp_servers=mcp_servers,
            )
            self._log.info("Session resumed successfully")
            return True
        except RequestError as e:
            if "Method not found" in str(e):
                self._log.info("Agent does not support resume_session")
                return False
            raise

    async def _send_prompt(self, code: str) -> str:
        """Send a prompt to the agent and return the response."""
        if not self.is_connected:
            await self._start_agent()

        if not self._conn or not self.state.session_id:
            raise RuntimeError("Not connected to agent")

        # Reset response text for this prompt
        self.state.response_text = ""
        self.state.tool_calls = {}

        try:
            response = await self._conn.prompt(
                session_id=self.state.session_id,
                prompt=[text_block(code)],
            )

            self._log.debug("Prompt completed with stopReason: %s", response.stop_reason)

            # Add newline after streaming output
            if self.state.response_text:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stdout", "text": "\n"},
                )

            return self.state.response_text

        except ValueError as e:
            # Check for stream buffer overflow error (agent response too large)
            # This happens when the agent sends a JSON-RPC message exceeding our buffer limit.
            # The proper fix is for the agent to truncate large outputs before sending.
            if "chunk is longer than limit" in str(e) or "Separator is found" in str(e):
                self._log.warning("Agent response exceeded stream buffer limit (1MB), restarting connection")
                # Clean up the broken connection
                await self._stop_agent()
                # Notify user about what happened - note that any streamed content before
                # the oversized message was already displayed
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": "\n⚠️ Agent sent a message exceeding 1MB buffer limit (likely large tool output). Connection reset.\n"},
                )
                raise RuntimeError(
                    "Agent sent a message larger than 1MB. This typically happens when "
                    "a tool (e.g., jupyter/execute_cell) returns very large output. "
                    "The agent should truncate large outputs before sending. "
                    "Any content streamed before the error is shown above."
                ) from e
            raise

        except Exception as e:
            self._log.error("Error sending prompt: %s", e, exc_info=True)
            raise

    async def _handle_magic(self, line: str) -> tuple[bool, str | None]:
        """
        Handle magic commands.
        
        Returns (is_magic, result).
        If is_magic is True, result is the magic output (or None if no output).
        If is_magic is False, this is not a magic command.
        """
        line = line.strip()
        if not line.startswith("%"):
            return False, None

        # Parse magic command
        match = self._magic_pattern.match(line)
        if not match:
            return False, None

        magic_name = match.group(1)
        args = match.group(2) or ""

        # Route to magic handlers
        handler = getattr(self, f"_magic_{magic_name}", None)
        if handler:
            result = handler(args.strip())
            # Support async handlers
            if asyncio.iscoroutine(result):
                result = await result
            return True, result

        return True, f"Unknown magic command: %{magic_name}"

    # Magic command handlers

    def _magic_agent(self, args: str) -> str:
        """Handle %agent magic command with subcommands."""
        if not args:
            return self._magic_help_agent()

        parts = args.split(None, 1)
        subcommand = parts[0]
        subargs = parts[1] if len(parts) > 1 else ""

        handler = getattr(self, f"_magic_agent_{subcommand}", None)
        if handler:
            return handler(subargs.strip())

        return f"Unknown subcommand: {subcommand}\n\nUse %agent for help."

    def _magic_help_agent(self) -> str:
        """Show help for %agent command."""
        return """Agent Client Protocol Kernel Commands

MCP Server Configuration:
  %agent mcp add NAME COMMAND [ARGS...]   - Add an MCP server
  %agent mcp list                          - List configured MCP servers
  %agent mcp remove NAME                   - Remove a user-added MCP server
  %agent mcp disable NAME                  - Disable a server for this session
  %agent mcp enable NAME                   - Re-enable a disabled server
  %agent mcp clear                         - Remove all user-added servers
  %agent mcp ignore-codex-config on|off    - Skip merging ~/.codex/config.toml

Slash Commands (advertised by the agent):
  %agent commands                          - List slash commands the agent supports
  /name [args]                             - Invoke a slash command directly

Session Management:
  %agent session new [CWD]                 - Create new session
  %agent session info                      - Show session information
  %agent session restart                   - Restart session

Permission Configuration:
  %agent permissions [auto|manual|deny]    - Set permission mode
  %agent permissions history               - View permission history

Configuration:
  %agent config                            - Show current configuration
  %agent env                               - Show environment variables
"""

    async def _magic_agent_mcp(self, args: str) -> str:
        """Handle %agent mcp subcommand."""
        if not args:
            return self._magic_agent_mcp_list("")

        parts = args.split(None, 1)
        action = parts[0]
        action_args = parts[1] if len(parts) > 1 else ""

        if action == "add":
            return await self._magic_agent_mcp_add(action_args)
        elif action == "list":
            return self._magic_agent_mcp_list(action_args)
        elif action == "remove":
            return await self._magic_agent_mcp_remove(action_args)
        elif action == "clear":
            return await self._magic_agent_mcp_clear(action_args)
        elif action == "disable":
            return await self._magic_agent_mcp_disable(action_args)
        elif action == "enable":
            return await self._magic_agent_mcp_enable(action_args)
        elif action == "ignore-codex-config":
            return await self._magic_agent_mcp_ignore_codex_config(action_args)

        return f"Unknown mcp action: {action}"

    async def _magic_agent_mcp_add(self, args: str) -> str:
        """Add an MCP server."""
        if not args:
            return "Usage: %agent mcp add NAME COMMAND [ARGS...]"

        parts = args.split(None, 2)
        if len(parts) < 2:
            return "Usage: %agent mcp add NAME COMMAND [ARGS...]"

        name = parts[0]
        command = parts[1]
        server_args = parts[2].split() if len(parts) > 2 else []

        # Check if server exists (update in place)
        updated = False
        for i, server in enumerate(self.state.mcp_servers):
            if server.name == name:
                self.state.mcp_servers[i] = MCPServer(name=name, command=command, args=server_args)
                updated = True
                break

        if not updated:
            self.state.mcp_servers.append(MCPServer(name=name, command=command, args=server_args))

        action = "Updated" if updated else "Added"
        
        if self.is_connected:
            # Try to use resume_session to apply changes without restart
            if await self._resume_session():
                return f"{action} MCP server '{name}'. Session updated."
            else:
                return f"{action} MCP server '{name}'. Run %agent session restart to apply."
        else:
            return f"{action} MCP server '{name}'. It will be available when the session starts."

    def _magic_agent_mcp_list(self, args: str) -> str:
        """List MCP servers from all sources, with origin and enabled state."""
        merged = self._merged_mcp_servers()
        if not merged:
            return "No MCP servers configured"

        # Stable display order: by source (user first), then name.
        order = {"user": 0, "codex-project": 1, "codex-global": 2}
        merged.sort(key=lambda s: (order.get(s.source, 99), s.name))

        lines = ["Configured MCP servers:"]
        for server in merged:
            args_str = " ".join(server.args) if server.args else ""
            state = "" if server.enabled else " (disabled)"
            tail = f" {args_str}".rstrip()
            lines.append(
                f"  - {server.name} [{server.source}{state}]: {server.command}{tail}"
            )
        if self.state.ignore_codex_config:
            lines.append("")
            lines.append("(ignore-codex-config is ON; codex-* entries are not loaded)")
        return "\n".join(lines)

    async def _magic_agent_mcp_remove(self, args: str) -> str:
        """Remove a user-added MCP server.

        Preconfigured servers (codex-global / codex-project) cannot be
        removed via this command — use ``%agent mcp disable NAME``.
        """
        if not args:
            return "Usage: %agent mcp remove NAME"

        name = args.strip()
        for i, server in enumerate(self.state.mcp_servers):
            if server.name == name:
                self.state.mcp_servers.pop(i)
                if self.is_connected:
                    # Try to use resume_session to apply changes without restart
                    if await self._resume_session():
                        return f"Removed MCP server '{name}'. Session updated."
                    else:
                        return f"Removed MCP server '{name}'. Run %agent session restart to apply."
                return f"Removed MCP server '{name}'"

        # Not in user state — see if it's preconfigured.
        for server in self._merged_mcp_servers():
            if server.name == name:
                return (
                    f"'{name}' is preconfigured ({server.source}); "
                    f"use '%agent mcp disable {name}' to skip it for this session."
                )
        return f"No MCP server named '{name}' found"

    async def _magic_agent_mcp_disable(self, args: str) -> str:
        """Disable an MCP server for this session (any source)."""
        name = args.strip()
        if not name:
            return "Usage: %agent mcp disable NAME"
        found = None
        for s in self._merged_mcp_servers():
            if s.name == name:
                found = s
                break
        if found is None:
            return f"No MCP server named '{name}' found"
        if found.source == "user":
            for s in self.state.mcp_servers:
                if s.name == name:
                    s.enabled = False
                    break
        else:
            self.state.disabled_preconfigured.add(name)
        if self.is_connected and await self._resume_session():
            return f"Disabled MCP server '{name}'. Session updated."
        return f"Disabled MCP server '{name}'. Run %agent session restart to apply."

    async def _magic_agent_mcp_enable(self, args: str) -> str:
        """Re-enable a previously disabled MCP server."""
        name = args.strip()
        if not name:
            return "Usage: %agent mcp enable NAME"
        changed = False
        for s in self.state.mcp_servers:
            if s.name == name and not s.enabled:
                s.enabled = True
                changed = True
                break
        if name in self.state.disabled_preconfigured:
            self.state.disabled_preconfigured.discard(name)
            changed = True
        if not changed:
            return f"MCP server '{name}' is not disabled (or does not exist)"
        if self.is_connected and await self._resume_session():
            return f"Enabled MCP server '{name}'. Session updated."
        return f"Enabled MCP server '{name}'. Run %agent session restart to apply."

    async def _magic_agent_mcp_ignore_codex_config(self, args: str) -> str:
        """Toggle merging of ~/.codex/config.toml entries."""
        val = args.strip().lower()
        if val not in ("on", "off"):
            current = "on" if self.state.ignore_codex_config else "off"
            return (
                f"ignore-codex-config is {current}\n"
                "Usage: %agent mcp ignore-codex-config on|off"
            )
        self.state.ignore_codex_config = (val == "on")
        if self.is_connected and await self._resume_session():
            return f"ignore-codex-config set to {val}. Session updated."
        return f"ignore-codex-config set to {val}. Run %agent session restart to apply."

    async def _magic_agent_mcp_clear(self, args: str) -> str:
        """Clear all MCP servers."""
        count = len(self.state.mcp_servers)
        self.state.mcp_servers = []
        if count > 0 and self.is_connected:
            # Try to use resume_session to apply changes without restart
            if await self._resume_session():
                return f"Removed {count} MCP server(s). Session updated."
            else:
                return f"Removed {count} MCP server(s). Run %agent session restart to apply."
        return f"Removed {count} MCP server(s)"

    def _magic_agent_commands(self, args: str) -> str:
        """List slash commands the agent has advertised.

        Populated from ACP ``AvailableCommandsUpdate`` notifications.
        Cells beginning with ``/`` are dispatched directly by
        ``do_execute``; this magic is for inspection.
        """
        cmds = self.state.available_commands
        if not cmds:
            return (
                "No slash commands advertised by the agent yet.\n"
                "(Some agents advertise them only after the first prompt.)"
            )
        lines = ["Available slash commands:"]
        for cmd in cmds:
            hint = cmd.get("input_hint")
            hint_str = f" {hint}" if hint else ""
            desc = cmd.get("description") or ""
            lines.append(f"  /{cmd['name']}{hint_str}")
            if desc:
                lines.append(f"      {desc}")
        return "\n".join(lines)

    def _magic_agent_session(self, args: str) -> str:
        """Handle %agent session subcommand."""
        if not args:
            return self._magic_agent_session_info("")

        parts = args.split(None, 1)
        action = parts[0]
        action_args = parts[1] if len(parts) > 1 else ""

        if action == "new":
            return self._magic_agent_session_new(action_args)
        elif action == "info":
            return self._magic_agent_session_info(action_args)
        elif action == "restart":
            return self._magic_agent_session_restart(action_args)

        return f"Unknown session action: {action}"

    async def _magic_agent_session_new(self, args: str) -> str:
        """Create a new session."""
        cwd = args.strip() if args.strip() else os.getcwd()

        if not os.path.isdir(cwd):
            return f"Directory does not exist: {cwd}"

        self.state.cwd = cwd
        await self._restart_session()
        return f"Created new session with working directory: {cwd}"

    def _magic_agent_session_info(self, args: str) -> str:
        """Show session information."""
        merged = self._merged_mcp_servers()
        active = [s for s in merged if s.enabled]
        lines = ["Session Information:"]
        lines.append(f"  Session ID: {self.state.session_id or '(not started)'}")
        lines.append(f"  Working Directory: {self.state.cwd}")
        lines.append(f"  Agent: {self._agent_command}")
        lines.append(f"  Connected: {self.is_connected}")
        lines.append(f"  Permission Mode: {self.state.permission_mode}")
        lines.append(
            f"  MCP Servers: {len(active)} active / {len(merged)} total"
        )
        if self.state.ignore_codex_config:
            lines.append("  (ignore-codex-config: on)")

        if self.state.available_commands:
            lines.append("\n  Slash Commands:")
            for cmd in self.state.available_commands:
                lines.append(f"    /{cmd['name']}: {cmd.get('description', '')}")

        return "\n".join(lines)

    async def _magic_agent_session_restart(self, args: str) -> str:
        """Restart the session."""
        await self._restart_session()
        return f"Session restarted. Session ID: {self.state.session_id}"

    def _magic_agent_permissions(self, args: str) -> str:
        """Handle %agent permissions subcommand."""
        if not args:
            return f"Permission mode: {self.state.permission_mode}\n\nUse: %agent permissions [auto|manual|deny]"

        if args == "history":
            if not self.state.permission_history:
                return "No permission requests recorded"
            lines = ["Permission History:"]
            for i, entry in enumerate(self.state.permission_history[-10:], 1):
                lines.append(f"  {i}. mode={entry['mode']}")
            return "\n".join(lines)

        if args in ("auto", "manual", "deny"):
            self.state.permission_mode = args
            return f"Permission mode set to: {args}"

        return f"Unknown permission mode: {args}\n\nUse: auto, manual, or deny"

    def _magic_agent_config(self, args: str) -> str:
        """Show configuration."""
        return f"""Configuration:
  Agent Command: {self._agent_command}
  Agent Args: {' '.join(self._agent_args) or '(none)'}
  Working Directory: {self.state.cwd}
  Permission Mode: {self.state.permission_mode}
  MCP Servers: {len(self.state.mcp_servers)}
"""

    def _magic_agent_env(self, args: str) -> str:
        """Show environment variables."""
        env_vars = [
            ("ACP_AGENT_COMMAND", os.environ.get("ACP_AGENT_COMMAND", "(not set)")),
            ("ACP_AGENT_ARGS", os.environ.get("ACP_AGENT_ARGS", "(not set)")),
            ("ACP_LOG_LEVEL", os.environ.get("ACP_LOG_LEVEL", "(not set)")),
            ("OPENAI_API_KEY", "***" if os.environ.get("OPENAI_API_KEY") else "(not set)"),
        ]
        lines = ["Environment Variables:"]
        for name, value in env_vars:
            lines.append(f"  {name}: {value}")
        return "\n".join(lines)

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: dict | None = None,
        allow_stdin: bool = False,
    ):
        """Execute code in the kernel."""
        # Store the current parent header for proper cell association of async output
        try:
            self._current_parent = self.get_parent()
        except AttributeError:
            # In test environments, get_parent() may not work
            self._current_parent = None

        code = code.strip()
        if not code:
            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        # Check for magic commands
        is_magic, magic_result = await self._handle_magic(code)
        if is_magic:
            if magic_result and not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stdout", "text": magic_result + "\n"},
                )
            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        # Cells that look like `/command ...` are dispatched as slash
        # commands.  We forward the literal text via session/prompt —
        # the agent already advertised which slashes it understands via
        # AvailableCommandsUpdate, so anything not in that list is
        # likely a typo or simply a chat message; either way the agent
        # is the right place to handle it.  We add a brief stderr hint
        # when the slash isn't in the advertised list so users notice.
        if code.startswith("/") and not silent:
            first_token = code.split(None, 1)[0].lstrip("/")
            known = {c["name"] for c in self.state.available_commands}
            if known and first_token not in known:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {
                        "name": "stderr",
                        "text": (
                            f"warning: '/{first_token}' is not in the agent's "
                            f"advertised slash-command list "
                            f"(see %agent commands).\n"
                        ),
                    },
                )

        # Send to agent
        try:
            result = await self._send_prompt(code)
            
            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        except Exception as e:
            error_msg = f"Error: {e}\n\nMake sure the ACP agent is configured correctly.\nCurrent agent: {self._agent_command}"
            
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": error_msg + "\n"},
                )

            return {
                "status": "error",
                "execution_count": self.execution_count,
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [error_msg],
            }

    async def do_shutdown(self, restart: bool):
        """Shutdown the kernel."""
        if self._proc is not None:
            try:
                await self._stop_agent()
            except Exception as e:
                self._log.error("Error stopping agent: %s", e)
        return {"status": "ok", "restart": restart}

    def do_complete(self, code: str, cursor_pos: int):
        """Handle code completion."""
        text = code[:cursor_pos]

        completions: list[str] = []
        cursor_start = cursor_pos

        if text.startswith("%agent "):
            subcommand_prefix = text[7:]
            top_subs = ["mcp", "session", "permissions", "config", "env", "commands"]
            mcp_subs = ["add", "list", "remove", "clear", "disable", "enable", "ignore-codex-config"]

            parts = subcommand_prefix.split(None, 1)
            if len(parts) <= 1 and not subcommand_prefix.endswith(" "):
                # Completing the top-level subcommand
                completions = [s for s in top_subs if s.startswith(subcommand_prefix)]
                cursor_start = cursor_pos - len(subcommand_prefix)
            elif parts and parts[0] == "mcp":
                rest = parts[1] if len(parts) > 1 else ""
                rest_parts = rest.split(None, 1)
                if len(rest_parts) <= 1 and not rest.endswith(" "):
                    completions = [s for s in mcp_subs if s.startswith(rest)]
                    cursor_start = cursor_pos - len(rest)
        elif text.startswith("/"):
            # Slash-command completion from the agent's advertised list.
            prefix = text[1:]
            names = [c["name"] for c in self.state.available_commands]
            completions = [f"/{n}" for n in names if n.startswith(prefix)]
            cursor_start = cursor_pos - len(text)

        return {
            "status": "ok",
            "matches": completions,
            "cursor_start": cursor_start,
            "cursor_end": cursor_pos,
            "metadata": {},
        }

    def do_inspect(self, code: str, cursor_pos: int, detail_level: int = 0):
        """Handle inspection requests."""
        return {
            "status": "ok",
            "found": False,
            "data": {},
            "metadata": {},
        }
