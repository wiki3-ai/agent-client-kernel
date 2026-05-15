"""
Tests for magic command handling.
"""

import os
import re
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from agent_client_kernel.kernel import ACPKernel, MCPServer, SessionState


@pytest.fixture
def kernel():
    """Create a kernel instance for testing magic commands."""
    # Patch ipykernel's Kernel initialization
    with patch.object(ACPKernel, '__init__', lambda x, **kw: None):
        k = ACPKernel.__new__(ACPKernel)
        k._log = MagicMock()
        k._proc = None
        k._conn = None
        k._client = None
        k.state = SessionState()
        k._agent_command = "codex-acp"
        k._agent_args = []
        k.iopub_socket = MagicMock()
        k.send_response = MagicMock()
        k.execution_count = 0
        # Need to initialize the magic pattern
        k._magic_pattern = re.compile(r"^%(\w+)\s*(.*)?$", re.MULTILINE)
        # Hermetic: don't pick up host's ~/.codex/config.toml.
        k._codex_mcp_cache = []
        # Mock async methods that would require connection
        k._resume_session = AsyncMock(return_value=False)  # Simulate agent not supporting resume
        k._restart_session = AsyncMock()
        return k


class TestMagicParsing:
    """Test magic command parsing."""

    @pytest.mark.asyncio
    async def test_non_magic_returns_false(self, kernel):
        """Test that non-magic lines return False."""
        is_magic, result = await kernel._handle_magic("hello world")
        assert is_magic is False
        assert result is None

    @pytest.mark.asyncio
    async def test_simple_magic_parsed(self, kernel):
        """Test that simple magic commands are parsed."""
        is_magic, result = await kernel._handle_magic("%agent")
        assert is_magic is True
        assert result is not None

    @pytest.mark.asyncio
    async def test_unknown_magic(self, kernel):
        """Test unknown magic command handling."""
        is_magic, result = await kernel._handle_magic("%unknownmagic")
        assert is_magic is True
        assert "Unknown magic command" in result


class TestAgentMagic:
    """Test %agent magic command."""

    @pytest.mark.asyncio
    async def test_agent_help(self, kernel):
        """Test %agent shows help."""
        is_magic, result = await kernel._handle_magic("%agent")
        assert is_magic is True
        assert "MCP Server Configuration" in result
        assert "Session Management" in result

    @pytest.mark.asyncio
    async def test_agent_unknown_subcommand(self, kernel):
        """Test unknown subcommand."""
        is_magic, result = await kernel._handle_magic("%agent foobar")
        assert "Unknown subcommand" in result


class TestMCPMagic:
    """Test %agent mcp subcommands."""

    @pytest.mark.asyncio
    async def test_mcp_list_empty(self, kernel):
        """Test listing when no MCP servers configured."""
        is_magic, result = await kernel._handle_magic("%agent mcp list")
        assert "No MCP servers configured" in result

    @pytest.mark.asyncio
    async def test_mcp_add(self, kernel):
        """Test adding an MCP server."""
        is_magic, result = await kernel._handle_magic("%agent mcp add myserver mycmd --arg1")
        assert "Added MCP server 'myserver'" in result
        assert len(kernel.state.mcp_servers) == 1
        assert kernel.state.mcp_servers[0].name == "myserver"
        assert kernel.state.mcp_servers[0].command == "mycmd"
        assert kernel.state.mcp_servers[0].args == ["--arg1"]

    @pytest.mark.asyncio
    async def test_mcp_add_update(self, kernel):
        """Test updating an existing MCP server."""
        await kernel._handle_magic("%agent mcp add myserver cmd1")
        await kernel._handle_magic("%agent mcp add myserver cmd2 --new-arg")
        
        assert len(kernel.state.mcp_servers) == 1
        assert kernel.state.mcp_servers[0].command == "cmd2"

    @pytest.mark.asyncio
    async def test_mcp_list_with_servers(self, kernel):
        """Test listing configured MCP servers."""
        await kernel._handle_magic("%agent mcp add server1 cmd1")
        await kernel._handle_magic("%agent mcp add server2 cmd2 --arg")
        
        is_magic, result = await kernel._handle_magic("%agent mcp list")
        assert "server1" in result
        assert "server2" in result
        assert "cmd1" in result

    @pytest.mark.asyncio
    async def test_mcp_remove(self, kernel):
        """Test removing an MCP server."""
        await kernel._handle_magic("%agent mcp add myserver cmd")
        is_magic, result = await kernel._handle_magic("%agent mcp remove myserver")
        
        assert "Removed MCP server 'myserver'" in result
        assert len(kernel.state.mcp_servers) == 0

    @pytest.mark.asyncio
    async def test_mcp_remove_nonexistent(self, kernel):
        """Test removing a non-existent server."""
        is_magic, result = await kernel._handle_magic("%agent mcp remove nosuchserver")
        assert "No MCP server named" in result

    @pytest.mark.asyncio
    async def test_mcp_clear(self, kernel):
        """Test clearing all MCP servers."""
        await kernel._handle_magic("%agent mcp add server1 cmd1")
        await kernel._handle_magic("%agent mcp add server2 cmd2")
        
        is_magic, result = await kernel._handle_magic("%agent mcp clear")
        assert "Removed 2 MCP server(s)" in result
        assert len(kernel.state.mcp_servers) == 0

    @pytest.mark.asyncio
    async def test_mcp_add_missing_args(self, kernel):
        """Test mcp add with missing arguments."""
        is_magic, result = await kernel._handle_magic("%agent mcp add")
        assert "Usage:" in result

        is_magic, result = await kernel._handle_magic("%agent mcp add nameonly")
        assert "Usage:" in result


class TestMCPSourcesAndDisable:
    """Coverage for MCP source merging, disable/enable, and codex-config integration."""

    @pytest.mark.asyncio
    async def test_list_shows_source_tag(self, kernel):
        await kernel._handle_magic("%agent mcp add myserver mycmd")
        _, result = await kernel._handle_magic("%agent mcp list")
        assert "[user]" in result
        assert "myserver" in result

    @pytest.mark.asyncio
    async def test_codex_global_merge(self, kernel):
        """Servers from ~/.codex/config.toml appear in the merged view."""
        kernel._codex_mcp_cache = [
            MCPServer(
                name="jupyter",
                command="uvx",
                args=["jupyter-mcp-server@latest"],
                source="codex-global",
            )
        ]
        _, result = await kernel._handle_magic("%agent mcp list")
        assert "jupyter" in result
        assert "[codex-global]" in result

    @pytest.mark.asyncio
    async def test_user_overrides_codex_global(self, kernel):
        kernel._codex_mcp_cache = [
            MCPServer(name="dup", command="from-codex", source="codex-global")
        ]
        await kernel._handle_magic("%agent mcp add dup from-user")
        active = kernel._active_mcp_servers()
        assert len(active) == 1
        assert active[0].command == "from-user"
        assert active[0].source == "user"

    @pytest.mark.asyncio
    async def test_remove_refuses_preconfigured(self, kernel):
        kernel._codex_mcp_cache = [
            MCPServer(name="jupyter", command="uvx", source="codex-global")
        ]
        _, result = await kernel._handle_magic("%agent mcp remove jupyter")
        assert "preconfigured" in result
        assert "disable" in result
        # And it should still appear in the merged view.
        _, listing = await kernel._handle_magic("%agent mcp list")
        assert "jupyter" in listing

    @pytest.mark.asyncio
    async def test_disable_then_enable_preconfigured(self, kernel):
        kernel._codex_mcp_cache = [
            MCPServer(name="jupyter", command="uvx", source="codex-global")
        ]
        _, result = await kernel._handle_magic("%agent mcp disable jupyter")
        assert "Disabled MCP server 'jupyter'" in result
        assert "jupyter" in kernel.state.disabled_preconfigured
        # Active list filters it out.
        assert kernel._active_mcp_servers() == []
        # Merged view still shows it, marked disabled.
        _, listing = await kernel._handle_magic("%agent mcp list")
        assert "disabled" in listing

        _, result = await kernel._handle_magic("%agent mcp enable jupyter")
        assert "Enabled MCP server 'jupyter'" in result
        assert "jupyter" not in kernel.state.disabled_preconfigured
        assert len(kernel._active_mcp_servers()) == 1

    @pytest.mark.asyncio
    async def test_disable_user_server(self, kernel):
        await kernel._handle_magic("%agent mcp add me cmd")
        _, result = await kernel._handle_magic("%agent mcp disable me")
        assert "Disabled" in result
        # User entry is still in state but with enabled=False.
        assert kernel.state.mcp_servers[0].enabled is False
        assert kernel._active_mcp_servers() == []

    @pytest.mark.asyncio
    async def test_ignore_codex_config_toggle(self, kernel):
        kernel._codex_mcp_cache = [
            MCPServer(name="jupyter", command="uvx", source="codex-global")
        ]
        _, result = await kernel._handle_magic("%agent mcp ignore-codex-config on")
        assert "on" in result.lower()
        assert kernel.state.ignore_codex_config is True
        # Even though cache has an entry, _load_codex_mcp_servers
        # returns [] when ignore is on (cache is bypassed).  Since we
        # set the cache directly, the implementation honors the
        # ignore flag by re-checking:
        kernel._codex_mcp_cache = None  # force reload
        # And with no real config to load, merged becomes empty:
        _, listing = await kernel._handle_magic("%agent mcp list")
        assert "No MCP servers configured" in listing


class TestSlashCommands:
    """Coverage for AvailableCommandsUpdate handling and %agent commands."""

    @pytest.mark.asyncio
    async def test_commands_empty(self, kernel):
        _, result = await kernel._handle_magic("%agent commands")
        assert "No slash commands" in result

    @pytest.mark.asyncio
    async def test_commands_listing(self, kernel):
        kernel.state.available_commands = [
            {"name": "mcp", "description": "Manage MCP servers", "input_hint": None},
            {"name": "init", "description": "Initialise the project", "input_hint": "<path>"},
        ]
        _, result = await kernel._handle_magic("%agent commands")
        assert "/mcp" in result
        assert "/init <path>" in result
        assert "Manage MCP servers" in result

    def test_complete_mcp_subcommand(self, kernel):
        out = kernel.do_complete("%agent mcp dis", len("%agent mcp dis"))
        assert "disable" in out["matches"]

    def test_no_special_slash_handling_in_complete(self, kernel):
        """Bare `/` text gets no kernel-level slash completion."""
        kernel.state.available_commands = [
            {"name": "mcp", "description": "", "input_hint": None},
        ]
        out = kernel.do_complete("/m", 2)
        assert out["matches"] == []

    @pytest.mark.asyncio
    async def test_agent_slash_is_unknown_subcommand(self, kernel):
        """`%agent /name` is no longer special — it's an unknown subcommand."""
        kernel._send_prompt = AsyncMock(return_value="")
        is_magic, result = await kernel._handle_magic("%agent /models")
        assert is_magic
        kernel._send_prompt.assert_not_awaited()
        assert "Unknown subcommand" in result


class TestSessionMagic:
    """Test %agent session subcommands."""

    @pytest.mark.asyncio
    async def test_session_info(self, kernel):
        """Test session info display."""
        is_magic, result = await kernel._handle_magic("%agent session info")
        assert "Session Information" in result
        assert "Working Directory" in result

    @pytest.mark.asyncio
    async def test_session_new(self, kernel):
        """Test creating new session."""
        cwd = os.getcwd()
        is_magic, result = await kernel._handle_magic(f"%agent session new {cwd}")
        assert "Created new session" in result
        assert kernel.state.cwd == cwd

    @pytest.mark.asyncio
    async def test_session_new_invalid_dir(self, kernel):
        """Test new session with invalid directory."""
        is_magic, result = await kernel._handle_magic("%agent session new /nonexistent/path")
        assert "Directory does not exist" in result

    @pytest.mark.asyncio
    async def test_session_restart(self, kernel):
        """Test session restart."""
        is_magic, result = await kernel._handle_magic("%agent session restart")
        assert "Session restarted" in result


class TestPermissionsMagic:
    """Test %agent permissions subcommands."""

    @pytest.mark.asyncio
    async def test_permissions_show(self, kernel):
        """Test showing current permission mode."""
        is_magic, result = await kernel._handle_magic("%agent permissions")
        assert "Permission mode: auto" in result

    @pytest.mark.asyncio
    async def test_permissions_set_manual(self, kernel):
        """Test setting manual permission mode."""
        is_magic, result = await kernel._handle_magic("%agent permissions manual")
        assert "Permission mode set to: manual" in result
        assert kernel.state.permission_mode == "manual"

    @pytest.mark.asyncio
    async def test_permissions_set_deny(self, kernel):
        """Test setting deny permission mode."""
        is_magic, result = await kernel._handle_magic("%agent permissions deny")
        assert "Permission mode set to: deny" in result
        assert kernel.state.permission_mode == "deny"

    @pytest.mark.asyncio
    async def test_permissions_set_invalid(self, kernel):
        """Test setting invalid permission mode."""
        is_magic, result = await kernel._handle_magic("%agent permissions invalid")
        assert "Unknown permission mode" in result

    @pytest.mark.asyncio
    async def test_permissions_history_empty(self, kernel):
        """Test empty permission history."""
        is_magic, result = await kernel._handle_magic("%agent permissions history")
        assert "No permission requests recorded" in result

    @pytest.mark.asyncio
    async def test_permissions_history_with_entries(self, kernel):
        """Test permission history with entries."""
        kernel.state.permission_history = [
            {"mode": "auto", "options": [], "tool_call": "test"}
        ]
        is_magic, result = await kernel._handle_magic("%agent permissions history")
        assert "Permission History" in result


class TestConfigMagic:
    """Test %agent config subcommand."""

    @pytest.mark.asyncio
    async def test_config_display(self, kernel):
        """Test config display."""
        is_magic, result = await kernel._handle_magic("%agent config")
        assert "Configuration" in result
        assert "Agent Command" in result
        assert "codex-acp" in result


class TestEnvMagic:
    """Test %agent env subcommand."""

    @pytest.mark.asyncio
    async def test_env_display(self, kernel):
        """Test environment variable display."""
        is_magic, result = await kernel._handle_magic("%agent env")
        assert "Environment Variables" in result
        assert "ACP_AGENT_COMMAND" in result

    @pytest.mark.asyncio
    async def test_env_hides_api_key(self, kernel):
        """Test that API keys are masked."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret123"}):
            is_magic, result = await kernel._handle_magic("%agent env")
            assert "***" in result
            assert "sk-secret123" not in result
