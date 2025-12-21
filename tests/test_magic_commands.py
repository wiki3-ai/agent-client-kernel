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
