"""
Tests for the ACPKernel class.
"""

import asyncio
import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_client_kernel.kernel import ACPKernel, SessionState


@pytest.fixture
def kernel():
    """Create a kernel instance for testing."""
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
        k.execution_count = 1
        # Need to initialize the magic pattern
        k._magic_pattern = re.compile(r"^%(\w+)\s*(.*)?$", re.MULTILINE)
        return k


class TestKernelProperties:
    """Test kernel properties."""

    def test_is_connected_no_process(self, kernel):
        """Test is_connected when no process."""
        kernel._proc = None
        assert kernel.is_connected is False

    def test_is_connected_no_connection(self, kernel):
        """Test is_connected when no connection."""
        proc = MagicMock()
        proc.returncode = None
        kernel._proc = proc
        kernel._conn = None
        assert kernel.is_connected is False

    def test_is_connected_process_exited(self, kernel):
        """Test is_connected when process has exited."""
        proc = MagicMock()
        proc.returncode = 0
        kernel._proc = proc
        kernel._conn = MagicMock()
        assert kernel.is_connected is False

    def test_is_connected_true(self, kernel):
        """Test is_connected when properly connected."""
        proc = MagicMock()
        proc.returncode = None
        kernel._proc = proc
        kernel._conn = MagicMock()
        assert kernel.is_connected is True


class TestDoExecute:
    """Test do_execute method."""

    @pytest.mark.asyncio
    async def test_execute_empty_code(self, kernel):
        """Test executing empty code."""
        result = await kernel.do_execute("", silent=False)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_magic_command(self, kernel):
        """Test executing magic command."""
        result = await kernel.do_execute("%agent config", silent=False)
        assert result["status"] == "ok"
        # Check that output was sent
        kernel.send_response.assert_called()

    @pytest.mark.asyncio
    async def test_execute_magic_silent(self, kernel):
        """Test executing magic command silently."""
        result = await kernel.do_execute("%agent config", silent=True)
        assert result["status"] == "ok"
        # Should not send response when silent
        kernel.send_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_prompt_no_agent(self, kernel):
        """Test executing a prompt when no agent is connected."""
        # Without an agent connected, this will try to start one and fail
        # The exact behavior depends on whether codex-acp is available
        result = await kernel.do_execute("Hello", silent=False)

        # We just verify it returns a valid result structure
        assert result["status"] in ("ok", "error")
        assert "execution_count" in result


class TestDoShutdown:
    """Test do_shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_no_process(self, kernel):
        """Test shutdown when no process running."""
        kernel._proc = None
        result = await kernel.do_shutdown(restart=False)
        assert result["status"] == "ok"
        assert result["restart"] is False

    @pytest.mark.asyncio
    async def test_shutdown_with_restart(self, kernel):
        """Test shutdown with restart flag."""
        kernel._proc = None
        result = await kernel.do_shutdown(restart=True)
        assert result["restart"] is True


class TestDoComplete:
    """Test do_complete method."""

    def test_complete_agent_subcommand(self, kernel):
        """Test completion for %agent subcommands."""
        result = kernel.do_complete("%agent mc", cursor_pos=10)
        assert result["status"] == "ok"
        assert "mcp" in result["matches"]

    def test_complete_no_matches(self, kernel):
        """Test completion with no matches."""
        result = kernel.do_complete("regular code", cursor_pos=12)
        assert result["status"] == "ok"
        assert result["matches"] == []


class TestDoInspect:
    """Test do_inspect method."""

    def test_inspect_returns_not_found(self, kernel):
        """Test that inspect returns not found."""
        result = kernel.do_inspect("code", cursor_pos=2, detail_level=0)
        assert result["status"] == "ok"
        assert result["found"] is False


class TestSessionState:
    """Test SessionState dataclass."""

    def test_default_values(self):
        """Test default values are correct."""
        state = SessionState()
        assert state.session_id is None
        assert state.permission_mode == "auto"
        assert state.mcp_servers == []
        assert state.response_text == ""
        assert state.tool_calls == {}

    def test_custom_cwd(self):
        """Test custom cwd."""
        state = SessionState(cwd="/custom/path")
        assert state.cwd == "/custom/path"
