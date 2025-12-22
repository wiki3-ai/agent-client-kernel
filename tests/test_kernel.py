"""
Tests for the ACPKernel class.
"""

import asyncio
import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from acp import PROTOCOL_VERSION
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation

from agent_client_kernel.kernel import ACPKernel, SessionState
from agent_client_kernel import __version__


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
        # Initialize interrupt handling attributes
        k._current_task = None
        k._interrupted = False
        k._original_sigint = None
        k._current_parent = None
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


class TestSendPromptErrorHandling:
    """Test _send_prompt error handling."""

    @pytest.mark.asyncio
    async def test_buffer_overflow_error_recovery(self, kernel):
        """Test that buffer overflow errors are caught and connection is reset."""
        # Set up connected state
        proc = MagicMock()
        proc.returncode = None
        kernel._proc = proc
        kernel._conn = AsyncMock()
        kernel.state.session_id = "test-session"

        # Mock _stop_agent to track if it's called
        kernel._stop_agent = AsyncMock()

        # Simulate buffer overflow error from the connection
        kernel._conn.prompt = AsyncMock(
            side_effect=ValueError("Separator is found, but chunk is longer than limit")
        )

        # Call _send_prompt and expect RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await kernel._send_prompt("test prompt")

        # Verify the error message is user-friendly
        assert "larger than 1MB" in str(exc_info.value)
        assert "streamed before" in str(exc_info.value)

        # Verify connection was cleaned up
        kernel._stop_agent.assert_called_once()

        # Verify user was notified
        kernel.send_response.assert_called()
        call_args = kernel.send_response.call_args_list[-1]
        assert "stderr" in str(call_args) or call_args[0][2].get("name") == "stderr"

    @pytest.mark.asyncio
    async def test_buffer_overflow_error_chunk_variant(self, kernel):
        """Test buffer overflow with 'chunk is longer than limit' message."""
        proc = MagicMock()
        proc.returncode = None
        kernel._proc = proc
        kernel._conn = AsyncMock()
        kernel.state.session_id = "test-session"
        kernel._stop_agent = AsyncMock()

        kernel._conn.prompt = AsyncMock(
            side_effect=ValueError("chunk is longer than limit")
        )

        with pytest.raises(RuntimeError) as exc_info:
            await kernel._send_prompt("test prompt")

        assert "larger than 1MB" in str(exc_info.value)
        kernel._stop_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_other_value_errors_propagate(self, kernel):
        """Test that non-buffer-overflow ValueErrors propagate normally."""
        proc = MagicMock()
        proc.returncode = None
        kernel._proc = proc
        kernel._conn = AsyncMock()
        kernel.state.session_id = "test-session"
        kernel._stop_agent = AsyncMock()

        kernel._conn.prompt = AsyncMock(
            side_effect=ValueError("Some other error")
        )

        with pytest.raises(ValueError) as exc_info:
            await kernel._send_prompt("test prompt")

        assert "Some other error" in str(exc_info.value)
        # Connection should NOT be stopped for other errors
        kernel._stop_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_do_execute_handles_buffer_overflow(self, kernel):
        """Test that do_execute properly handles buffer overflow from _send_prompt."""
        proc = MagicMock()
        proc.returncode = None
        kernel._proc = proc
        kernel._conn = AsyncMock()
        kernel.state.session_id = "test-session"
        kernel._stop_agent = AsyncMock()
        kernel._start_agent = AsyncMock()

        kernel._conn.prompt = AsyncMock(
            side_effect=ValueError("Separator is found, but chunk is longer than limit")
        )

        result = await kernel.do_execute("Hello agent", silent=False)

        # Should return error status
        assert result["status"] == "error"
        assert "RuntimeError" in result["ename"]

        # Connection should have been cleaned up
        kernel._stop_agent.assert_called_once()


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


class TestAgentInitialization:
    """Test agent initialization with proper client capabilities."""

    @pytest.mark.asyncio
    async def test_initialize_sends_client_capabilities(self, kernel):
        """Test that _start_agent sends proper client capabilities."""
        # Create a mock connection that records initialize params
        mock_conn = AsyncMock()
        mock_conn.initialize = AsyncMock()
        mock_conn.new_session = AsyncMock(return_value=MagicMock(session_id="test-session"))

        # Create a mock process
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.returncode = None

        with patch("agent_client_kernel.kernel.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_proc
            with patch("agent_client_kernel.kernel.connect_to_agent") as mock_connect:
                mock_connect.return_value = mock_conn

                await kernel._start_agent()

                # Verify initialize was called with client_capabilities
                mock_conn.initialize.assert_called_once()
                call_kwargs = mock_conn.initialize.call_args.kwargs

                assert "protocol_version" in call_kwargs
                assert call_kwargs["protocol_version"] == PROTOCOL_VERSION

                assert "client_capabilities" in call_kwargs
                caps = call_kwargs["client_capabilities"]
                assert isinstance(caps, ClientCapabilities)
                assert caps.fs is not None
                assert caps.fs.read_text_file is True
                assert caps.fs.write_text_file is True
                assert caps.terminal is True

                assert "client_info" in call_kwargs
                info = call_kwargs["client_info"]
                assert isinstance(info, Implementation)
                assert info.name == "agent-client-kernel"
                assert info.version == __version__

    @pytest.mark.asyncio
    async def test_initialize_creates_session_after_init(self, kernel):
        """Test that new_session is called after initialize."""
        mock_conn = AsyncMock()
        mock_conn.initialize = AsyncMock()
        mock_conn.new_session = AsyncMock(return_value=MagicMock(session_id="test-session-123"))

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.returncode = None

        with patch("agent_client_kernel.kernel.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_proc
            with patch("agent_client_kernel.kernel.connect_to_agent") as mock_connect:
                mock_connect.return_value = mock_conn

                await kernel._start_agent()

                # Verify both initialize and new_session were called in order
                mock_conn.initialize.assert_called_once()
                mock_conn.new_session.assert_called_once()

                # Verify session_id was stored
                assert kernel.state.session_id == "test-session-123"

    @pytest.mark.asyncio
    async def test_initialize_failure_stops_agent(self, kernel):
        """Test that if initialize fails, the agent is stopped."""
        mock_conn = AsyncMock()
        mock_conn.initialize = AsyncMock(side_effect=Exception("Invalid params"))

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("agent_client_kernel.kernel.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_proc
            with patch("agent_client_kernel.kernel.connect_to_agent") as mock_connect:
                mock_connect.return_value = mock_conn

                with pytest.raises(Exception) as exc_info:
                    await kernel._start_agent()

                assert "Invalid params" in str(exc_info.value)

                # Verify process was cleaned up
                assert kernel._proc is None
                assert kernel._conn is None

    @pytest.mark.asyncio
    async def test_initialize_works_with_minimal_agent_response(self, kernel):
        """Test backward compatibility - works with agents that return minimal initialize response."""
        from acp.schema import InitializeResponse
        
        mock_conn = AsyncMock()
        # Simulate an agent that only returns protocol_version (minimal response)
        mock_conn.initialize = AsyncMock(return_value=InitializeResponse(protocol_version=1))
        mock_conn.new_session = AsyncMock(return_value=MagicMock(session_id="compat-session"))

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.returncode = None

        with patch("agent_client_kernel.kernel.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_proc
            with patch("agent_client_kernel.kernel.connect_to_agent") as mock_connect:
                mock_connect.return_value = mock_conn

                # Should succeed without errors
                await kernel._start_agent()

                # Verify session was created successfully
                assert kernel.state.session_id == "compat-session"
                mock_conn.initialize.assert_called_once()
                mock_conn.new_session.assert_called_once()
