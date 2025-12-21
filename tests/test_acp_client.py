"""
Tests for the ACPClientImpl class.
"""

import pytest
from unittest.mock import MagicMock

from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommand,
    AvailableCommandsUpdate,
    PlanEntry,
    PermissionOption,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
)

from agent_client_kernel.kernel import ACPClientImpl, SessionState


def make_text_chunk(text: str) -> AgentMessageChunk:
    """Helper to create AgentMessageChunk with required fields."""
    return AgentMessageChunk(
        session_update="agent_message_chunk",
        content=TextContentBlock(type="text", text=text),
    )


def make_thought_chunk(text: str) -> AgentThoughtChunk:
    """Helper to create AgentThoughtChunk with required fields."""
    return AgentThoughtChunk(
        session_update="agent_thought_chunk",
        content=TextContentBlock(type="text", text=text),
    )


class TestSessionUpdate:
    """Test session_update callback handling."""

    @pytest.mark.asyncio
    async def test_agent_message_chunk_text(self, acp_client, mock_kernel):
        """Test handling text message chunks."""
        chunk = make_text_chunk("Hello, world!")

        await acp_client.session_update(session_id="test", update=chunk)

        # Check text was accumulated
        assert mock_kernel.state.response_text == "Hello, world!"

        # Check stream output was sent
        mock_kernel.send_response.assert_called_once()
        call_args = mock_kernel.send_response.call_args
        assert call_args[0][1] == "stream"
        assert call_args[0][2]["name"] == "stdout"
        assert call_args[0][2]["text"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_agent_message_chunk_accumulates(self, acp_client, mock_kernel):
        """Test that multiple chunks accumulate."""
        chunks = [
            make_text_chunk("Hello"),
            make_text_chunk(" "),
            make_text_chunk("world!"),
        ]

        for chunk in chunks:
            await acp_client.session_update(session_id="test", update=chunk)

        assert mock_kernel.state.response_text == "Hello world!"

    @pytest.mark.asyncio
    async def test_agent_thought_chunk(self, acp_client, mock_kernel):
        """Test handling thought chunks."""
        chunk = make_thought_chunk("thinking...")

        await acp_client.session_update(session_id="test", update=chunk)

        # Thoughts go to stderr
        mock_kernel.send_response.assert_called_once()
        call_args = mock_kernel.send_response.call_args
        assert call_args[0][2]["name"] == "stderr"
        assert "💭" in call_args[0][2]["text"]

    @pytest.mark.asyncio
    async def test_tool_call_start(self, acp_client, mock_kernel):
        """Test handling tool call start."""
        update = ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc-123",
            title="Reading file",
            kind="read",
            status="in_progress",
        )

        await acp_client.session_update(session_id="test", update=update)

        assert "tc-123" in mock_kernel.state.tool_calls
        assert mock_kernel.state.tool_calls["tc-123"]["title"] == "Reading file"
        assert mock_kernel.state.tool_calls["tc-123"]["started"] is True

    @pytest.mark.asyncio
    async def test_tool_call_progress_completed(self, acp_client, mock_kernel):
        """Test handling tool call completion."""
        # First start the tool
        start = ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc-123",
            title="Test",
            kind="other",
            status="in_progress",
        )
        await acp_client.session_update(session_id="test", update=start)

        # Then complete it
        progress = ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="tc-123",
            title="Test",
            status="completed",
        )
        await acp_client.session_update(session_id="test", update=progress)

        assert mock_kernel.state.tool_calls["tc-123"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_available_commands_update(self, acp_client, mock_kernel):
        """Test handling available commands update."""
        update = AvailableCommandsUpdate(
            session_update="available_commands_update",
            available_commands=[
                AvailableCommand(name="help", description="Show help"),
                AvailableCommand(name="clear", description="Clear context"),
            ],
        )

        await acp_client.session_update(session_id="test", update=update)

        assert len(mock_kernel.state.available_commands) == 2
        assert mock_kernel.state.available_commands[0]["name"] == "help"
        assert mock_kernel.state.available_commands[1]["name"] == "clear"

    @pytest.mark.asyncio
    async def test_agent_plan_update(self, acp_client, mock_kernel):
        """Test handling plan updates."""
        update = AgentPlanUpdate(
            session_update="plan",
            entries=[
                PlanEntry(content="Step 1", status="completed", priority="high"),
                PlanEntry(content="Step 2", status="in_progress", priority="medium"),
                PlanEntry(content="Step 3", status="pending", priority="low"),
            ],
        )

        await acp_client.session_update(session_id="test", update=update)

        # Check that plan output was sent
        mock_kernel.send_response.assert_called_once()
        call_args = mock_kernel.send_response.call_args
        assert "📋" in call_args[0][2]["text"]


class TestRequestPermission:
    """Test permission request handling."""

    @pytest.mark.asyncio
    async def test_auto_mode_allows(self, acp_client, mock_kernel):
        """Test that auto mode selects allow option."""
        mock_kernel.state.permission_mode = "auto"

        options = [
            PermissionOption(kind="allow_once", name="Allow once", option_id="opt-1"),
            PermissionOption(kind="reject_once", name="Deny", option_id="opt-2"),
        ]
        tool_call = ToolCallUpdate(tool_call_id="tc-1", status="pending")

        response = await acp_client.request_permission(
            options=options, session_id="test", tool_call=tool_call
        )

        assert response.outcome.outcome == "selected"
        assert response.outcome.option_id == "opt-1"

    @pytest.mark.asyncio
    async def test_deny_mode_cancels(self, acp_client, mock_kernel):
        """Test that deny mode returns cancelled."""
        mock_kernel.state.permission_mode = "deny"

        options = [
            PermissionOption(kind="allow_once", name="Allow once", option_id="opt-1"),
        ]
        tool_call = ToolCallUpdate(tool_call_id="tc-1", status="pending")

        response = await acp_client.request_permission(
            options=options, session_id="test", tool_call=tool_call
        )

        assert response.outcome.outcome == "cancelled"

    @pytest.mark.asyncio
    async def test_permission_history_recorded(self, acp_client, mock_kernel):
        """Test that permissions are recorded in history."""
        options = [
            PermissionOption(kind="allow_once", name="Allow once", option_id="opt-1"),
        ]
        tool_call = ToolCallUpdate(tool_call_id="tc-1", status="pending")

        await acp_client.request_permission(
            options=options, session_id="test", tool_call=tool_call
        )

        assert len(mock_kernel.state.permission_history) == 1
        assert mock_kernel.state.permission_history[0]["mode"] == "auto"


class TestFileOperations:
    """Test file operation callbacks."""

    @pytest.mark.asyncio
    async def test_read_text_file(self, acp_client, tmp_path):
        """Test reading a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        response = await acp_client.read_text_file(
            path=str(test_file), session_id="test"
        )

        assert response.content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_read_text_file_with_lines(self, acp_client, tmp_path):
        """Test reading specific lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        response = await acp_client.read_text_file(
            path=str(test_file), session_id="test", line=2, limit=2
        )

        assert response.content == "line2\nline3\n"

    @pytest.mark.asyncio
    async def test_read_text_file_not_found(self, acp_client, tmp_path):
        """Test reading a non-existent file."""
        from acp import RequestError

        with pytest.raises(RequestError):
            await acp_client.read_text_file(
                path=str(tmp_path / "nonexistent.txt"), session_id="test"
            )

    @pytest.mark.asyncio
    async def test_write_text_file(self, acp_client, mock_kernel, tmp_path):
        """Test writing a file."""
        test_file = tmp_path / "output.txt"

        response = await acp_client.write_text_file(
            content="Hello, world!",
            path=str(test_file),
            session_id="test",
        )

        assert test_file.read_text() == "Hello, world!"

    @pytest.mark.asyncio
    async def test_write_text_file_creates_dirs(self, acp_client, mock_kernel, tmp_path):
        """Test that write creates parent directories."""
        test_file = tmp_path / "subdir" / "nested" / "output.txt"

        await acp_client.write_text_file(
            content="nested content",
            path=str(test_file),
            session_id="test",
        )

        assert test_file.read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_read_text_file_truncates_large_files(self, acp_client, mock_kernel, tmp_path):
        """Test that large files are truncated to prevent JSON-RPC buffer overflow."""
        from agent_client_kernel.kernel import ACPClientImpl
        
        # Create a file larger than MAX_FILE_CONTENT_SIZE
        large_content = "x" * (ACPClientImpl.MAX_FILE_CONTENT_SIZE + 10000)
        test_file = tmp_path / "large.txt"
        test_file.write_text(large_content)

        response = await acp_client.read_text_file(
            path=str(test_file), session_id="test"
        )

        # Content should be truncated
        assert len(response.content) < len(large_content)
        assert "truncated" in response.content
        # Warning should be sent
        mock_kernel.send_response.assert_called()

    @pytest.mark.asyncio
    async def test_read_text_file_truncates_at_line_boundary(self, acp_client, mock_kernel, tmp_path):
        """Test that truncation happens at line boundaries."""
        from agent_client_kernel.kernel import ACPClientImpl
        
        # Create a file with lines, larger than limit
        line = "This is a test line that is reasonably long.\n"
        num_lines = (ACPClientImpl.MAX_FILE_CONTENT_SIZE // len(line)) + 100
        large_content = line * num_lines
        test_file = tmp_path / "large_lines.txt"
        test_file.write_text(large_content)

        response = await acp_client.read_text_file(
            path=str(test_file), session_id="test"
        )

        # Content before the truncation message should end with newline
        truncation_marker = "... [truncated:"
        marker_pos = response.content.find(truncation_marker)
        assert marker_pos > 0
        content_before_marker = response.content[:marker_pos]
        assert content_before_marker.endswith("\n")

    @pytest.mark.asyncio
    async def test_read_text_file_small_file_not_truncated(self, acp_client, mock_kernel, tmp_path):
        """Test that small files are not truncated."""
        small_content = "Small file content\nwith multiple lines\n"
        test_file = tmp_path / "small.txt"
        test_file.write_text(small_content)

        response = await acp_client.read_text_file(
            path=str(test_file), session_id="test"
        )

        assert response.content == small_content
        assert "truncated" not in response.content
