"""
Pytest configuration and fixtures for agent-client-kernel tests.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.schema import (
    AgentMessageChunk,
    AvailableCommand,
    AvailableCommandsUpdate,
    NewSessionResponse,
    PromptResponse,
    TextContentBlock,
)

from agent_client_kernel.kernel import ACPClientImpl, ACPKernel, MCPServer, SessionState


@pytest.fixture
def mock_kernel() -> MagicMock:
    """Create a mock kernel for testing ACPClientImpl."""
    kernel = MagicMock(spec=ACPKernel)
    kernel.state = SessionState()
    kernel.iopub_socket = MagicMock()
    kernel.send_response = MagicMock()
    return kernel


@pytest.fixture
def acp_client(mock_kernel: MagicMock) -> ACPClientImpl:
    """Create an ACP client for testing.

    Tests that exercise line-buffered progressive streaming behavior need
    ``_stream_progress`` enabled (it defaults to off, behind the
    ``ACP_STREAM_PROGRESS`` env flag, since the live JupyterLab renderer
    duplicates progressive stream messages). Enabling it here keeps the
    existing buffering tests meaningful; tests that want default (off)
    behavior can flip the attribute back.
    """
    client = ACPClientImpl(mock_kernel)
    client._stream_progress = True
    return client


@pytest.fixture
def acp_client_no_progress(mock_kernel: MagicMock) -> ACPClientImpl:
    """ACP client with the default (off) progressive-streaming behavior."""
    client = ACPClientImpl(mock_kernel)
    client._stream_progress = False
    return client


@pytest.fixture
def session_state() -> SessionState:
    """Create a fresh session state."""
    return SessionState()


@pytest.fixture
def mcp_server() -> MCPServer:
    """Create a sample MCP server config."""
    return MCPServer(
        name="test-server",
        command="test-cmd",
        args=["--arg1", "--arg2"],
    )


class MockProcess:
    """Mock asyncio.subprocess.Process for testing."""

    def __init__(self, return_code: int | None = None):
        self.stdin = AsyncMock()
        self.stdout = AsyncMock()
        self.stderr = AsyncMock()
        self.returncode = return_code

    def terminate(self):
        pass

    def kill(self):
        pass

    async def wait(self):
        self.returncode = 0
        return 0


class MockConnection:
    """Mock ClientSideConnection for testing."""

    def __init__(self, session_id: str = "test-session"):
        self._session_id = session_id
        self._closed = False

    async def initialize(self, protocol_version: str) -> None:
        pass

    async def new_session(
        self,
        mcp_servers: list | None = None,
        cwd: str | None = None,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=self._session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list,
    ) -> PromptResponse:
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str) -> None:
        pass

    async def close(self) -> None:
        self._closed = True


@pytest.fixture
def mock_connection() -> MockConnection:
    """Create a mock connection."""
    return MockConnection()


@pytest.fixture
def mock_process() -> MockProcess:
    """Create a mock process."""
    return MockProcess()
