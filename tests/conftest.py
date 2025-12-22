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
    ClientCapabilities,
    FileSystemCapability,
    Implementation,
    InitializeResponse,
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
    """Create an ACP client for testing."""
    return ACPClientImpl(mock_kernel)


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
        self._initialized = False
        self._init_params = {}

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
    ) -> InitializeResponse:
        """Mock initialize that records the parameters passed."""
        self._initialized = True
        self._init_params = {
            "protocol_version": protocol_version,
            "client_capabilities": client_capabilities,
            "client_info": client_info,
        }
        return InitializeResponse(protocol_version=protocol_version)

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
