"""End-to-end integration tests for the Pi ACP devcontainer.

These tests exercise the kernel's ACP client against the **real** ``pi-acp``
binary spawned as a subprocess. They are skipped automatically when ``pi``
and ``pi-acp`` aren't installed (e.g., on most local dev machines and on
the default unit-test CI matrix), so they are safe to commit.

The dedicated ``pi-acp-e2e`` CI job installs Node.js, ``pi`` and ``pi-acp``
and then runs this file.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from unittest.mock import MagicMock

import pytest

from acp import PROTOCOL_VERSION, connect_to_agent

from agent_client_kernel.kernel import ACPClientImpl, SessionState


PI = shutil.which("pi")
PI_ACP = shutil.which("pi-acp")

pytestmark = pytest.mark.skipif(
    PI is None or PI_ACP is None,
    reason="pi and pi-acp must be installed and on PATH for the e2e test",
)


def _make_mock_kernel() -> MagicMock:
    """Build a kernel test double accepted by ACPClientImpl."""
    kernel = MagicMock()
    kernel.state = SessionState()
    kernel.iopub_socket = MagicMock()
    kernel.send_response = MagicMock()
    return kernel


@pytest.mark.asyncio
async def test_pi_acp_handshake_and_new_session(tmp_path):
    """End-to-end ACP handshake against the real pi-acp adapter.

    Verifies that the kernel's ACP client can:

      1. spawn pi-acp,
      2. complete the ACP ``initialize`` handshake,
      3. create a new ACP session (which in turn forces pi-acp to spawn
         ``pi --mode rpc`` successfully),
      4. cleanly shut down.

    No model call is made: this test only validates that the agent kernel
    plumbing is wired up correctly. If any of the pi / pi-acp / ACP-SDK
    pieces are broken or incompatible, this test fails.
    """
    assert PI_ACP is not None  # for type checkers; pytestmark guarantees it.

    env = os.environ.copy()
    # PI_ACP_ENABLE_EMBEDDED_CONTEXT mirrors the devcontainer config.
    env.setdefault("PI_ACP_ENABLE_EMBEDDED_CONTEXT", "true")

    proc = await asyncio.create_subprocess_exec(
        PI_ACP,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(tmp_path),
        env=env,
        limit=1 * 1024 * 1024,
    )
    assert proc.stdin is not None and proc.stdout is not None

    client = ACPClientImpl(_make_mock_kernel())
    conn = connect_to_agent(client, proc.stdin, proc.stdout)

    try:
        await asyncio.wait_for(
            conn.initialize(protocol_version=PROTOCOL_VERSION), timeout=30
        )

        # Empty MCP server list: we only care that session creation succeeds.
        # pi-acp accepts MCP params but doesn't currently forward them to pi
        # (see pi-acp README "Limitations"); the dedicated jupyter-mcp-server
        # entry baked into ~/.pi/agent/mcp.json is wired through pi-mcp-adapter
        # instead, which is loaded by pi itself when the adapter is installed.
        session = await asyncio.wait_for(
            conn.new_session(mcp_servers=[], cwd=str(tmp_path)),
            timeout=60,
        )
        assert session.session_id, "pi-acp returned an empty session id"
    finally:
        try:
            await asyncio.wait_for(conn.close(), timeout=5)
        except Exception:
            pass

        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
