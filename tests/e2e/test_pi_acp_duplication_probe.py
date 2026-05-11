"""Probe: capture EVERY session_update from pi-acp for the duplication repro.

Sends the user's exact failing prompt ("what example notebooks do we
have?") to real pi-acp. Records every session_update object — type, the
raw text content if any — to ``/tmp/pi_acp_dup_probe.log`` so we can see
where the duplication is actually coming from (AgentMessageChunk text
vs ToolCallProgress content vs something else).

Skipped automatically if pi/pi-acp/LMStudio aren't available.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import urllib.request

from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from agent_client_kernel.kernel import ACPClientImpl, SessionState


PI = shutil.which("pi")
PI_ACP = shutil.which("pi-acp")


def _lms_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            "http://host.docker.internal:1234/v1/models", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    PI is None or PI_ACP is None or not _lms_reachable(),
    reason="needs pi, pi-acp, and a reachable LMStudio backend",
)


def _extract_text(content) -> str | None:
    if isinstance(content, TextContentBlock):
        return content.text
    if isinstance(content, list):
        parts = [
            c.text for c in content if isinstance(c, TextContentBlock)
        ]
        return "".join(parts) if parts else None
    return None


class _ProbeClient(ACPClientImpl):
    """ACP client that records every session_update verbatim."""

    def __init__(self, kernel) -> None:
        super().__init__(kernel)
        self.events: list[dict] = []

    async def session_update(self, *, session_id, update, **kwargs):
        record: dict = {"type": type(update).__name__}
        if isinstance(update, (AgentMessageChunk, AgentThoughtChunk)):
            record["text"] = _extract_text(update.content)
        elif isinstance(update, ToolCallStart):
            record["tool_call_id"] = update.tool_call_id
            record["title"] = update.title
            record["status"] = update.status
            record["text"] = _extract_text(update.content)
        elif isinstance(update, ToolCallProgress):
            record["tool_call_id"] = update.tool_call_id
            record["status"] = update.status
            record["text"] = _extract_text(update.content)
        else:
            try:
                record["repr"] = repr(update)[:300]
            except Exception:
                pass
        self.events.append(record)
        return await super().session_update(
            session_id=session_id, update=update, **kwargs
        )


@pytest.mark.asyncio
async def test_pi_acp_duplication_repro(tmp_path):
    """Reproduce the user's pi test 2 scenario and dump every event."""
    env = os.environ.copy()
    env.setdefault("PI_ACP_ENABLE_EMBEDDED_CONTEXT", "true")
    # Run pi-acp from the workspace so it sees the real examples/ dir.
    workspace = "/workspaces/agent-client-kernel"

    proc = await asyncio.create_subprocess_exec(
        PI_ACP,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace,
        env=env,
        limit=4 * 1024 * 1024,
    )
    assert proc.stdin is not None and proc.stdout is not None

    kernel = MagicMock()
    kernel.state = SessionState()
    kernel.iopub_socket = MagicMock()
    kernel.send_response = MagicMock()
    kernel._current_parent = None  # fallback path; we don't actually publish

    client = _ProbeClient(kernel)
    conn = connect_to_agent(client, proc.stdin, proc.stdout)

    try:
        await asyncio.wait_for(
            conn.initialize(protocol_version=PROTOCOL_VERSION), timeout=30
        )
        session = await asyncio.wait_for(
            conn.new_session(mcp_servers=[], cwd=workspace),
            timeout=60,
        )
        await asyncio.wait_for(
            conn.prompt(
                session_id=session.session_id,
                prompt=[text_block("what example notebooks do we have?")],
            ),
            timeout=240,
        )
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

    log_path = Path("/tmp/pi_acp_dup_probe.log")
    log_path.write_text(json.dumps(client.events, indent=2, ensure_ascii=False))

    # Also dump the running state.response_text so we see what the kernel
    # accumulated through dedup.
    Path("/tmp/pi_acp_dup_response.txt").write_text(
        kernel.state.response_text or ""
    )

    # Always-on summary so test output shows the smoking gun.
    by_type: dict[str, int] = {}
    for e in client.events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    print("\n=== pi-acp duplication probe summary ===")
    print("Event counts:", by_type)
    print("Log written to:", log_path)
    print("Final response_text length:", len(kernel.state.response_text or ""))

    assert client.events, "pi-acp emitted nothing"
