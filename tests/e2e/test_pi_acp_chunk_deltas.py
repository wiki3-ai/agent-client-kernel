"""Probe: capture exactly what pi-acp emits as agent_message_chunk text.

Runs the real ``pi-acp`` against the configured LMStudio backend, sends a
prompt that produces multi-line output, and dumps every received
``agent_message_chunk`` text to ``/tmp/pi_acp_chunks.log``.

Skipped automatically if ``pi``/``pi-acp`` aren't installed or if LMStudio
is unreachable.
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
from acp.schema import AgentMessageChunk, TextContentBlock

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


class _RecordingClient(ACPClientImpl):
    """ACP client that records every agent_message_chunk text as it arrives."""

    def __init__(self, kernel) -> None:
        super().__init__(kernel)
        self.chunk_log: list[str] = []

    async def session_update(self, *, session_id, update, **kwargs):
        if isinstance(update, AgentMessageChunk):
            content = update.content
            if isinstance(content, TextContentBlock):
                self.chunk_log.append(content.text)
        return await super().session_update(
            session_id=session_id, update=update, **kwargs
        )


@pytest.mark.asyncio
async def test_pi_acp_emits_true_deltas(tmp_path):
    """pi-acp must emit each agent_message_chunk as a delta, not cumulative.

    Property under test: for every pair of consecutive agent_message_chunk
    texts (a, b), b must NOT start with a (which would indicate b is the
    cumulative content including a). The total concatenation must form a
    coherent message.
    """
    env = os.environ.copy()
    env.setdefault("PI_ACP_ENABLE_EMBEDDED_CONTEXT", "true")

    proc = await asyncio.create_subprocess_exec(
        PI_ACP,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(tmp_path),
        env=env,
        limit=4 * 1024 * 1024,
    )
    assert proc.stdin is not None and proc.stdout is not None

    kernel = MagicMock()
    kernel.state = SessionState()
    kernel.iopub_socket = MagicMock()
    kernel.send_response = MagicMock()
    kernel._current_parent = None  # force fallback path so we don't actually send

    client = _RecordingClient(kernel)
    conn = connect_to_agent(client, proc.stdin, proc.stdout)

    try:
        await asyncio.wait_for(
            conn.initialize(protocol_version=PROTOCOL_VERSION), timeout=30
        )
        session = await asyncio.wait_for(
            conn.new_session(mcp_servers=[], cwd=str(tmp_path)),
            timeout=60,
        )

        # Prompt that should produce a multi-line markdown response without
        # invoking tools, to keep the test focused on text streaming.
        prompt_text = (
            "Without using any tools, reply with exactly this markdown table "
            "(and nothing else):\n\n"
            "| Notebook | Size |\n"
            "|----------|------|\n"
            "| a.ipynb | 1 KB |\n"
            "| b.ipynb | 2 KB |\n"
            "| c.ipynb | 3 KB |\n"
        )
        await asyncio.wait_for(
            conn.prompt(
                session_id=session.session_id,
                prompt=[text_block(prompt_text)],
            ),
            timeout=120,
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

    # Persist the raw log so a human can eyeball it.
    Path("/tmp/pi_acp_chunks.log").write_text(
        json.dumps(client.chunk_log, indent=2, ensure_ascii=False)
    )

    assert client.chunk_log, "pi-acp emitted no agent_message_chunk text"

    # Property: no chunk text is a prefix of the concatenation that came
    # before it (which would be the smoking gun for cumulative-not-delta).
    accumulated = ""
    for i, text in enumerate(client.chunk_log):
        if text and accumulated and text == accumulated:
            pytest.fail(
                f"chunk[{i}] is exactly the accumulated text: pi-acp is "
                f"sending cumulative content as deltas. text={text!r}"
            )
        if text and accumulated.startswith(text) and len(text) > 5:
            pytest.fail(
                f"chunk[{i}] is contained in accumulated content: possible "
                f"duplication. text={text!r}"
            )
        accumulated += text
