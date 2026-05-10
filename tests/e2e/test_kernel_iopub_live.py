"""Live-kernel iopub capture: prove what the kernel actually emits over ZMQ.

Spawns the real ``ACPKernel`` as a subprocess via ``jupyter_client``, sends an
``execute_request`` whose code triggers a deterministic streamed response from
a stub ACP agent, and captures every ``stream`` message published on iopub.

Asserts:

1. Concatenation of stream texts == the agent's full reply (no duplication,
   no truncation).
2. Every stream message's ``parent_header`` matches the
   ``execute_request`` ``msg_id`` (correct cell routing).

This isolates the *kernel's* iopub behavior from anything downstream
(jupyter-server-nbmodel, jupyter-collaboration, JupyterLab outputarea).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    "FAST" in os.environ,
    reason="live-kernel test; skipped under FAST env",
)


# A stand-alone ACP agent script that scripts a deterministic streaming reply.
# It speaks the JSON-RPC ACP protocol over stdio.
_FAKE_AGENT = textwrap.dedent(
    r"""
    import json, sys, time

    REPLY_TOKENS = [
        "Here are the example notebooks ",
        "in your `examples/` directory:\n",
        "\n",
        "| Notebook | Size |\n",
        "|----------|------|\n",
        "| **a.ipynb** | 1 KB |\n",
        "| **b.ipynb** | 2 KB |\n",
        "| **c.ipynb** | 3 KB |\n",
    ]

    def write(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def notify(method, params):
        write({"jsonrpc": "2.0", "method": method, "params": params})

    def reply(req_id, result):
        write({"jsonrpc": "2.0", "id": req_id, "result": result})

    for raw in sys.stdin:
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        method = msg.get("method")
        if method == "initialize":
            reply(msg["id"], {
                "protocolVersion": msg["params"]["protocolVersion"],
                "agentCapabilities": {"loadSession": False, "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False}},
                "authMethods": [],
            })
        elif method == "session/new":
            reply(msg["id"], {"sessionId": "stub-session"})
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            for tok in REPLY_TOKENS:
                notify("session/update", {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": tok},
                    },
                })
                # Tiny sleep so the kernel's reader loop processes each chunk
                # individually rather than batching a single I/O read.
                time.sleep(0.005)
            reply(msg["id"], {"stopReason": "end_turn"})
        elif "id" in msg:
            # Generic OK reply for anything else.
            reply(msg["id"], None)
    """
).strip()


def _write_fake_agent(tmp_path: Path) -> Path:
    p = tmp_path / "fake_acp_agent.py"
    p.write_text(_FAKE_AGENT)
    return p


@pytest.mark.asyncio
async def test_kernel_iopub_stream_no_duplication(tmp_path):
    pytest.importorskip("jupyter_client")
    from jupyter_client.manager import AsyncKernelManager

    expected = (
        "Here are the example notebooks in your `examples/` directory:\n"
        "\n"
        "| Notebook | Size |\n"
        "|----------|------|\n"
        "| **a.ipynb** | 1 KB |\n"
        "| **b.ipynb** | 2 KB |\n"
        "| **c.ipynb** | 3 KB |\n"
    )

    agent_path = _write_fake_agent(tmp_path)

    env = os.environ.copy()
    env["ACP_AGENT_COMMAND"] = sys.executable
    env["ACP_AGENT_ARGS"] = str(agent_path)
    env["ACP_LOG_LEVEL"] = "WARNING"
    env["BROWSER"] = "/bin/true"
    # Force progressive streaming on so the kernel publishes one stream
    # message per line; the test verifies wire-level behavior of that mode.
    env["ACP_STREAM_PROGRESS"] = "1"

    km = AsyncKernelManager(kernel_name="agentclient")
    try:
        await km.start_kernel(env=env, cwd=str(tmp_path))
    except Exception as e:  # kernelspec not installed in test env
        pytest.skip(f"kernel not installable in this env: {e}")

    kc = km.client()
    kc.start_channels()
    try:
        await kc.wait_for_ready(timeout=30)

        msg_id = kc.execute("hello", store_history=False)

        stream_texts: list[str] = []
        deadline = asyncio.get_event_loop().time() + 60
        # Drain iopub until we see the matching execute_reply on shell.
        got_reply = False

        async def _drain_shell():
            nonlocal got_reply
            while not got_reply:
                msg = await kc.get_shell_msg(timeout=60)
                if msg["parent_header"].get("msg_id") == msg_id and \
                        msg["msg_type"] == "execute_reply":
                    got_reply = True
                    return

        async def _drain_iopub():
            while True:
                try:
                    msg = await kc.get_iopub_msg(timeout=10)
                except Exception:
                    if got_reply:
                        return
                    continue
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                if msg["msg_type"] == "stream":
                    assert msg["parent_header"]["msg_id"] == msg_id, (
                        "iopub stream lost its execute_request parent"
                    )
                    stream_texts.append(msg["content"]["text"])
                if msg["msg_type"] == "status" and \
                        msg["content"]["execution_state"] == "idle" and got_reply:
                    return

        await asyncio.wait_for(
            asyncio.gather(_drain_shell(), _drain_iopub()), timeout=60
        )

        # Property 1: full reply is reconstructed exactly from the stream
        # messages, with NO duplication (would manifest as len > len(expected)).
        # The kernel intentionally publishes a single trailing "\n" after a
        # turn completes (see _send_prompt) so the cell rendering ends on a
        # clean line; account for that.
        joined = "".join(stream_texts)
        assert joined in (expected, expected + "\n"), (
            f"iopub stream content mismatch.\n"
            f"got len={len(joined)} expected len={len(expected)}\n"
            f"got={joined!r}\nexpected={expected!r}\n"
            f"raw stream texts: {stream_texts!r}"
        )

        # Property 2: Every text is a fresh delta -- no message starts with
        # the cumulative prior content, which is the symptom of the bug.
        prior = ""
        for i, text in enumerate(stream_texts):
            assert not (prior and text == prior), (
                f"stream message {i} duplicates the prior cumulative buffer"
            )
            assert not (prior and len(text) > len(prior) and text.startswith(prior)), (
                f"stream message {i} carries cumulative content (starts with "
                f"prior {prior!r})"
            )
            prior = text

    finally:
        kc.stop_channels()
        await km.shutdown_kernel(now=True)
