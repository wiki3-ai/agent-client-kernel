"""Wire-level tests for iopub stream messages produced by ``ACPClientImpl``.

These tests reproduce the exact code path used at runtime --
``self._kernel.session.send(self._kernel.iopub_socket, "stream", ...)`` --
using a *real* ``jupyter_client.session.Session`` and a fake socket that
captures the serialized multipart frames. They let us assert two properties
of what is actually published on iopub:

1. **Deltas only.** Each ``stream`` message text must be a fresh delta; no
   message may contain content that was already emitted by a prior message.
   (The cumulative "every line is preceded by all prior content" pattern
   observed in the JupyterLab UI would be visible here if the kernel were
   the source of the duplication.)
2. **Stable parent header.** Every iopub message published in response to a
   single execute request must carry the same ``parent_header`` so clients
   route them to the originating cell.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from jupyter_client.session import Session

from agent_client_kernel.kernel import ACPClientImpl, SessionState

from tests.test_acp_client import make_text_chunk


class _CapturingSocket:
    """Minimal stand-in for a ZMQ socket that records send_multipart calls."""

    def __init__(self) -> None:
        self.frames: list[list[bytes]] = []

    def send_multipart(self, frames, *args, **kwargs) -> None:  # noqa: D401
        # Session.send may pass an iterable; materialize for inspection.
        self.frames.append([bytes(f) for f in frames])


def _make_real_session_kernel() -> tuple[MagicMock, _CapturingSocket, Session, dict]:
    """Build a kernel double that uses a real Session + capturing socket."""
    session = Session(key=b"test-key", username="test")
    socket = _CapturingSocket()

    parent_header = {
        "msg_id": "parent-msg-id",
        "msg_type": "execute_request",
        "username": "test",
        "session": "parent-session",
        "date": "2026-05-10T00:00:00Z",
        "version": "5.3",
    }
    parent = {"header": parent_header, "parent_header": {}, "metadata": {}, "content": {}}

    kernel = MagicMock()
    kernel.state = SessionState()
    kernel.session = session
    kernel.iopub_socket = socket
    # ACPClientImpl reads this attribute via getattr().
    kernel._current_parent = parent
    # send_response is the fallback path; not expected to be used here.
    kernel.send_response = MagicMock()
    return kernel, socket, session, parent_header


def _decode_stream_messages(socket: _CapturingSocket, session: Session) -> list[dict]:
    """Deserialize captured frames into messages, keeping only ``stream`` ones."""
    messages: list[dict] = []
    for frames in socket.frames:
        # Session.feed_identities strips the routing prefix up to the
        # DELIM frame ("<IDS|MSG>"), then unserialize parses the rest.
        idents, body = session.feed_identities(list(frames), copy=True)
        # Reset signature digest history so re-deserializing for inspection
        # mid-test (after additional sends) doesn't trip the replay guard.
        session.digest_history.clear()
        msg = session.deserialize(body, content=True, copy=True)
        if msg["header"]["msg_type"] == "stream":
            messages.append(msg)
    return messages


@pytest.mark.asyncio
async def test_iopub_stream_emits_deltas_only_on_wire():
    """Each iopub stream frame contains a delta, never cumulative content.

    Replays the token-by-token emission pattern observed from pi-acp
    (``message_update`` text_delta events) for a multi-line markdown table
    and asserts that concatenating the wire-level texts equals the original
    response, with no overlap between consecutive messages.
    """
    kernel, socket, session, parent_header = _make_real_session_kernel()
    client = ACPClientImpl(kernel)
    client._stream_progress = True

    # Token stream that mirrors the user-reported reproduction:
    # an intro line, blank line, then a markdown table built up token by token.
    response = (
        "Here are the example notebooks in your `examples/` directory:\n"
        "\n"
        "| Notebook | Size |\n"
        "|----------|------|\n"
        "| **a.ipynb** | 1 KB |\n"
        "| **b.ipynb** | 2 KB |\n"
    )
    # Split into small pieces, sometimes mid-line, sometimes containing newlines.
    tokens = [
        "Here are the example notebooks ",
        "in your `examples/` directory:\n",
        "\n",
        "| Notebook ",
        "| Size |\n",
        "|----------|------|\n",
        "| **a.ipynb** | 1 KB |\n",
        "| **b.ipynb** ",
        "| 2 KB |\n",
    ]
    assert "".join(tokens) == response

    for tok in tokens:
        await client.session_update(session_id="s", update=make_text_chunk(tok))
    client._flush_streams()

    msgs = _decode_stream_messages(socket, session)
    assert msgs, "expected at least one stream message on iopub"

    texts = [m["content"]["text"] for m in msgs]

    # Property 1: concatenation of wire-level texts equals the response.
    assert "".join(texts) == response

    # Property 2: no message's text appears as a prefix-overlap with a
    # prior message's text. The cumulative-duplication symptom would show
    # up here as message N starting with message N-1's text.
    seen = ""
    for text in texts:
        assert not text.startswith(seen) or seen == "", (
            f"stream message duplicates prior cumulative content: "
            f"prior={seen!r} current={text!r}"
        )
        seen = text

    # Property 3: every stream message carries the same parent header so
    # clients route them to the originating cell.
    for m in msgs:
        assert m["parent_header"]["msg_id"] == parent_header["msg_id"]
        assert m["parent_header"]["session"] == parent_header["session"]


@pytest.mark.asyncio
async def test_iopub_stream_one_line_per_message():
    """Each emitted iopub stream message corresponds to a complete line.

    The line-buffering in ``_send_stream`` should produce one message per
    newline boundary; partial lines stay buffered until ``_flush_streams``.
    """
    kernel, socket, session, _ = _make_real_session_kernel()
    client = ACPClientImpl(kernel)
    client._stream_progress = True

    for tok in ["foo", " bar\n", "baz", " qux\n", "trailing"]:
        await client.session_update(session_id="s", update=make_text_chunk(tok))

    msgs = _decode_stream_messages(socket, session)
    # Two complete lines flushed by newline; trailing partial still buffered.
    assert [m["content"]["text"] for m in msgs] == ["foo bar\n", "baz qux\n"]

    client._flush_streams()
    msgs = _decode_stream_messages(socket, session)
    # _flush_streams adds a synthetic newline to the trailing partial.
    assert [m["content"]["text"] for m in msgs] == [
        "foo bar\n",
        "baz qux\n",
        "trailing\n",
    ]


@pytest.mark.asyncio
async def test_iopub_stream_default_no_progressive_publishing():
    """With progressive streaming OFF (the default), no stream message is
    published mid-prompt; the entire response is emitted as a single
    coalesced message at end-of-turn flush.
    """
    kernel, socket, session, _ = _make_real_session_kernel()
    client = ACPClientImpl(kernel)  # default: _stream_progress is False

    for tok in [
        "Here are the example notebooks ",
        "in your `examples/` directory:\n",
        "\n",
        "| Notebook | Size |\n",
        "|----------|------|\n",
        "| **a.ipynb** | 1 KB |\n",
    ]:
        await client.session_update(session_id="s", update=make_text_chunk(tok))

    # Nothing on the wire yet.
    assert _decode_stream_messages(socket, session) == []

    client._flush_streams()
    msgs = _decode_stream_messages(socket, session)
    # Exactly one stream message carrying the full response.
    assert len(msgs) == 1
    assert msgs[0]["content"]["name"] == "stdout"
    assert msgs[0]["content"]["text"] == (
        "Here are the example notebooks in your `examples/` directory:\n"
        "\n"
        "| Notebook | Size |\n"
        "|----------|------|\n"
        "| **a.ipynb** | 1 KB |\n"
    )


@pytest.mark.asyncio
async def test_iopub_parent_header_correct_across_async_tasks():
    """Parent header is preserved when chunks arrive from a different task.

    At runtime, the ACP connection's reader task delivers ``session_update``
    callbacks; the parent header was captured by ``do_execute`` running on a
    different task. The kernel passes ``parent`` explicitly to ``session.send``
    so the parent_header on the wire must match regardless of which task runs
    the callback.
    """
    import asyncio

    kernel, socket, session, parent_header = _make_real_session_kernel()
    client = ACPClientImpl(kernel)
    client._stream_progress = True

    async def deliver_chunks() -> None:
        # This coroutine is scheduled as a separate Task so it has its own
        # async context (no inherited shell-parent ContextVar).
        for tok in ["line A\n", "line B\n"]:
            await client.session_update(session_id="s", update=make_text_chunk(tok))

    await asyncio.create_task(deliver_chunks())
    client._flush_streams()

    msgs = _decode_stream_messages(socket, session)
    assert [m["content"]["text"] for m in msgs] == ["line A\n", "line B\n"]
    for m in msgs:
        assert m["parent_header"]["msg_id"] == parent_header["msg_id"], (
            "iopub stream lost its execute_request parent when delivered from "
            "a different asyncio task"
        )
