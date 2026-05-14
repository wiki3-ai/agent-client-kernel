"""Tests for the MCP namespace ↔ flat-name translation in ``lmstudio_shim``.

Background
----------
Codex 0.130 advertises MCP tools to the Responses API as a *namespace*
envelope::

    {"type":"namespace","name":"mcp__jupyter__",
     "description":"...","tools":[{"type":"function","name":"list_notebooks",...}]}

Codex's protocol type for a function call is::

    pub enum ResponseItem {
        FunctionCall {
            name: String,
            namespace: Option<String>,
            arguments: String,
            call_id: String,
            ...
        },
        ...
    }

(see codex-rs/protocol/src/models.rs in openai/codex). When the model returns
a function call, codex routes it via
``ToolName::new(namespace, name)`` (codex-rs/core/src/tools/router.rs). The
registered MCP handler is keyed by
``ToolName { namespace: Some("mcp__jupyter__"), name: "list_notebooks" }``,
so a function call without a ``namespace`` field is looked up as
``ToolName { namespace: None, name: "mcp__jupyter__list_notebooks" }`` and
fails with ``unsupported call: mcp__jupyter__list_notebooks``.

LM Studio's OpenAI-compat layer doesn't understand the ``namespace``
envelope, so the shim flattens it to plain function tools on the request
side. The model therefore replies with a flat name like
``mcp__jupyter__list_notebooks``. The shim must split that flat name
back into ``namespace="mcp__jupyter__"`` and ``name="list_notebooks"``
before codex sees it; conversely, when codex sends conversation history
back upstream, any ``function_call`` items that already carry a
``namespace`` field must be flattened (concatenated and the field
dropped) so LM Studio doesn't choke on the unknown field.
"""

from __future__ import annotations

import json

import pytest

from agent_client_kernel import lmstudio_shim


# --- helpers exported by the shim ------------------------------------------


def test_split_flat_name_strips_known_namespace() -> None:
    ns, name = lmstudio_shim.split_flat_name(
        "mcp__jupyter__list_notebooks", ["mcp__jupyter__"]
    )
    assert ns == "mcp__jupyter__"
    assert name == "list_notebooks"


def test_split_flat_name_handles_inner_underscores() -> None:
    """Tool names may legitimately contain ``__`` runs."""
    ns, name = lmstudio_shim.split_flat_name(
        "mcp__jupyter__execute__cell", ["mcp__jupyter__"]
    )
    assert ns == "mcp__jupyter__"
    assert name == "execute__cell"


def test_split_flat_name_unknown_namespace_returns_none() -> None:
    ns, name = lmstudio_shim.split_flat_name(
        "shell", ["mcp__jupyter__"]
    )
    assert ns is None
    assert name == "shell"


def test_split_flat_name_no_namespaces_known() -> None:
    ns, name = lmstudio_shim.split_flat_name(
        "mcp__jupyter__list_notebooks", []
    )
    assert ns is None
    assert name == "mcp__jupyter__list_notebooks"


# --- collect_namespaces from a request payload -----------------------------


def test_collect_namespaces_picks_up_namespace_envelopes() -> None:
    payload = {
        "tools": [
            {"type": "function", "name": "shell"},
            {
                "type": "namespace",
                "name": "mcp__jupyter__",
                "tools": [{"type": "function", "name": "list_notebooks"}],
            },
            {
                "type": "namespace",
                "name": "mcp__weather__",
                "tools": [{"type": "function", "name": "get_forecast"}],
            },
        ]
    }
    namespaces = lmstudio_shim.collect_namespaces(payload)
    assert sorted(namespaces) == ["mcp__jupyter__", "mcp__weather__"]


def test_collect_namespaces_empty_when_no_namespace_envelopes() -> None:
    payload = {"tools": [{"type": "function", "name": "shell"}]}
    assert lmstudio_shim.collect_namespaces(payload) == []


# --- request-side input-history flattening ---------------------------------


def test_flatten_input_history_combines_namespace_and_name() -> None:
    """Codex's ``input`` array on follow-up turns may contain previously
    emitted ``function_call`` items already split into ``namespace`` +
    ``name``. LM Studio doesn't know the ``namespace`` field, so combine
    them back into a single flat ``name`` and drop the field."""
    payload = {
        "input": [
            {
                "type": "function_call",
                "name": "list_notebooks",
                "namespace": "mcp__jupyter__",
                "arguments": "{}",
                "call_id": "call_1",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "...",
            },
        ]
    }
    out = lmstudio_shim.flatten_input_history(payload)
    item = out["input"][0]
    assert item["name"] == "mcp__jupyter__list_notebooks"
    assert "namespace" not in item
    # function_call_output passes through untouched.
    assert out["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "...",
    }


def test_flatten_input_history_passthrough_when_no_namespace() -> None:
    payload = {
        "input": [
            {
                "type": "function_call",
                "name": "shell",
                "arguments": "{}",
                "call_id": "c",
            }
        ]
    }
    out = lmstudio_shim.flatten_input_history(payload)
    assert out["input"][0] == payload["input"][0]


# --- response-side rewrite of a JSON object --------------------------------


def test_rewrite_response_function_call_splits_known_namespace() -> None:
    obj = {
        "type": "function_call",
        "id": "fc_1",
        "name": "mcp__jupyter__list_notebooks",
        "arguments": "{}",
        "call_id": "call_1",
    }
    lmstudio_shim.rewrite_response_obj(obj, ["mcp__jupyter__"])
    assert obj["name"] == "list_notebooks"
    assert obj["namespace"] == "mcp__jupyter__"
    # Other keys preserved.
    assert obj["call_id"] == "call_1"
    assert obj["arguments"] == "{}"


def test_rewrite_response_function_call_unknown_namespace_unchanged() -> None:
    obj = {
        "type": "function_call",
        "name": "shell",
        "arguments": "{}",
        "call_id": "c",
    }
    snapshot = json.loads(json.dumps(obj))
    lmstudio_shim.rewrite_response_obj(obj, ["mcp__jupyter__"])
    assert obj == snapshot


def test_rewrite_response_walks_nested_items() -> None:
    """Responses-API events nest the function_call inside ``item`` /
    ``output`` / ``response.output[]``; the rewriter must descend."""
    event = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "name": "mcp__jupyter__list_notebooks",
            "arguments": "{}",
            "call_id": "call_1",
        },
    }
    lmstudio_shim.rewrite_response_obj(event, ["mcp__jupyter__"])
    assert event["item"]["name"] == "list_notebooks"
    assert event["item"]["namespace"] == "mcp__jupyter__"


def test_rewrite_response_walks_response_output_list() -> None:
    event = {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "output": [
                {
                    "type": "function_call",
                    "name": "mcp__jupyter__list_notebooks",
                    "arguments": "{}",
                    "call_id": "c",
                }
            ],
        },
    }
    lmstudio_shim.rewrite_response_obj(event, ["mcp__jupyter__"])
    fc = event["response"]["output"][0]
    assert fc["name"] == "list_notebooks"
    assert fc["namespace"] == "mcp__jupyter__"


# --- SSE stream rewriter ---------------------------------------------------


def _sse_event(obj: dict, event_name: str | None = None) -> bytes:
    head = f"event: {event_name}\n" if event_name else ""
    return (head + "data: " + json.dumps(obj) + "\n\n").encode("utf-8")


def test_sse_rewriter_rewrites_complete_event() -> None:
    rw = lmstudio_shim.SSERewriter(["mcp__jupyter__"])
    chunk = _sse_event(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "mcp__jupyter__list_notebooks",
                "arguments": "{}",
                "call_id": "c",
            },
        },
        event_name="response.output_item.done",
    )
    out = rw.feed(chunk)
    text = out.decode("utf-8")
    assert "data: " in text
    # Find the data: line and parse it.
    data_line = [
        line for line in text.split("\n") if line.startswith("data: ")
    ][0]
    obj = json.loads(data_line[len("data: "):])
    assert obj["item"]["name"] == "list_notebooks"
    assert obj["item"]["namespace"] == "mcp__jupyter__"


def test_sse_rewriter_buffers_partial_event() -> None:
    rw = lmstudio_shim.SSERewriter(["mcp__jupyter__"])
    full = _sse_event(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "mcp__jupyter__list_notebooks",
                "arguments": "{}",
                "call_id": "c",
            },
        }
    )
    # Split mid-event.
    split = len(full) // 2
    first = rw.feed(full[:split])
    # Nothing complete yet.
    assert b"data:" not in first or b"\n\n" not in first
    second = rw.feed(full[split:])
    combined = (first + second).decode("utf-8")
    data_line = [
        line for line in combined.split("\n") if line.startswith("data: ")
    ][0]
    obj = json.loads(data_line[len("data: "):])
    assert obj["item"]["name"] == "list_notebooks"
    assert obj["item"]["namespace"] == "mcp__jupyter__"


def test_sse_rewriter_passthrough_done_sentinel() -> None:
    """Some servers send ``data: [DONE]\\n\\n`` which is not JSON."""
    rw = lmstudio_shim.SSERewriter(["mcp__jupyter__"])
    out = rw.feed(b"data: [DONE]\n\n")
    assert out == b"data: [DONE]\n\n"


def test_sse_rewriter_passthrough_non_function_event() -> None:
    rw = lmstudio_shim.SSERewriter(["mcp__jupyter__"])
    chunk = _sse_event({"type": "response.created", "response": {"id": "r"}})
    out = rw.feed(chunk)
    assert out == chunk


def test_sse_rewriter_flush_returns_pending() -> None:
    """Trailing bytes without a terminating blank line should be flushed
    via :meth:`SSERewriter.flush`. (A real server always terminates, but
    flush guarantees we don't drop data on connection close.)"""
    rw = lmstudio_shim.SSERewriter(["mcp__jupyter__"])
    out = rw.feed(b"data: trailing-no-newline")
    assert out == b""
    flushed = rw.flush()
    assert flushed == b"data: trailing-no-newline"


def test_sse_rewriter_handles_multiple_events_in_one_chunk() -> None:
    rw = lmstudio_shim.SSERewriter(["mcp__jupyter__"])
    e1 = _sse_event({"type": "response.created", "response": {"id": "r"}})
    e2 = _sse_event(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "mcp__jupyter__list_notebooks",
                "arguments": "{}",
                "call_id": "c",
            },
        }
    )
    out = rw.feed(e1 + e2)
    text = out.decode("utf-8")
    # Two data: lines, second one rewritten.
    data_lines = [
        line for line in text.split("\n") if line.startswith("data: ")
    ]
    assert len(data_lines) == 2
    obj2 = json.loads(data_lines[1][len("data: "):])
    assert obj2["item"]["name"] == "list_notebooks"
    assert obj2["item"]["namespace"] == "mcp__jupyter__"


def test_sse_rewriter_no_namespaces_is_passthrough() -> None:
    """When no MCP tools were advertised in the request, the rewriter
    should be a pure passthrough — no JSON parsing, no mutation."""
    rw = lmstudio_shim.SSERewriter([])
    chunk = _sse_event(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "mcp__jupyter__list_notebooks",
                "arguments": "{}",
                "call_id": "c",
            },
        }
    )
    out = rw.feed(chunk)
    assert out == chunk
