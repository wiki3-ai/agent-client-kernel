"""Tests for ``agent_client_kernel.lmstudio_shim``.

The unit tests exercise ``flatten_tools`` without any network. The
integration test boots the shim against a stub upstream HTTP server and
verifies the body rewrite end-to-end.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_client_kernel import lmstudio_shim


# --- flatten_tools unit tests ----------------------------------------------


def test_flatten_tools_keeps_function_entries() -> None:
    payload = {
        "tools": [
            {"type": "function", "name": "shell", "description": "..."},
            {"type": "function", "name": "edit", "description": "..."},
        ]
    }
    out = lmstudio_shim.flatten_tools(payload)
    assert [t["name"] for t in out["tools"]] == ["shell", "edit"]
    assert all(t["type"] == "function" for t in out["tools"])


def test_flatten_tools_expands_namespace() -> None:
    payload = {
        "tools": [
            {"type": "function", "name": "shell"},
            {
                "type": "namespace",
                "name": "mcp__jupyter__",
                "tools": [
                    {"type": "function", "name": "list_notebooks"},
                    {"type": "function", "name": "execute_cell"},
                ],
            },
        ]
    }
    out = lmstudio_shim.flatten_tools(payload)
    names = [t["name"] for t in out["tools"]]
    assert names == ["shell", "mcp__jupyter__list_notebooks", "mcp__jupyter__execute_cell"]
    assert all(t["type"] == "function" for t in out["tools"])


def test_flatten_tools_drops_unknown_types() -> None:
    payload = {
        "tools": [
            {"type": "function", "name": "shell"},
            {"type": "web_search"},
            {"type": "image_generation"},
        ]
    }
    out = lmstudio_shim.flatten_tools(payload)
    assert [t["name"] for t in out["tools"]] == ["shell"]


def test_flatten_tools_no_tools_key_is_noop() -> None:
    payload = {"model": "x", "input": []}
    assert lmstudio_shim.flatten_tools(payload) == payload


def test_flatten_tools_non_list_tools_is_noop() -> None:
    payload = {"tools": "not-a-list"}
    assert lmstudio_shim.flatten_tools(payload) == {"tools": "not-a-list"}


def test_serve_forever_defaults_to_host_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With HOST_GATEWAY_IP set and no explicit override, target is derived."""
    monkeypatch.delenv("ACP_LMSTUDIO_SHIM_TARGET", raising=False)
    monkeypatch.delenv("SHIM_TARGET", raising=False)
    monkeypatch.setenv("HOST_GATEWAY_IP", "10.0.0.5")
    assert lmstudio_shim._env_target() == "http://10.0.0.5:1234"


def test_env_target_falls_back_to_host_docker_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HOST_GATEWAY_IP, no override -> host.docker.internal default."""
    monkeypatch.delenv("ACP_LMSTUDIO_SHIM_TARGET", raising=False)
    monkeypatch.delenv("SHIM_TARGET", raising=False)
    monkeypatch.delenv("HOST_GATEWAY_IP", raising=False)
    assert lmstudio_shim._env_target() == "http://host.docker.internal:1234"


def test_ensure_running_off_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACP_LMSTUDIO_SHIM", "off")
    monkeypatch.setattr(lmstudio_shim, "_BACKGROUND_THREAD", None)
    assert lmstudio_shim.ensure_running() is None


def test_ensure_running_starts_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACP_LMSTUDIO_SHIM", "auto")
    monkeypatch.setenv("ACP_LMSTUDIO_SHIM_TARGET", "http://127.0.0.1:1")
    monkeypatch.setenv("ACP_LMSTUDIO_SHIM_PORT", str(_free_port()))
    monkeypatch.setattr(lmstudio_shim, "_BACKGROUND_THREAD", None)
    target = lmstudio_shim.ensure_running()
    assert target == "http://127.0.0.1:1"
    assert lmstudio_shim._BACKGROUND_THREAD is not None
    assert lmstudio_shim._BACKGROUND_THREAD.is_alive()
    # Second call is a no-op while thread is alive.
    again = lmstudio_shim.ensure_running()
    assert again == "http://127.0.0.1:1"


# --- end-to-end with a stub upstream ---------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubUpstream:
    """Captures the most recent request body and returns a canned JSON response."""

    def __init__(self) -> None:
        self.last_body: bytes = b""
        self.last_path: str = ""
        self.port = _free_port()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                outer.last_body = self.rfile.read(length) if length else b""
                outer.last_path = self.path
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self.do_POST()

            def log_message(self, *_args, **_kwargs) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_StubUpstream":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def shim_against_stub():
    """Start the shim pointed at a fresh stub upstream on random ports."""
    with _StubUpstream() as upstream:
        shim_port = _free_port()
        server = lmstudio_shim._ThreadedServer(
            ("127.0.0.1", shim_port),
            lmstudio_shim._make_handler(f"http://127.0.0.1:{upstream.port}"),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Wait briefly for the socket to be accepting.
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", shim_port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        try:
            yield upstream, shim_port
        finally:
            server.shutdown()
            server.server_close()


def test_shim_flattens_request_body(shim_against_stub) -> None:
    upstream, shim_port = shim_against_stub
    payload = {
        "model": "qwen3-coder",
        "tools": [
            {"type": "function", "name": "shell"},
            {
                "type": "namespace",
                "name": "mcp__jupyter__",
                "tools": [
                    {"type": "function", "name": "list_notebooks"},
                ],
            },
        ],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{shim_port}/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read()) == {"ok": True}

    forwarded = json.loads(upstream.last_body)
    assert upstream.last_path == "/v1/responses"
    assert [t["name"] for t in forwarded["tools"]] == [
        "shell",
        "mcp__jupyter__list_notebooks",
    ]
    assert all(t["type"] == "function" for t in forwarded["tools"])


def test_shim_passes_through_non_json(shim_against_stub) -> None:
    _upstream, shim_port = shim_against_stub
    req = urllib.request.Request(
        f"http://127.0.0.1:{shim_port}/v1/models",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
