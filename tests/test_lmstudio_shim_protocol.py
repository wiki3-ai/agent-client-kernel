"""HTTP-protocol integration tests for ``agent_client_kernel.lmstudio_shim``.

These cover the wire-level behaviour Codex (reqwest) relies on but that
the basic urllib roundtrip in ``test_lmstudio_shim.py`` doesn't exercise:

* ``Expect: 100-continue`` handshake — Codex sends Expect on large POSTs
  and aborts the request if no 100 ever arrives. Regression test for the
  "stream disconnected before completion: error sending request" failure.
* ``text/event-stream`` chunked passthrough — verifies the shim doesn't
  buffer the upstream response and that ``Connection: close`` is sent so
  the client can use EOF to mark end-of-stream.
* Upstream error paths (4xx body, connection refused -> 502).
* A combined Codex-shaped roundtrip: Expect header + namespace tools in
  body + streamed SSE response, asserting end-to-end correctness.

The tests speak raw HTTP over sockets so the actual wire bytes are
exercised, not just whatever the urllib/requests client happens to do.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

import pytest

from agent_client_kernel import lmstudio_shim


# --- helpers ---------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError(f"port {port} never became reachable")


class _UpstreamRunner:
    """Run a configurable upstream HTTP/1.1 server on a free port."""

    def __init__(self, handler_factory: Callable[[], type[BaseHTTPRequestHandler]]) -> None:
        self.port = _free_port()
        self.captured: dict[str, object] = {}
        cls = handler_factory()
        # Force HTTP/1.1 so streamed responses work.
        cls.protocol_version = "HTTP/1.1"
        self._server = HTTPServer(("127.0.0.1", self.port), cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_UpstreamRunner":
        self._thread.start()
        _wait_port(self.port)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class _ShimRunner:
    """Run the shim against a chosen upstream URL on a free port."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.port = _free_port()
        self._server = lmstudio_shim._ThreadedServer(
            ("127.0.0.1", self.port), lmstudio_shim._make_handler(target)
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_ShimRunner":
        self._thread.start()
        _wait_port(self.port)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _read_until(sock: socket.socket, sentinel: bytes, timeout: float = 5.0) -> bytes:
    sock.settimeout(timeout)
    buf = b""
    while sentinel not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def _read_line(sock: socket.socket, timeout: float = 5.0) -> bytes:
    sock.settimeout(timeout)
    buf = b""
    while not buf.endswith(b"\r\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf


# --- Expect: 100-continue regression ---------------------------------------


def _echo_json_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            resp = json.dumps({"echo": json.loads(body or b"{}")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *_a, **_kw) -> None:
            return

    return Handler


def test_shim_sends_100_continue_before_body() -> None:
    """Codex sends ``Expect: 100-continue``; the shim must respond with 100.

    Regression for the "error sending request" failure where the shim was
    speaking HTTP/1.0 and never honoured Expect, causing reqwest to abort.
    """
    with _UpstreamRunner(_echo_json_handler) as up, _ShimRunner(
        f"http://127.0.0.1:{up.port}"
    ) as shim:
        body = json.dumps({"hi": "there"}).encode()
        head = (
            f"POST /v1/responses HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{shim.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Expect: 100-continue\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()

        with socket.create_connection(("127.0.0.1", shim.port), timeout=5) as sock:
            sock.sendall(head)
            # Server must send 100 Continue before we send the body.
            interim = _read_line(sock)
            assert interim.startswith(b"HTTP/1.1 100"), (
                f"expected 100 Continue, got {interim!r}"
            )
            # Drain through end of interim headers (\r\n\r\n is sent after 100).
            # BaseHTTPServer emits "HTTP/1.1 100 Continue\r\n\r\n".
            terminator = _read_line(sock)
            assert terminator == b"\r\n"

            sock.sendall(body)
            # Now read the final response.
            raw = _read_until(sock, b"\r\n\r\n")
            assert raw.startswith(b"HTTP/1.1 200"), raw
            # The shim sends Content-Length when proxying JSON upstreams.
            tail = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                tail += chunk
            assert b'"echo"' in tail


# --- SSE streaming passthrough ---------------------------------------------


def _sse_streaming_handler(
    chunks: list[bytes], delay_s: float = 0.05
) -> Callable[[], type[BaseHTTPRequestHandler]]:
    def factory() -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(delay_s)

            def log_message(self, *_a, **_kw) -> None:
                return

        return Handler

    return factory


def test_shim_streams_sse_chunks_in_arrival_order() -> None:
    chunks = [
        b"data: {\"type\":\"response.delta\",\"text\":\"Hello\"}\n\n",
        b"data: {\"type\":\"response.delta\",\"text\":\" world\"}\n\n",
        b"data: {\"type\":\"response.completed\"}\n\n",
    ]
    with _UpstreamRunner(_sse_streaming_handler(chunks)) as up, _ShimRunner(
        f"http://127.0.0.1:{up.port}"
    ) as shim:
        body = b"{}"
        head = (
            f"POST /v1/responses HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{shim.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()

        with socket.create_connection(("127.0.0.1", shim.port), timeout=10) as sock:
            sock.sendall(head + body)
            raw = b""
            sock.settimeout(10)
            while True:
                buf = sock.recv(4096)
                if not buf:
                    break
                raw += buf

        # Split off headers and verify the body contains every chunk in order.
        header_blob, _, body_blob = raw.partition(b"\r\n\r\n")
        assert b"text/event-stream" in header_blob
        assert b"Connection: close" in header_blob
        assert b"\r\nContent-Length:" not in header_blob, (
            "shim must not advertise Content-Length on streamed responses"
        )
        idx = 0
        for chunk in chunks:
            found = body_blob.find(chunk, idx)
            assert found >= 0, f"missing chunk {chunk!r} after offset {idx}"
            idx = found + len(chunk)


# --- combined Codex-shaped roundtrip ---------------------------------------


def _capturing_sse_handler(
    captured: dict, chunks: list[bytes]
) -> Callable[[], type[BaseHTTPRequestHandler]]:
    def factory() -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                captured["body"] = self.rfile.read(length) if length else b""
                captured["path"] = self.path
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()

            def log_message(self, *_a, **_kw) -> None:
                return

        return Handler

    return factory


def test_codex_shaped_roundtrip() -> None:
    """Full Codex request shape: Expect header + namespace tools + SSE response."""
    captured: dict = {}
    chunks = [
        b"data: {\"type\":\"response.delta\",\"text\":\"hi\"}\n\n",
        b"data: {\"type\":\"response.completed\"}\n\n",
    ]
    payload = {
        "model": "qwen3-coder",
        "input": [{"role": "user", "content": "hi"}],
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
            {"type": "web_search"},
        ],
    }
    body = json.dumps(payload).encode()

    with _UpstreamRunner(_capturing_sse_handler(captured, chunks)) as up, _ShimRunner(
        f"http://127.0.0.1:{up.port}"
    ) as shim:
        head = (
            f"POST /v1/responses HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{shim.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Expect: 100-continue\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()

        with socket.create_connection(("127.0.0.1", shim.port), timeout=10) as sock:
            sock.sendall(head)
            interim = _read_line(sock)
            assert interim.startswith(b"HTTP/1.1 100"), interim
            assert _read_line(sock) == b"\r\n"
            sock.sendall(body)

            raw = b""
            sock.settimeout(10)
            while True:
                buf = sock.recv(4096)
                if not buf:
                    break
                raw += buf

    assert captured["path"] == "/v1/responses"
    forwarded = json.loads(captured["body"])
    names = [t["name"] for t in forwarded["tools"]]
    assert names == [
        "shell",
        "mcp__jupyter__list_notebooks",
        "mcp__jupyter__execute_cell",
    ]
    assert all(t["type"] == "function" for t in forwarded["tools"])

    # Client side: status 200, both SSE chunks streamed through.
    head_blob, _, body_blob = raw.partition(b"\r\n\r\n")
    assert head_blob.startswith(b"HTTP/1.1 200"), head_blob
    for chunk in chunks:
        assert chunk in body_blob


# --- error paths -----------------------------------------------------------


def _error_handler(status: int, msg: bytes) -> Callable[[], type[BaseHTTPRequestHandler]]:
    def factory() -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

            def log_message(self, *_a, **_kw) -> None:
                return

        return Handler

    return factory


def test_shim_propagates_upstream_4xx_with_body() -> None:
    err_body = b'{"error":{"message":"Invalid","param":"tools.0.type"}}'
    with _UpstreamRunner(_error_handler(400, err_body)) as up, _ShimRunner(
        f"http://127.0.0.1:{up.port}"
    ) as shim:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{shim.port}/v1/responses",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400
        assert exc.value.read() == err_body


def test_shim_returns_502_when_upstream_unreachable() -> None:
    unreachable_port = _free_port()  # nothing bound on it
    with _ShimRunner(f"http://127.0.0.1:{unreachable_port}") as shim:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{shim.port}/v1/responses",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 502
        payload = json.loads(exc.value.read())
        assert "shim upstream error" in payload["error"]["message"]
