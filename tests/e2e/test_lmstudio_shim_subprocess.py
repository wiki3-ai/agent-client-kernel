"""End-to-end test: ``python -m agent_client_kernel.lmstudio_shim`` works.

This launches the shim as Codex would actually see it — a separate
process spawned via the module entry point — and exercises a Codex-shaped
request (Expect: 100-continue, namespace tools, SSE upstream). Opt-in:

    LMSTUDIO_SHIM_E2E=1 pytest tests/e2e/test_lmstudio_shim_subprocess.py -v

This guards against regressions where the in-process unit tests pass but
the standalone launcher path (env wiring, signal handling, module main)
breaks.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LMSTUDIO_SHIM_E2E") != "1",
    reason="set LMSTUDIO_SHIM_E2E=1 to run the shim subprocess e2e test",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"port {port} never became reachable")


@pytest.fixture
def stub_upstream():
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            captured["body"] = self.rfile.read(length) if length else b""
            captured["path"] = self.path
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"data: {\"type\":\"response.delta\",\"text\":\"hi\"}\n\n")
            self.wfile.flush()
            self.wfile.write(b"data: {\"type\":\"response.completed\"}\n\n")
            self.wfile.flush()

        def log_message(self, *_a, **_kw) -> None:
            return

    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, captured
    finally:
        srv.shutdown()
        srv.server_close()


def test_module_main_serves_codex_shaped_request(stub_upstream) -> None:
    upstream_port, captured = stub_upstream
    shim_port = _free_port()
    env = {
        **os.environ,
        "ACP_LMSTUDIO_SHIM_TARGET": f"http://127.0.0.1:{upstream_port}",
        "ACP_LMSTUDIO_SHIM_PORT": str(shim_port),
        "ACP_LMSTUDIO_SHIM": "auto",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_client_kernel.lmstudio_shim"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_port(shim_port)

        payload = json.dumps(
            {
                "model": "qwen3-coder",
                "tools": [
                    {"type": "function", "name": "shell"},
                    {
                        "type": "namespace",
                        "name": "mcp__jupyter__",
                        "tools": [{"type": "function", "name": "list_notebooks"}],
                    },
                ],
            }
        ).encode()
        head = (
            f"POST /v1/responses HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{shim_port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Expect: 100-continue\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()

        with socket.create_connection(("127.0.0.1", shim_port), timeout=10) as sock:
            sock.sendall(head)
            sock.settimeout(10)
            # 100 Continue line.
            line = b""
            while not line.endswith(b"\r\n"):
                line += sock.recv(1)
            assert line.startswith(b"HTTP/1.1 100"), line
            term = b""
            while not term.endswith(b"\r\n"):
                term += sock.recv(1)
            sock.sendall(payload)
            raw = b""
            while True:
                buf = sock.recv(4096)
                if not buf:
                    break
                raw += buf

        head_blob, _, body_blob = raw.partition(b"\r\n\r\n")
        assert head_blob.startswith(b"HTTP/1.1 200"), head_blob
        assert b"response.delta" in body_blob
        assert b"response.completed" in body_blob

        forwarded = json.loads(captured["body"])
        names = [t["name"] for t in forwarded["tools"]]
        assert names == ["shell", "mcp__jupyter__list_notebooks"]
        assert all(t["type"] == "function" for t in forwarded["tools"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
