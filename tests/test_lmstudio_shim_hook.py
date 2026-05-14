"""Integration tests for the ``before-notebook.d`` startup hook.

The hook script is what makes the shim available to shell invocations of
``codex`` and any other consumer outside the kernel. These tests exercise
it as a real subprocess against a stub upstream so we don't have to do
manual acceptance testing every time the script changes.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HOOK = (
    Path(__file__).resolve().parents[1]
    / ".devcontainer"
    / "codex"
    / "before-notebook.d"
    / "10-lmstudio-shim.sh"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _hook_env(**overrides: str) -> dict[str, str]:
    """Env for invoking the hook with the in-repo agent_client_kernel on PYTHONPATH."""
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    env.update(overrides)
    return env


def _run_hook(env: dict[str, str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(HOOK)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _kill_listeners(port: int) -> None:
    """Best-effort: kill any child python -m agent_client_kernel.lmstudio_shim on this port."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"agent_client_kernel.lmstudio_shim"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for pid in out.stdout.split():
            try:
                os.kill(int(pid), 15)
            except ProcessLookupError:
                pass
        time.sleep(0.1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


@pytest.fixture
def stub_upstream():
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            captured["body"] = self.rfile.read(length) if length else b""
            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(resp)

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


def test_hook_file_exists_and_is_executable() -> None:
    assert HOOK.is_file(), f"missing hook at {HOOK}"
    assert os.access(HOOK, os.X_OK), f"hook not executable: {HOOK}"
    contents = HOOK.read_text()
    assert contents.startswith("#!/usr/bin/env bash"), "hook missing shebang"


def test_hook_is_safe_to_source() -> None:
    """Regression: start.sh's run-hooks.sh *sources* .sh files.

    The hook must therefore not call ``exit`` or leave ``set -e``/``set -u``
    active in the parent shell — otherwise start.sh aborts before exec'ing
    Jupyter Lab. Simulate the sourcing path and assert the parent shell
    survives, runs subsequent commands, and has no lingering strict-mode
    options.
    """
    shim_port = _free_port()
    env = _hook_env(
        ACP_LMSTUDIO_SHIM="off",  # shortest path through the hook
        ACP_LMSTUDIO_SHIM_PORT=str(shim_port),
    )
    # Probe `$-` *after* sourcing to confirm no `e`/`u` flags leaked into
    # the parent shell, and emit a sentinel to prove control returned.
    script = f"source {HOOK}\necho POST_SOURCE_FLAGS=$-\necho SENTINEL_OK\n"
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "SENTINEL_OK" in result.stdout, (
        f"sourcing the hook killed the parent shell; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    flags_line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("POST_SOURCE_FLAGS=")),
        "",
    )
    flags = flags_line.split("=", 1)[1] if "=" in flags_line else ""
    assert "e" not in flags, f"hook left `set -e` active in parent: $-={flags!r}"
    assert "u" not in flags, f"hook left `set -u` active in parent: $-={flags!r}"


def test_hook_respects_shim_off() -> None:
    """ACP_LMSTUDIO_SHIM=off must skip startup entirely (no child process)."""
    shim_port = _free_port()
    env = _hook_env(
        ACP_LMSTUDIO_SHIM="off",
        ACP_LMSTUDIO_SHIM_PORT=str(shim_port),
    )
    result = _run_hook(env)
    assert result.returncode == 0, result.stderr
    assert "skipping" in result.stderr.lower()
    assert not _wait_port(shim_port, timeout=0.5), (
        "hook started shim despite ACP_LMSTUDIO_SHIM=off"
    )


def test_hook_starts_shim_and_proxies_to_upstream(stub_upstream) -> None:
    upstream_port, captured = stub_upstream
    shim_port = _free_port()
    env = _hook_env(
        ACP_LMSTUDIO_SHIM="auto",
        ACP_LMSTUDIO_SHIM_PORT=str(shim_port),
        ACP_LMSTUDIO_SHIM_TARGET=f"http://127.0.0.1:{upstream_port}",
        ACP_LMSTUDIO_SHIM_LOG=f"/tmp/lmstudio-shim-test-{shim_port}.log",
    )
    try:
        result = _run_hook(env)
        assert result.returncode == 0, result.stderr
        assert _wait_port(shim_port, timeout=10.0), (
            f"shim did not come up on port {shim_port}; hook stderr={result.stderr!r}"
        )

        # Send a namespace-tool payload through the shim and verify the
        # upstream sees it flattened — i.e., the hook started a working
        # shim, not just any HTTP listener.
        import urllib.request

        payload = json.dumps(
            {
                "tools": [
                    {"type": "function", "name": "shell"},
                    {
                        "type": "namespace",
                        "name": "mcp__jupyter__",
                        "tools": [{"type": "function", "name": "list_notebooks"}],
                    },
                ]
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{shim_port}/v1/responses",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200

        forwarded = json.loads(captured["body"])
        names = [t["name"] for t in forwarded["tools"]]
        assert names == ["shell", "mcp__jupyter__list_notebooks"]
    finally:
        _kill_listeners(shim_port)


def test_hook_is_idempotent_when_port_already_used(stub_upstream) -> None:
    """If the shim port is already bound, the hook must exit cleanly."""
    upstream_port, _ = stub_upstream
    shim_port = _free_port()
    # Hold the port from this process. The hook should detect it and bail
    # without spawning a duplicate.
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", shim_port))
    holder.listen(1)
    try:
        env = _hook_env(
            ACP_LMSTUDIO_SHIM="auto",
            ACP_LMSTUDIO_SHIM_PORT=str(shim_port),
            ACP_LMSTUDIO_SHIM_TARGET=f"http://127.0.0.1:{upstream_port}",
        )
        result = _run_hook(env)
        assert result.returncode == 0, result.stderr
        assert "already in use" in result.stderr.lower()
    finally:
        holder.close()
