"""Regression test: pi-acp + pi-mcp-adapter must not spawn host browser tabs.

The Pi devcontainer auto-loads ``pi-mcp-adapter`` as a pi extension. The
adapter's MCP-UI codepath (and a couple of OAuth helpers) ultimately invoke
``$BROWSER`` (falling back to ``xdg-open``) to pop a window/tab on the host.
Inside a Codespaces / VS Code remote container, ``$BROWSER`` is wired to a
helper that opens a *real* tab in the user's browser, so any tool-call loop
that re-triggers an MCP-UI session (or any of the auth helpers) floods the
host with new browser tabs and can OOM the workstation.

This test guards against that by:

  1. Pointing ``$BROWSER`` at a "spy" script that records every URL it is
     asked to open (instead of doing anything).
  2. Driving a small prompt through the real kernel ``ACPClientImpl`` over
     a real ``pi-acp`` subprocess (which in turn spawns ``pi --mode rpc``
     loaded with ``pi-mcp-adapter``).
  3. Asserting the spy log is empty after the round trip.

If pi-mcp-adapter (or anything else in the agent chain) tries to open a
browser tab during a normal request, this test fails.

The test is skipped when ``pi``/``pi-acp`` are not on PATH so it stays safe
to commit. It is exercised by the dedicated ``pi-acp-e2e`` CI job that
installs them.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from unittest.mock import MagicMock

import pytest

from acp import PROTOCOL_VERSION, connect_to_agent, text_block

from agent_client_kernel.kernel import ACPClientImpl, SessionState


PI = shutil.which("pi")
PI_ACP = shutil.which("pi-acp")

pytestmark = pytest.mark.skipif(
    PI is None or PI_ACP is None,
    reason="pi and pi-acp must be installed and on PATH for the e2e test",
)


def _make_mock_kernel() -> MagicMock:
    kernel = MagicMock()
    kernel.state = SessionState()
    kernel.iopub_socket = MagicMock()
    kernel.send_response = MagicMock()
    return kernel


def _write_browser_spy(tmp_path) -> tuple[str, str]:
    """Create a spy script that logs every browser-open invocation.

    Returns (script_path, log_path). The script writes one line per call,
    containing argv joined by tabs, then exits 0 so the caller can't tell
    the difference from a successful ``xdg-open``.
    """
    log_path = tmp_path / "browser_calls.log"
    script_path = tmp_path / "browser_spy.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log_path}"\n'
        "exit 0\n"
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script_path), str(log_path)


@pytest.mark.asyncio
async def test_pi_acp_does_not_open_host_browser(tmp_path):
    """End-to-end: a normal prompt must not invoke ``$BROWSER``."""
    assert PI_ACP is not None  # pytestmark guarantees it.

    spy_path, log_path = _write_browser_spy(tmp_path)
    # Also shadow the common Linux fallbacks so even direct ``xdg-open``
    # calls land in the spy. We prepend a bin dir containing symlinks.
    bin_shim = tmp_path / "bin"
    bin_shim.mkdir()
    for name in ("xdg-open", "open", "sensible-browser", "x-www-browser"):
        (bin_shim / name).symlink_to(spy_path)

    env = os.environ.copy()
    env["BROWSER"] = spy_path
    env["PATH"] = f"{bin_shim}{os.pathsep}{env.get('PATH', '')}"
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
        session = await asyncio.wait_for(
            conn.new_session(mcp_servers=[], cwd=str(tmp_path)),
            timeout=60,
        )
        assert session.session_id

        # A prompt that exercises a normal tool-using path. We don't care
        # about the model's answer — only that the request/response round
        # trip didn't trigger any browser-open syscalls. We also keep this
        # short so the test doesn't hang waiting for an LLM with no key.
        try:
            await asyncio.wait_for(
                conn.prompt(
                    session_id=session.session_id,
                    prompt=[text_block("List the files in the current directory and stop.")],
                ),
                timeout=120,
            )
        except (asyncio.TimeoutError, Exception):
            # Even if the prompt itself fails (no model configured, etc.),
            # the session/init/extension-load path must not have opened a
            # browser. So we still proceed to the assertion.
            pass
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

    # The spy must be empty. If anything (pi, pi-acp, pi-mcp-adapter, or an
    # MCP server) tried to open a browser tab, the file will have at least
    # one entry — that's the bug we're guarding against.
    if os.path.exists(log_path):
        with open(log_path) as f:
            calls = [line for line in f.read().splitlines() if line.strip()]
    else:
        calls = []

    assert calls == [], (
        "pi-acp / pi-mcp-adapter attempted to open browser tabs on the host "
        "during a normal prompt round-trip. Each line below is one URL the "
        "agent tried to open via $BROWSER / xdg-open:\n  "
        + "\n  ".join(calls)
    )
