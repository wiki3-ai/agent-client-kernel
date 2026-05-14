"""End-to-end test: Jupyter Lab in the devcontainer starts and serves /lab.

This guards against the failure mode where ``jupyter lab`` boots far enough
to log "Jupyter Server is running" but then deadlocks serving the page.
Run opt-in:

    JUPYTER_E2E=1 pytest tests/e2e/test_jupyter_startup.py -v

Skipped by default to avoid slowing the regular test suite.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("JUPYTER_E2E") != "1",
    reason="set JUPYTER_E2E=1 to run the Jupyter Lab startup e2e test",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_url(url: str, timeout: float) -> tuple[int, float]:
    """Poll until ``url`` returns any HTTP status. Returns (status, elapsed)."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, time.monotonic() - (deadline - timeout)
        except urllib.error.HTTPError as err:
            return err.code, time.monotonic() - (deadline - timeout)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as err:
            last_err = err
            time.sleep(0.5)
    raise TimeoutError(f"{url} not reachable within {timeout}s ({last_err!r})")


@pytest.fixture
def jupyter_server(tmp_path: Path):
    jupyter = shutil.which("jupyter")
    if jupyter is None:
        pytest.skip("jupyter not on PATH")

    port = _free_port()
    log_path = tmp_path / "jupyter.log"
    # Use minimal flags so we test the actual server, but disable auth and
    # browser auto-open so the test is hermetic.
    proc = subprocess.Popen(
        [
            jupyter,
            "lab",
            "--no-browser",
            "--ServerApp.ip=127.0.0.1",
            f"--ServerApp.port={port}",
            "--ServerApp.port_retries=0",
            "--IdentityProvider.token=",
            "--ServerApp.password=",
            "--ServerApp.disable_check_xsrf=True",
            f"--ServerApp.root_dir={tmp_path}",
        ],
        stdout=log_path.open("wb"),
        stderr=subprocess.STDOUT,
        cwd=tmp_path,
        env={**os.environ, "JUPYTER_PLATFORM_DIRS": "1"},
    )
    try:
        yield port, log_path, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_jupyter_lab_serves_lab_page(jupyter_server) -> None:
    """The /lab page must respond with HTTP 200 within a reasonable timeout."""
    port, log_path, proc = jupyter_server
    base = f"http://127.0.0.1:{port}"

    # /api/status is the lightest readiness endpoint.
    status, _ = _wait_for_url(f"{base}/api/status", timeout=60)
    assert status == 200, (
        f"jupyter /api/status returned {status}; log tail:\n"
        f"{log_path.read_text()[-2000:]}"
    )

    # /lab is the actual UI entry point — this is what deadlocks on us.
    start = time.monotonic()
    with urllib.request.urlopen(f"{base}/lab", timeout=30) as resp:
        body = resp.read(4096)
        status = resp.status
    elapsed = time.monotonic() - start

    assert status == 200, f"/lab returned {status}"
    assert elapsed < 30, f"/lab took {elapsed:.1f}s (deadlock?)"
    assert b"<html" in body.lower() or b"<!doctype" in body.lower(), (
        f"/lab body did not look like HTML: {body[:200]!r}"
    )

    # The kernel spec should be discoverable too.
    with urllib.request.urlopen(f"{base}/api/kernelspecs", timeout=10) as resp:
        assert resp.status == 200


def test_jupyter_lab_kernel_spec_present(jupyter_server) -> None:
    """The agentclient kernel must be listed if it has been installed."""
    import json

    port, log_path, _ = jupyter_server
    base = f"http://127.0.0.1:{port}"
    _wait_for_url(f"{base}/api/status", timeout=60)

    with urllib.request.urlopen(f"{base}/api/kernelspecs", timeout=10) as resp:
        data = json.loads(resp.read())

    kernelspecs = data.get("kernelspecs", {})
    if "agentclient" not in kernelspecs:
        pytest.skip(
            "agentclient kernelspec not installed; "
            "run `python -m agent_client_kernel install --user` first"
        )
    assert kernelspecs["agentclient"]["spec"]["language"]
