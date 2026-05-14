"""End-to-end: ``codex exec`` builds a runnable ``golden.ipynb`` via MCP.

This is the regression test for the "all newlines stripped from one
code cell" failure documented in
``docs/golden-notebook-newlines-postmortem.md``. The model is asked to
build the golden-ratio fractal example notebook using the standalone
``uvx jupyter-mcp-server`` that Codex launches as an MCP stdio server.
We assert that the resulting notebook contains multi-line code cells —
which means the MCP cell-insert path actually worked and the model did
not have to fall back to hand-rolling nbformat.

Opt-in: requires a live LM Studio, the in-container lmstudio shim, a
running Jupyter Lab with the ``jupyter-collaboration`` extension, and a
``~/.codex`` already configured for the ``local_lmstudio`` provider
(the devcontainer default). Run with::

    ACK_GOLDEN_E2E=1 pytest tests/e2e/test_golden_notebook_e2e.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("ACK_GOLDEN_E2E") != "1",
    reason="set ACK_GOLDEN_E2E=1 to run the golden-notebook e2e test",
)


PROMPT = (
    "Create an example notebook at golden.ipynb that draws the golden "
    "ratio rectangle/square fractal with an interactive slider for the "
    "number of recursion levels. Use matplotlib for drawing and "
    "ipywidgets.interact with an IntSlider for the control. Use the "
    "jupyter MCP tools to create the notebook and insert the cells; do "
    "not write the notebook JSON by hand. When the notebook is saved "
    "and the cells look right, stop."
)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _collab_endpoint_alive(url: str = "http://127.0.0.1:8888") -> bool:
    """Return True iff ``jupyter-collaboration`` is loaded.

    The collaboration endpoint returns 405 for GET (it only accepts
    PUT) when present, and 404 when the extension is not installed.
    """
    req = urllib.request.Request(f"{url}/api/collaboration/session/_probe.ipynb")
    try:
        urllib.request.urlopen(req, timeout=2)
        return True
    except urllib.error.HTTPError as exc:
        return exc.code != 404
    except Exception:
        return False


@pytest.fixture(scope="module")
def _required_services() -> None:
    if not _port_open("127.0.0.1", 18234):
        pytest.skip("lmstudio_shim not running on 127.0.0.1:18234")
    if not _port_open("127.0.0.1", 8888):
        pytest.skip("Jupyter Lab not running on 127.0.0.1:8888")
    if not _collab_endpoint_alive():
        pytest.skip(
            "jupyter-collaboration endpoint missing on 8888; install "
            "`jupyter-collaboration` and restart Jupyter Lab"
        )
    # LM Studio is on the docker host. ``host.docker.internal`` is
    # routed through the shim, so probing the shim port above is
    # sufficient — but warn if the upstream is obviously offline.


def _seed_codex_home(src: Path, dst: Path) -> None:
    """Copy just the user-config bits Codex needs; skip session state."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.toml", "user-config.toml", "version.json"):
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)
    for name in ("model-catalogs",):
        s = src / name
        if s.is_dir():
            shutil.copytree(s, dst / name, dirs_exist_ok=True)


def _cell_source_text(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def test_golden_notebook_built_via_mcp(tmp_path: Path, _required_services) -> None:
    src_codex = Path(os.path.expanduser("~/.codex"))
    if not (src_codex / "config.toml").exists():
        pytest.skip("no ~/.codex/config.toml to seed from")

    codex_home = tmp_path / "codex_home"
    _seed_codex_home(src_codex, codex_home)

    workdir = tmp_path / "work"
    workdir.mkdir()

    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        # ``codex exec`` writes nothing to HOME beyond CODEX_HOME, but
        # some sub-tools (uvx caches) do. Keep HOME stable so uvx can
        # reuse its cached jupyter-mcp-server install instead of
        # rebuilding for every test run.
    }

    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(workdir),
        "--json",
        PROMPT,
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )

    # Always dump tails so a failure is debuggable.
    sys.stderr.write("---- codex stdout tail ----\n")
    sys.stderr.write(proc.stdout[-4000:])
    sys.stderr.write("\n---- codex stderr tail ----\n")
    sys.stderr.write(proc.stderr[-2000:])
    sys.stderr.write("\n---------------------------\n")

    assert proc.returncode == 0, f"codex exec failed: rc={proc.returncode}"

    nb_path = workdir / "golden.ipynb"
    assert nb_path.exists(), "agent did not produce golden.ipynb"

    nb = json.loads(nb_path.read_text())
    assert nb.get("nbformat") == 4, "not a v4 notebook"

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert code_cells, "notebook has no code cells"

    joined = "\n\n".join(_cell_source_text(c) for c in code_cells)

    # The bug we are guarding against: every line collapsed onto one
    # physical line. A real fractal cell has well over a dozen
    # newlines (imports + def + body + plotting).
    newline_count = joined.count("\n")
    assert newline_count >= 8, (
        f"code cells appear collapsed (only {newline_count} newlines); "
        f"first cell source repr: {_cell_source_text(code_cells[0])!r}"
    )

    # And: the cells must actually reference the libraries the prompt
    # called for, otherwise we just got an empty-but-multi-line cell.
    for needle in ("matplotlib", "interact"):
        assert needle in joined, f"expected {needle!r} in code cells"

    # Sanity: no cell-source value is a single giant unbroken line of
    # Python (the original symptom). Allow short single-line cells like
    # ``%matplotlib inline``.
    for i, cell in enumerate(code_cells):
        text = _cell_source_text(cell)
        if len(text) > 120 and "\n" not in text:
            pytest.fail(
                f"code cell {i} is {len(text)} chars on a single line "
                f"(newline-strip regression): {text!r}"
            )
