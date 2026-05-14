# `jupyter-mcp-server` breaks JupyterLab cell execution

## Symptom

After installing this project with its dev dependencies (`pip install -e
'.[dev]'` or `uv sync --extra dev`), JupyterLab silently stops running
cells. Both the `agentclient` kernel **and** the stock `python3` kernel
appear hung: the cell shows the "running" indicator (or nothing at all)
and never produces output. Restarting the kernel doesn't help; the
kernels themselves are healthy and respond to direct `jupyter_client`
traffic — only the path from the Lab UI through the Jupyter Server is
broken.

## Root cause

`jupyter-mcp-server` (Datalayer) depends on `jupyter-server-nbmodel`.
When pip installs that package it auto-registers
`jupyter_server_nbmodel` as a **Jupyter Server extension** that hooks
the notebook-execution endpoint and reroutes cell submissions through
its own server-side execution pipeline (an RTC/YDoc model). The version
that gets pulled in today is a pre-release (`0.1.1a4`) and is not
compatible with the rest of the stack we ship — the result is that
every cell `execute_request` is intercepted by the extension and never
forwarded to the kernel, so cells appear to hang forever.

Confirm with:

```bash
jupyter server extension list
```

If you see `jupyter_server_nbmodel enabled`, the extension is in the
load path even if you didn't ask for it.

We never actually need `jupyter-mcp-server` installed inside the kernel
environment. Codex talks to it as a **standalone** stdio MCP server,
launched on demand via `uvx jupyter-mcp-server@latest` (see
[AGENTS.md](../AGENTS.md)). That launches the server in its own
isolated `uv`-managed virtualenv, so installing it again into the
kernel env buys nothing and has the side effect described above.

## Fix

The project no longer declares `jupyter-mcp-server` or
`jupyter-mcp-tools` as runtime dependencies. The Dockerfile no longer
pre-installs `jupyter-mcp-tools` either. Codex continues to launch the
MCP server via `uvx`, exactly as documented in `AGENTS.md`.

### But the MCP server still needs `jupyter-collaboration`

`jupyter-mcp-server` mutates notebooks by talking to the Jupyter Server
over the YDoc collaboration endpoint
(`PUT /api/collaboration/session/<path>`). That endpoint is **not**
part of stock `jupyter-server`; it is registered by the official
`jupyter-collaboration` package (which ships
`jupyter_server_ydoc`). If `jupyter-collaboration` isn't installed,
every cell-mutation MCP tool (`insert_cell`,
`insert_execute_code_cell`, `overwrite_cell_source`, ...) fails with:

```
Error executing tool insert_cell: 404 Client Error: Not Found for url:
  http://localhost:8888/api/collaboration/session/<notebook>.ipynb
```

We observed an LM Studio / Qwen3-coder agent encounter exactly this
failure mode after the original fix above: the model retried the MCP
calls a few times, then fell back to `execute_code` and hand-rolled a
notebook JSON. In its fallback, it serialized cell source as a
nbformat list-of-strings **without the trailing `\n` on each entry**,
producing a `golden.ipynb` whose single code cell ran together on one
line. The shim and the function-call wire format were innocent — the
raw rollout shows every `cell_source` argument was emitted with
correct `\n` escapes; the cells just never got delivered to the
notebook because the MCP server's required endpoint was missing.

`jupyter-collaboration` is therefore listed as a runtime dependency.
It is **not** the same package as the alpha
`jupyter-server-nbmodel`: `jupyter-collaboration` is maintained by the
JupyterLab project and its `jupyter_server_ydoc` extension does
**not** hijack `execute_request`. We verified that with
`jupyter-collaboration 4.x` installed, `jupyter_client` can still
drive a `python3` kernel end-to-end and the collaboration endpoint
returns `201` instead of `404`.

If you have an environment that was created **before** this fix, clean
it up:

```bash
# Disable the server extensions in case they were already enabled.
jupyter server extension disable jupyter_server_nbmodel
jupyter server extension disable jupyter_mcp_server
jupyter server extension disable jupyter_mcp_tools

# Uninstall the packages so a future `pip` / `uv` operation doesn't
# silently re-enable them.
pip uninstall -y \
    jupyter-mcp-server jupyter-mcp-tools \
    jupyter-server-nbmodel jupyter-nbmodel-client \
    jupyter-kernel-client jupyter-server-client

# Restart Jupyter Lab. Cell execution should work for both `python3`
# and `agentclient` kernels again.
```

A fresh container build (or `uv sync` against the refreshed
`uv.lock`) will not pull these packages back in.

## Diagnostic recipe

If cells ever silently hang again, the fastest check is to bypass
Jupyter Server and talk to a kernel directly:

```python
from jupyter_client.manager import KernelManager

km = KernelManager(kernel_name="python3")
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=10)
msg_id = kc.execute("print(1 + 1)")
while True:
    msg = kc.get_iopub_msg(timeout=5)
    print(msg["msg_type"], msg["content"])
    if (
        msg["msg_type"] == "status"
        and msg["content"].get("execution_state") == "idle"
        and msg["parent_header"].get("msg_id") == msg_id
    ):
        break
kc.stop_channels()
km.shutdown_kernel()
```

If that prints `stream` / `2` / `status idle` cleanly, the kernel side
is fine and the breakage is in a Jupyter Server extension. Run
`jupyter server extension list` and disable suspects until normal
execution returns.

## Related files

- [pyproject.toml](../pyproject.toml) — `dependencies`
- [Dockerfile](../Dockerfile) — bootstrap install
- [AGENTS.md](../AGENTS.md) — how Codex launches `jupyter-mcp-server`
