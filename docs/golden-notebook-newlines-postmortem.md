# Postmortem: `golden.ipynb` cell with all newlines collapsed

## Symptom

A user ran the agent prompt

> create an example called golden that draws the rectangle/square
> fractal thing with a control for how many levels

through the `agentclient` kernel (Codex / `codex-acp` driving Qwen3-coder
via LM Studio). The session eventually produced a `golden.ipynb` whose
only code cell contained the entire program on a single line — every
`\n` between `import`s, function definitions, and statements was gone,
so the cell could not be executed. Eventually the agent gave up on the
notebook MCP tools entirely and wrote a `.py` file instead.

The user's question: was this a problem with how the agent called the
MCP tools, or with the
[`lmstudio_shim`](../agent_client_kernel/lmstudio_shim.py) that
translates Codex's namespace tool envelopes into the flat function
tools LM Studio's Responses API expects?

## Short answer

**Neither.** The shim and the function-call wire format are innocent.
The cell-mutation MCP tools never actually ran. The agent fell back to
writing the notebook JSON by hand and made a separate, model-side
encoding mistake while doing so.

## Investigation

The Codex session rollout was read directly from
`~/.codex/sessions/.../rollout-*.jsonl`. Every `function_call` whose
name was `insert_cell` or `insert_execute_code_cell` had a payload like

```json
{
  "cell_index": 0,
  "cell_source": "import matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport numpy as np\nfrom ipywidgets import interact, IntSlider\n\ndef draw_golden_spiral_fractal(...):\n    ..."
}
```

— `\n` escapes intact, in the verbatim bytes that Codex received from
the upstream model (via the shim). The
[`SSERewriter`](../agent_client_kernel/lmstudio_shim.py) is
double-tested for this: events that don't need namespace splitting pass
through unchanged, and events that do are re-encoded with
`json.dumps`, which preserves `\n` inside string values. So the
arguments reached the MCP server intact.

The corresponding `function_call_output` records told a very different
story:

```
Error executing tool insert_cell:
  404 Client Error: Not Found for url:
  http://localhost:8888/api/collaboration/session/golden.ipynb
```

…for every cell-write attempt. `jupyter-mcp-server` performs all
notebook mutations through the YDoc collaboration endpoint
`PUT /api/collaboration/session/<path>`, which is **not** part of
stock `jupyter-server`. It is registered by the official
`jupyter-collaboration` package (`jupyter_server_ydoc` extension). That
package was missing from the environment because in the previous round
of debugging we had removed `jupyter-mcp-server` from our runtime
dependencies (it transitively pulled in the broken Datalayer alpha
`jupyter-server-nbmodel` — see
[jupyter-mcp-server-breaks-cell-execution.md](jupyter-mcp-server-breaks-cell-execution.md)).
With `jupyter-mcp-server` gone, `jupyter-collaboration` went with it,
but the standalone `uvx jupyter-mcp-server` Codex spawns still needs
the YDoc endpoint to be reachable on the Jupyter server it talks to.

After several 404s and "cell index 0 out of range" errors, the model
abandoned MCP and used the generic `execute_code` tool to run inline
Python that wrote the notebook JSON itself. That `execute_code` payload
looked like:

```python
notebook = {
    "cells": [
        {"cell_type": "code", ...,
         "source": ["%matplotlib inline",
                    "import matplotlib.pyplot as plt",
                    "import matplotlib.patches as patches",
                    ...]},
        ...
    ],
    ...
}
with open("golden.ipynb", "w") as f:
    json.dump(notebook, f)
```

That is *almost* the right shape — nbformat does accept `source` as a
list of strings — but each element in the list is required to carry
its own trailing `\n`. The model omitted those, so when the notebook
is later read back, `"".join(source)` yields
`"%matplotlib inlineimport matplotlib.pyplot as pltimport matplotlib..."`
all on a single line. That is the cell the user saw.

## Conclusion

| Layer                         | Did it corrupt newlines? |
| ----------------------------- | ------------------------ |
| Qwen3-coder function-call args | No — `\n` present in raw bytes |
| LM Studio Responses transport  | No — bytes preserved end-to-end |
| `lmstudio_shim` SSE rewriter   | No — pass-through verified |
| `jupyter-mcp-server` MCP tools | Never executed (404 on every call) |
| Model fallback `execute_code`  | **Yes** — used the list-of-lines form of nbformat `source` but forgot the trailing `\n` on each element |

The fix is therefore not in the shim. The fix is to make sure the
canonical MCP tool path works so the model never has to roll its own
nbformat:

1. **Add `jupyter-collaboration` as a runtime dependency** so the
   YDoc endpoint exists on the Jupyter Server the MCP process talks
   to. It is *not* the alpha `jupyter-server-nbmodel` extension we
   removed previously; it is the JupyterLab project's stable
   collaboration package, and it does **not** hijack
   `execute_request`. Verified locally that with it installed,
   `PUT /api/collaboration/session/<path>` returns `201` and
   `jupyter_client` still drives a `python3` kernel end-to-end.
2. **Regression coverage** lives in
   `tests/e2e/test_golden_notebook_e2e.py`. It is gated behind
   `ACK_GOLDEN_E2E=1` because it requires a live LM Studio, the
   `lmstudio_shim`, and a Jupyter Lab with `jupyter-collaboration`
   installed; when those are available it runs the full prompt end
   to end in a temporary `CODEX_HOME` + workdir and asserts that
   the resulting `golden.ipynb` contains multi-line code cells and
   the expected symbols (`matplotlib`, `interact`).

## What would have caught this earlier?

- The MCP `function_call_output` text was logged but not surfaced
  prominently. A "tool failed N times in a row" surface in the
  agent-kernel UI would have made the 404 cascade obvious well
  before the model entered its `execute_code` fallback. (Out of
  scope for this fix.)
- The agent-kernel doesn't currently inspect MCP tool dependencies
  at startup. A health check that probes
  `/api/collaboration/session/_probe.ipynb` and warns if the
  endpoint is missing would catch this class of misconfiguration
  immediately. (Worth doing; tracked separately.)
