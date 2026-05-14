# Agent guidance

## Jupyter MCP server

The `jupyter` MCP server (datalayer/jupyter-mcp-server, started by Codex via
`uvx jupyter-mcp-server@latest`) is configured and connected. Use its tools
directly when the user asks anything notebook-related — listing, reading,
writing, executing cells, etc.

### Tool naming — IMPORTANT

When this Codex session is talking to a Responses-API server that doesn't
support namespace tool envelopes (LM Studio, Ollama OpenAI compat layer),
Codex's MCP tools are flattened into single-name function tools by the
local proxy. The flattened name is:

    mcp__<server>__<tool>

with **double underscores** between each segment. So the Jupyter MCP
tools are exposed to the model as:

- `mcp__jupyter__list_notebooks`
- `mcp__jupyter__list_files`
- `mcp__jupyter__list_kernels`
- `mcp__jupyter__use_notebook`
- `mcp__jupyter__unuse_notebook`
- `mcp__jupyter__restart_notebook`
- `mcp__jupyter__read_notebook`
- `mcp__jupyter__read_cell`
- `mcp__jupyter__insert_cell`
- `mcp__jupyter__delete_cell`
- `mcp__jupyter__move_cell`
- `mcp__jupyter__overwrite_cell_source`
- `mcp__jupyter__edit_cell_source`
- `mcp__jupyter__execute_cell`
- `mcp__jupyter__execute_code`
- `mcp__jupyter__insert_execute_code_cell`
- `mcp__jupyter__connect_to_jupyter` (only needed if `JUPYTER_URL` env wasn't injected)

Use those exact names. **Do NOT** invent variants like `jupyter.list_notebooks`,
`mcp_jupyter_list_notebooks` (single underscores), `mcp.jupyter.list_notebooks`,
or `jupyter_list_notebooks` — none of those will resolve and you will see
"unsupported call" errors.

### Pitfalls

- **Do NOT call `list_mcp_resources` to test whether the server works.** That
  enumerates MCP *resources* (a separate concept); jupyter-mcp-server publishes
  *tools*, not resources, so it always returns `[]`. An empty resource list is
  not a failure — just call a tool directly (e.g. `mcp__jupyter__list_notebooks`).
- Prefer the MCP tools over shelling out to `find ... -name "*.ipynb"`,
  `jupyter notebook list`, `curl http://localhost:8888/...`, etc. Those work
  but bypass the live notebook state (unsaved cells, kernel sessions) that the
  MCP server has access to.
- The Jupyter server is at `http://localhost:8888` with an empty token — this
  is already wired into the MCP server via env. No need to discover it.
