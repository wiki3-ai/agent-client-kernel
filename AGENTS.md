# Repository Guidelines

## Project Structure

```
agent-client-kernel/
├── agent_client_kernel/    # Core kernel implementation
│   ├── __init__.py
│   ├── __main__.py         # Entry point (kernel installation)
│   ├── kernel.py           # Main ACPKernel class
│   └── lmstudio_shim.py    # LM Studio integration shim
├── tests/                  # Test suite (unit + e2e)
│   ├── conftest.py
│   ├── test_*.py          # Unit tests (pytest)
│   └── e2e/               # End-to-end integration tests
├── examples/              # Usage example notebooks
├── docs/                  # Documentation and postmortems
├── scripts/               # Build/installation utilities
│   ├── agent-versions.json
│   └── install-acp-agents.sh
├── .devcontainer/         # VS Code devcontainer configs
└── Dockerfile             # Container build definition
```

## Build & Development

```bash
# Install dependencies (requires Python >= 3.10)
uv sync --locked --extra dev

# Install the kernel for Jupyter
uv run python -m agent_client_kernel install --user

# Run tests
uv run pytest                      # Unit tests only
ACK_GOLDEN_E2E=1 uv run pytest -m live_e2e  # E2E tests
```

## Coding Standards

- **Python**: Follow PEP 8 style; use type hints where practical
- **Tests**: Include tests for all new features; prefer pytest fixtures
- **Notebooks**: Keep cell output clean; use descriptive variable names
- **Commit Messages**: Use imperative mood ("Add feature", "Fix bug")

## Testing Guidelines

- **Unit tests**: Place in `tests/test_*.py`; fast, isolated
- **E2E tests**: Tag with `@pytest.mark.live_e2e`; require external services
- **Coverage**: Aim for >80% on new code; run `pytest --cov=agent_client_kernel`

## Commit & PR Guidelines

- **Describe the change**: Explain "what" and "why", not "how"
- **Link issues**: Reference GitHub issues in PR descriptions
- **Test results**: Include test output or CI status in PR comments
- **Documentation**: Update README.md or docs/ for user-facing changes

## Agent Configuration

Default agent is Codex. Configure others via environment variables:

```bash
export ACP_AGENT_COMMAND=/path/to/agent
export ACP_AGENT_ARGS="--option value"
```

## MCP Integration

The kernel includes `%agent` magic commands for MCP server and session control:

```python
%agent mcp add/remove/list/clear   # Manage MCP servers
%agent session new/restart/info     # Session management
%agent config/env/permissions       # Configuration
```

See `examples/jupyter-mcp.ipynb` for usage examples.

## Requirements

- Python >= 3.10
- ipykernel >= 7.0
- agent-client-protocol (Python SDK)
- JupyterLab or Jupyter Notebook
