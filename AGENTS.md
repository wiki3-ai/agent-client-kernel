# Repository Guidelines

## Project Structure & Module Organization
- `agent_client_kernel/` is the core Python package: `__main__.py` boots the kernel, `kernel.py` implements the client protocol, and `resources/` keeps bundled metadata (icons, MCP definitions, etc.).
- `examples/` houses notebooks (`basic_usage.ipynb`, `configuration_demo.ipynb`, `jupyter-mcp.ipynb`) that demonstrate `%agent` flows and MCP service wiring.
- `tests/` mirrors the kernel surface with `test_kernel.py`, `test_magic_commands.py`, and helper fixtures in `conftest.py`; add new suites here alongside any integration helpers.
- Project metadata and tooling live at `pyproject.toml` (hatchling + pytest config), `package.json`/`package-lock.json` (agent adapters), and `uv.lock` (UV version pinning). Keep README and DEVCONTAINER notes up to date when altering features.

## Build, Test, and Development Commands
- `pip install -e .` builds the package in editable mode so local edits run in notebooks immediately; rerun after new modules or dependencies.
- `python -m agent_client_kernel install --user` registers the kernel with the active user Jupyter environment.
- `pytest` (or `python -m pytest`) runs the full test suite defined under `tests/`; rely on the `[project.optional-dependencies]` dev extras when installing locally (`pip install -e .[dev]`) before executing.
- Use the Dev Container workflow (`.devcontainer/…`) to boot a ready environment: select one of the agent presets (Codex, Gemini, Goose, Kimi, cagent) to get JupyterLab, the kernel, and an ACP-compatible agent prewired.

## Coding Style & Naming Conventions
- Python files use 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and descriptive docstrings when behavior is not obvious.
- Keep logic in `kernel.py` focused on ACP session handling; helpers in submodules should stay small and testable.
- Follow PEP 8, prefer type annotations where they clarify ACP plumbing, and avoid reformatting unrelated files. There is no enforced formatter—keep diffs tight.
- Name tests `test_<feature>.py` and keep fixtures in `tests/conftest.py`; root-level helper modules should be lowercase.

## Testing Guidelines
- Pytest drives all checks (`testpaths = ["tests"]` in `pyproject.toml`). New kernel behaviors need at least one unit/integration test under `tests/`.
- Prefix new files with `test_`, keep tests focused (one kernel feature per test), and favor `pytest` fixtures for shared agent setup.
- Use `pytest --maxfail=1` locally when hunting regressions; CI should run the same suite without extra args.

## Commit & Pull Request Guidelines
- Commit messages should remain short, descriptive, and tense-neutral (e.g., “Add ACP permissions helper” or “Fix kernel restart handling”). Keep the scope narrow to make history easy to scan.
- Pull requests need a clear description, linked issue or rationale, and a note on how testers can reproduce the change. Include screenshots only when user-visible UI changes exist, and attach logs/output snippets when debugging prompts or kernel failures.

## Agent Configuration & Documentation Notes
- Document new environment variables or MCP servers in `README.md` and, if relevant, add a corresponding example notebook under `examples/`.
- When adding support for a new ACP adapter, update `package.json`/`package-lock.json` (if it requires npm deps) and clarify any additional setup steps (install commands, auth flows) in README or AGENTS.
