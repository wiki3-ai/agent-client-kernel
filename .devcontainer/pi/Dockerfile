# syntax=docker/dockerfile:1.7
#
# Pi devcontainer for agent-client-kernel.
#
# Installs:
#   * Node.js 22 (required by pi-acp >= 20 and recommended by pi)
#   * pi (the coding agent) from @earendil-works/pi-coding-agent
#   * pi-acp (the ACP adapter) — bridges ACP <-> pi --mode rpc
#   * pi-mcp-adapter — wires MCP servers (incl. jupyter-mcp-server) through to pi
#
# Pi's default config dir (~/.pi/agent) is populated from the project root
# (.pi/agent/{models,settings,mcp}.json) by COPYing the repo into /home/jovyan,
# matching the pattern used by the Codex devcontainer.

FROM quay.io/jupyter/minimal-notebook:latest

USER root

# Install Node.js 22 (NodeSource), graphviz (used by example notebooks), git, curl.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git graphviz gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Install pi, pi-acp, and the MCP adapter globally as jovyan.
# We use NPM_CONFIG_PREFIX=$HOME/.local so npm doesn't try to write to /usr.
ENV NPM_CONFIG_PREFIX=/home/jovyan/.local
ENV PATH=/home/jovyan/.local/bin:${PATH}

RUN mkdir -p "$HOME/.local" \
    && npm install -g \
        @earendil-works/pi-coding-agent \
        pi-acp \
        pi-mcp-adapter \
    && pi --version \
    && test -x "$HOME/.local/bin/pi-acp"

# Copy project files into jovyan's home (matches the Codex devcontainer layout).
# Crucially, this makes .pi/agent/{models,settings,mcp}.json from the repo land
# at ~/.pi/agent/... so pi picks them up as user-global config.
# Note: --chown does not expand ENV from the base image, so use literal user:group.
COPY --chown=jovyan:users . /home/jovyan/
WORKDIR /home/jovyan

# Install uv and the project (with deps from pyproject.toml) into the
# conda base environment.
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# graphviz is used by some example notebooks.
RUN conda install -y -c conda-forge graphviz && conda clean -afy

# `pip install uv` is just bootstrap (the base image has pip but not uv).
# After that we use `uv pip install -e .`, which resolves dependencies
# directly from pyproject.toml.
RUN pip install --upgrade "uv>=0.9.15" \
    && uv pip install --system -e .

# Register pi-mcp-adapter as a pi extension so the adapter loads on startup.
# Best-effort: failure here should not break the image build.
RUN pi install npm:pi-mcp-adapter || true

# Install the kernel for the current user.
RUN python -m agent_client_kernel install --user
