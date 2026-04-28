# syntax=docker/dockerfile:1.7
#
# Codex devcontainer for agent-client-kernel.
#
# Installs the codex and codex-acp ACP agents directly from their upstream
# GitHub releases as native (Rust) binaries with SHA-256 checksum
# verification. Versions are pinned in agent-versions.json (single source of
# truth) and can be refreshed with scripts/update-agent-versions.py.
#
# We deliberately do NOT use npm to install codex / codex-acp:
#   * The npm packages are thin shims around the same native binaries; using
#     them would just add a Node runtime and an unverified shim layer.
#   * `npm install` does not verify upstream signatures; checksum-pinned
#     downloads of the GitHub release artifacts give us reproducible,
#     auditable installs with zero npm supply-chain surface.
#
# The build is multi-arch (linux/amd64 and linux/arm64) and uses buildx's
# automatic TARGETARCH build-arg.

FROM quay.io/jupyter/minimal-notebook:latest

ARG TARGETARCH

# Everything in this image is built as jovyan. The base image puts
# ~/.local/bin on PATH for jovyan, so installing the ACP agents there is
# equivalent to /usr/local/bin for our purposes and avoids any need for
# root. Keeping the whole build under a single non-root user also prevents
# accidentally creating root-owned files under /home/jovyan (notably
# ~/.cache, which the base image's HOME=/home/jovyan would otherwise cause
# tools like conda/pip/uv to populate as root).
# This is the default user in the base image so no need to set it here.
# USER jovyan

# Copy project files into jovyan's home (matches the original devcontainer layout).
# Note: --chown does not expand ENV from the base image, so use literal user:group.
COPY --chown=jovyan:users . /home/jovyan/
WORKDIR /home/jovyan

# Install pinned ACP agent binaries into ~/.local/bin (jovyan-writable,
# already on PATH in the base image).
RUN mkdir -p "$HOME/.local/bin" && \
    TARGETARCH="${TARGETARCH}" scripts/install-acp-agents.sh scripts/agent-versions.json "$HOME/.local" && \
    "$HOME/.local/bin/codex" --version && \
    ("$HOME/.local/bin/codex-acp" --version 2>/dev/null || "$HOME/.local/bin/codex-acp" --help >/dev/null)

# bubblewrap is needed by Codex for Linux sandboxing but we're not actually using it.
# TODO: Use https://github.com/openai/codex/blob/main/.devcontainer/devcontainer.secure.json
# graphviz is used by some example notebooks.
RUN conda install -y -c conda-forge bubblewrap graphviz && conda clean -afy

# Install uv and the project (with deps from pyproject.toml) into the
# conda base environment.
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Disable uv's on-disk cache: we don't want a cache baked into the image,
# and disabling it avoids creating ~/.cache/uv at all.
# ENV UV_NO_CACHE=1

# `pip install uv` is just bootstrap (the base image has pip but not uv).
# After that we use `uv pip install -e .`, which resolves dependencies
# directly from pyproject.toml. We don't pin to uv.lock here because
# devcontainer rebuilds intentionally pick up upstream updates.
RUN pip install --upgrade "uv>=0.9.15" && \
    uv pip install --system -e .

# Install the kernel for the current user.
RUN python -m agent_client_kernel install --user
