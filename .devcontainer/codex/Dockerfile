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

# graphviz is used by some example notebooks.
RUN conda install -y -c conda-forge graphviz && conda clean -afy

# Install uv and the project's locked Python dependencies into the conda
# base environment for fast, reproducible installs.
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Point uv's cache at a writable tmp location. We pass --no-cache to
# `uv pip install` below, but `uv export` has no equivalent flag and will
# otherwise try to initialize ~/.cache/uv.
ENV UV_CACHE_DIR=/tmp/uv-cache

RUN pip install --upgrade "uv>=0.9.15" && \
    uv export --no-hashes --frozen --format requirements-txt -o /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache --no-deps -e . && \
    rm -rf /tmp/requirements.lock.txt /tmp/uv-cache

# Install the kernel for the current user.
RUN python -m agent_client_kernel install --user
