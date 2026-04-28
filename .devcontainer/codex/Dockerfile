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

# Install pinned ACP agent binaries (as root, into /usr/local/bin).
USER root
COPY .devcontainer/codex/agent-versions.json    /tmp/agent-versions.json
COPY .devcontainer/codex/install-acp-agents.sh  /tmp/install-acp-agents.sh
RUN chmod +x /tmp/install-acp-agents.sh && \
    TARGETARCH="${TARGETARCH}" /tmp/install-acp-agents.sh /tmp/agent-versions.json /usr/local && \
    rm -f /tmp/agent-versions.json /tmp/install-acp-agents.sh && \
    codex --version && \
    (codex-acp --version 2>/dev/null || codex-acp --help >/dev/null)

# graphviz is used by some example notebooks.
RUN conda install -y -c conda-forge graphviz && conda clean -afy

USER ${NB_UID}

# Copy project files into jovyan's home (matches the original devcontainer layout).
COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/
WORKDIR /home/${NB_USER}

# Install uv and the project's locked Python dependencies into the conda
# base environment for fast, reproducible installs.
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN pip install --upgrade "uv>=0.9.15" && \
    uv export --no-hashes --frozen --format requirements-txt -o /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache --no-deps -e . && \
    rm -f /tmp/requirements.lock.txt

# Install the kernel for the current user.
RUN python -m agent_client_kernel install --user
