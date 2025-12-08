FROM quay.io/jupyter/minimal-notebook:latest

USER root

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Install cagent using pre-built binaries from GitHub releases
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then CAGENT_ARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then CAGENT_ARCH="arm64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    mkdir -p /home/jovyan/.local/bin && \
    curl -fsSL "https://github.com/docker/cagent/releases/latest/download/cagent-linux-${CAGENT_ARCH}" \
         -o /home/jovyan/.local/bin/cagent && \
    chmod +x /home/jovyan/.local/bin/cagent

ENV PATH="/home/jovyan/.local/bin:${PATH}"

COPY --chown=jovyan:users . /home/jovyan/agent-client-kernel

WORKDIR /home/jovyan

RUN cd /home/jovyan/agent-client-kernel \
    && pip install --upgrade pip \
    && pip install --upgrade uv jupyter-mcp-tools \
    && git submodule update --init --recursive \
    && pip install -e . \
    && python -m agent_client_kernel install --user
