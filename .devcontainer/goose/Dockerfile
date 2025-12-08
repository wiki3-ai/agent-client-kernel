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

# Set PATH first so GOOSE_BIN_DIR is recognized by installer
ENV PATH="/home/jovyan/.local/bin:${PATH}"

# Install Goose using the official install script
RUN mkdir -p /home/jovyan/.local/bin && \
    curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | \
    GOOSE_BIN_DIR=/home/jovyan/.local/bin CONFIGURE=false bash

COPY --chown=jovyan:users . /home/jovyan/agent-client-kernel

WORKDIR /home/jovyan

RUN cd /home/jovyan/agent-client-kernel \
    && pip install --upgrade pip \
    && pip install --upgrade uv jupyter-mcp-tools \
    && git submodule update --init --recursive \
    && pip install -e . \
    && python -m agentclientkernel install --user
