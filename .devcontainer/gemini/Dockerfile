FROM quay.io/jupyter/minimal-notebook:latest

USER root
RUN apt-get update && sudo apt-get install -y \
    nodejs npm \
    graphviz \
    python3 \
    make \
    g++ \
    curl \
    git \
    ca-certificates

# Install Gemini CLI from npm
RUN npm install -g @google/gemini-cli --unsafe-perm

USER jovyan

COPY --chown=jovyan:users . /home/jovyan/agent-client-kernel

WORKDIR /home/jovyan

RUN cd /home/jovyan/agent-client-kernel \
    && pip install --upgrade pip \
    && pip install --upgrade uv jupyter-mcp-tools \
    && git submodule update --init --recursive \
    && pip install -e . \
    && python -m agentclientkernel install --user
