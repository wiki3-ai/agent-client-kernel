FROM quay.io/jupyter/minimal-notebook:latest

USER root
RUN apt-get update && sudo apt-get install -y \
    nodejs npm \
    graphviz

# Install npm packages
RUN npm install -g @openai/codex@latest @zed-industries/codex-acp@latest

USER jovyan

COPY --chown=jovyan:users . .

RUN pip install --upgrade pip \
    && pip install --upgrade jupyter-mcp-tools \
    && git submodule update --init --recursive \
    && pip install -e metakernel/ \
    && pip install -e . \
    && python -m agent_client_kernel install --user
