FROM quay.io/jupyter/minimal-notebook:latest

RUN conda install conda-forge::nodejs conda-forge::graphviz

# Install npm packages
RUN npm install -g @openai/codex@latest @zed-industries/codex-acp@latest

COPY --chown=jovyan:users . .

RUN pip install --upgrade uv jupyter-mcp-tools jupyter-collaboration \
    && pip install -e . \
    && python -m agent_client_kernel install --user
