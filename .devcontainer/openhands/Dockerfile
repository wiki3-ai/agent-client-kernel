FROM quay.io/jupyter/minimal-notebook:latest

# Install OpenHands CLI using uv
RUN mamba install uv  \
    && uv tool install openhands

ENV PATH="/home/jovyan/.local/bin:${PATH}"

COPY --chown=jovyan:users . /home/jovyan/agent-client-kernel

WORKDIR /home/jovyan/agent-client-kernel

RUN uv venv \
    && uv pip install jupyter-mcp-tools \
    && uv pip install git+https://github.com/jimwhite/metakernel.git@ipykernel7 \
    && uv pip install -e . \
    && uv run python -m agent_client_kernel install --user
