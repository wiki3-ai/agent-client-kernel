FROM quay.io/jupyter/minimal-notebook:latest

RUN conda install -y -c conda-forge nodejs graphviz

# Install npm packages
RUN npm install -g @openai/codex@latest @zed-industries/codex-acp@latest

COPY --chown=jovyan:users . .

RUN conda create -n jupyter \
    && conda init --all

RUN conda activate jupyter \
    && pip install "uv>0.9.15" \
    && uv sync \
    && uv pip install --link-mode=copy -e . \
    && python -m agent_client_kernel install --user 
