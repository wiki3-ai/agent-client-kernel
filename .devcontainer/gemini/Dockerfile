FROM quay.io/jupyter/minimal-notebook:latest

RUN conda install -y -c conda-forge \
    nodejs \
    graphviz \
    make \
    gxx \
    curl \
    git \
    ca-certificates

# Install Gemini CLI from npm
RUN npm install -g @google/gemini-cli@latest

COPY --chown=jovyan:users . .

RUN pip install --user -e . \
    && python -m agent_client_kernel install --user
