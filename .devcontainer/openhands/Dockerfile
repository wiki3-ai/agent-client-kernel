FROM quay.io/jupyter/minimal-notebook:python-3.12

# Install nodejs and graphviz via conda (no root required)
RUN conda install -y -c conda-forge nodejs graphviz

# Copy project files with correct ownership
COPY --chown=jovyan:users . .

RUN pip install --user openhands fastapi

# --system flag tells uv to install into the current environment (conda base)
# --no-cache reduces image size
RUN pip install --user -e .

# Install the kernel for the current user
RUN python -m agent_client_kernel install --user
