FROM quay.io/jupyter/minimal-notebook:python-3.12

# Install nodejs and graphviz via conda (no root required)
RUN conda install -y -c conda-forge nodejs graphviz

# Copy project files with correct ownership
COPY --chown=jovyan:users . .

# Install uv and project dependencies
# Using conda run to ensure we're in the base conda environment
# Then using uv for fast Python package installation
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN pip install --upgrade "uv>=0.9.15"

RUN uv tool install openhands --with fastapi

# Use uv to install dependencies from pyproject.toml
# --system flag tells uv to install into the current environment (conda base)
# --no-cache reduces image size
RUN uv pip install --system --no-cache -e .

# Install the kernel for the current user
RUN python -m agent_client_kernel install --user
