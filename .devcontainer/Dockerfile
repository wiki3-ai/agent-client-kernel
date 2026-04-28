FROM quay.io/jupyter/minimal-notebook:latest

# Install a modern Node.js (codex requires node >= 16; the base image's conda
# environment otherwise resolves to an ancient nodejs 11 that cannot parse
# codex's ESM syntax). Pin a major version to avoid surprise regressions.
RUN conda install -y -c conda-forge "nodejs=22" graphviz && \
    conda clean -afy

# Copy project files with correct ownership. We do this before the npm install
# so that we can install npm packages reproducibly from package-lock.json
# rather than pulling whatever `@latest` happens to publish that day.
COPY --chown=jovyan:users . .

# Install npm packages globally at the exact versions pinned in package.json.
# Reading versions from package.json keeps a single source of truth and
# avoids the previous `@latest` foot-gun where a breaking upstream release
# would silently break new container builds.
RUN CODEX_VERSION=$(node -p "require('/home/jovyan/package.json').dependencies['@openai/codex'].replace(/^[^0-9]*/, '')") && \
    CODEX_ACP_VERSION=$(node -p "require('/home/jovyan/package.json').dependencies['@zed-industries/codex-acp'].replace(/^[^0-9]*/, '')") && \
    echo "Installing @openai/codex@${CODEX_VERSION} and @zed-industries/codex-acp@${CODEX_ACP_VERSION}" && \
    npm install -g \
        "@openai/codex@${CODEX_VERSION}" \
        "@zed-industries/codex-acp@${CODEX_ACP_VERSION}"

# Install uv and project dependencies
# Using conda run to ensure we're in the base conda environment
# Then using uv for fast Python package installation
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN pip install --upgrade "uv>=0.9.15"

# Use uv to install dependencies from the locked uv.lock for reproducibility.
# Export the lock to a requirements file, install deps, then install the
# project itself in editable mode without re-resolving deps.
RUN uv export --no-hashes --frozen --format requirements-txt -o /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.lock.txt && \
    uv pip install --system --no-cache --no-deps -e . && \
    rm -f /tmp/requirements.lock.txt

# Install the kernel for the current user
RUN python -m agent_client_kernel install --user
