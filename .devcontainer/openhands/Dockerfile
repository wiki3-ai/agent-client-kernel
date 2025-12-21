FROM quay.io/jupyter/minimal-notebook:python-3.12

# Copy the repo into the container for installation
COPY --chown=jovyan:users . .

# Create a user venv for installing packages
ENV VIRTUAL_ENV=${HOME}/.venv
RUN python -m venv ${VIRTUAL_ENV}

# Add venv activation to bashrc so all new shells use it
RUN echo 'source ${VIRTUAL_ENV}/bin/activate' >> ${HOME}/.bashrc

# Ensure the venv is used for this build and set PATH permanently
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install packages in the user venv
RUN pip install openhands jupyter-mcp-tools \
  && pip install -e . \
  && python -m agent_client_kernel install --user 
