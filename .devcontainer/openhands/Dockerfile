FROM quay.io/jupyter/minimal-notebook:python-3.12

# Create a user venv for installing packages
ENV VIRTUAL_ENV=/home/jovyan/.venv
RUN python -m venv ${VIRTUAL_ENV}

# Add venv activation to bashrc so all new shells use it
RUN echo 'source ${VIRTUAL_ENV}/bin/activate' >> ~/.bashrc

# Ensure the venv is used for this build and set PATH permanently
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy the repo into the container for installation
COPY --chown=jovyan:users . /tmp/agent-client-kernel

# Install packages in the user venv
RUN pip install --upgrade pip \
  && pip install git+https://github.com/jimwhite/metakernel.git@ipykernel7 \
  && pip install openhands jupyter-mcp-tools \
  && pip install -e /tmp/agent-client-kernel \
  && python -m agent_client_kernel install --user 
