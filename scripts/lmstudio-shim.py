#!/usr/bin/env python3
"""Standalone CLI wrapper for the LM Studio request shim.

The kernel starts the same shim in-process on construction, so running this
script is only needed when using Codex outside of the Jupyter kernel (e.g.
plain ``codex`` from a shell). See ``agent_client_kernel/lmstudio_shim.py``
for implementation details and supported environment variables.

Usage:

    python3 scripts/lmstudio-shim.py

Environment variables:

    ACP_LMSTUDIO_SHIM_TARGET (alias SHIM_TARGET)  upstream LM Studio base URL
                                                  (default http://192.168.64.1:1234)
    ACP_LMSTUDIO_SHIM_PORT   (alias SHIM_PORT)    local port (default 18234)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the in-tree package importable when run from a checkout without `pip
# install -e .`.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from agent_client_kernel.lmstudio_shim import serve_forever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    serve_forever()


if __name__ == "__main__":
    main()
