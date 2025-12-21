"""Agent Client Kernel - A Jupyter kernel for ACP agents."""

__version__ = "0.3.0"

KERNEL_NAME = "agent_client_kernel"
DISPLAY_NAME = "Agent Client Protocol"
LANGUAGE = "agent"

from .kernel import ACPKernel

__all__ = ["ACPKernel", "__version__", "KERNEL_NAME", "DISPLAY_NAME", "LANGUAGE"]
