"""
Entry point for the Agent Client Protocol kernel
"""

from ipykernel.kernelapp import IPKernelApp
from .kernel import ACPKernel


def main():
    """Entry point for the kernel"""
    IPKernelApp.launch_instance(kernel_class=ACPKernel)


if __name__ == '__main__':
    main()

