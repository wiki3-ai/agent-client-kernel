"""
Entry point for the Agent Client Protocol kernel
"""

from .kernel import ACPKernel


def main():
    """Entry point for the kernel"""
    ACPKernel.run_as_main()


if __name__ == '__main__':
    main()

