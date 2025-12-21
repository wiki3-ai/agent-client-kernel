"""
Entry point for the Agent Client Protocol kernel
"""

import argparse
import json
import os
import shutil
import sys

from . import DISPLAY_NAME, LANGUAGE


def get_kernel_spec():
    """Return the kernel specification dictionary."""
    return {
        "argv": [sys.executable, "-m", "agent_client_kernel", "-f", "{connection_file}"],
        "display_name": DISPLAY_NAME,
        "language": LANGUAGE,
        "name": "agentclient"
    }


def install_kernel(user=False, prefix=None):
    """Install the kernel spec."""
    from jupyter_client.kernelspec import KernelSpecManager
    
    # Create a temporary directory with kernel spec
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_dir = os.path.join(tmpdir, "agentclient")
        os.makedirs(kernel_dir)
        
        # Write kernel.json
        kernel_json_path = os.path.join(kernel_dir, "kernel.json")
        with open(kernel_json_path, "w") as f:
            json.dump(get_kernel_spec(), f, indent=2)
        
        # Copy logo files if they exist
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        for logo in ["logo-32x32.png", "logo-64x64.png"]:
            logo_src = os.path.join(resources_dir, logo)
            if os.path.exists(logo_src):
                shutil.copy(logo_src, kernel_dir)
        
        # Install the kernel spec
        ksm = KernelSpecManager()
        ksm.install_kernel_spec(kernel_dir, kernel_name="agentclient", user=user, prefix=prefix)
        
        if user:
            print(f"Installed kernel spec 'agentclient' in user directory")
        elif prefix:
            print(f"Installed kernel spec 'agentclient' in {prefix}")
        else:
            print(f"Installed kernel spec 'agentclient' in system directory")


def main():
    """Entry point for the kernel."""
    # Check if we're being called with install command
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        parser = argparse.ArgumentParser(description="Install the Agent Client Protocol kernel")
        parser.add_argument("install", help="Install the kernel spec")
        parser.add_argument("--user", action="store_true", help="Install for current user only")
        parser.add_argument("--prefix", type=str, help="Install to a specific prefix")
        args = parser.parse_args()
        
        install_kernel(user=args.user, prefix=args.prefix)
    else:
        # Launch the kernel
        from ipykernel.kernelapp import IPKernelApp
        from .kernel import ACPKernel
        IPKernelApp.launch_instance(kernel_class=ACPKernel)


if __name__ == '__main__':
    main()

