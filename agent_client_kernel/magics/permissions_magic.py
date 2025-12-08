# Copyright (c) Jim White.
# Distributed under the terms of the BSD 3-Clause License.

from metakernel import Magic


class PermissionsMagic(Magic):
    """Magic commands for permission configuration"""

    def line_permissions(self, args=''):
        """
        %permissions [MODE] - configure permission handling mode

        This magic configures how the kernel handles permission requests
        from the agent.

        Modes:
            auto    - automatically approve all permission requests (default)
            manual  - prompt user for each permission request
            deny    - automatically deny all permission requests
            show    - show current permission mode

        Examples:
            %permissions auto
            %permissions manual
            %permissions show

        Note: Currently only 'auto' mode is fully implemented.
        """
        if not args.strip() or args.strip() == 'show':
            mode = getattr(self.kernel, '_permission_mode', 'auto')
            self.kernel.Print(f"Current permission mode: {mode}")
            self.kernel.Print("\nAvailable modes:")
            self.kernel.Print("  auto   - automatically approve all requests (current default)")
            self.kernel.Print("  manual - prompt for each request (not yet implemented)")
            self.kernel.Print("  deny   - automatically deny all requests")
            return

        mode = args.strip().lower()
        valid_modes = ['auto', 'manual', 'deny']

        if mode not in valid_modes:
            self.kernel.Error(f"Invalid mode: {mode}")
            self.kernel.Print(f"Valid modes: {', '.join(valid_modes)}")
            return

        if mode == 'manual':
            self.kernel.Print("Warning: manual mode is not yet fully implemented")
            self.kernel.Print("Falling back to 'auto' mode")
            mode = 'auto'

        self.kernel._permission_mode = mode
        self.kernel.Print(f"Permission mode set to: {mode}")

    def line_permissions_list(self, args=''):
        """
        %permissions_list - list recent permission requests

        This magic displays a history of recent permission requests
        and how they were handled.

        Example:
            %permissions_list
        """
        if not hasattr(self.kernel, '_permission_history'):
            self.kernel.Print("No permission requests recorded")
            return

        if not self.kernel._permission_history:
            self.kernel.Print("No permission requests recorded")
            return

        self.kernel.Print("Recent permission requests:")
        for i, entry in enumerate(self.kernel._permission_history[-10:], 1):
            status = "✓ APPROVED" if entry['approved'] else "✗ DENIED"
            self.kernel.Print(f"  {i}. {status} - {entry['request']}")


def register_magics(kernel):
    kernel.register_magics(PermissionsMagic)
