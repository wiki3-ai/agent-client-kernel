# Copyright (c) Jim White.
# Distributed under the terms of the BSD 3-Clause License.

from metakernel import Magic


class SessionMagic(Magic):
    """Magic commands for session management"""

    def line_new_session(self, args=''):
        """
        %new_session [CWD] - create a new agent session

        This magic creates a new session with the agent. Any configured
        MCP servers will be started with the new session.

        Arguments:
            CWD - optional working directory for the session (defaults to current directory)

        Examples:
            %new_session
            %new_session /path/to/project

        Note: Creating a new session will terminate any existing session.
        """
        import os
        import asyncio

        cwd = args.strip() if args.strip() else os.getcwd()

        # Validate the directory
        if not os.path.isdir(cwd):
            self.kernel.Error(f"Directory does not exist: {cwd}")
            return

        self.kernel.Print(f"Creating new session with working directory: {cwd}")

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Stop existing session
        if hasattr(self.kernel, '_session_id') and self.kernel._session_id:
            self.kernel.Print("Terminating existing session...")
            try:
                loop.run_until_complete(self.kernel._stop_agent())
            except Exception as e:
                self.kernel.Error(f"Error stopping existing session: {e}")

        # Start new session with configured MCP servers and working directory
        try:
            self.kernel._session_cwd = cwd
            loop.run_until_complete(self.kernel._start_agent())
            self.kernel.Print(f"New session created: {self.kernel._session_id}")
            
            # List MCP servers if any were configured
            if hasattr(self.kernel, '_mcp_servers') and self.kernel._mcp_servers:
                self.kernel.Print(f"\nMCP servers configured: {len(self.kernel._mcp_servers)}")
                for server in self.kernel._mcp_servers:
                    self.kernel.Print(f"  - {server['name']}")
        except Exception as e:
            self.kernel.Error(f"Error creating session: {e}")
            import traceback
            self.kernel.Error(traceback.format_exc())

    def line_session_info(self, args=''):
        """
        %session_info - display current session information

        This magic displays information about the current agent session,
        including session ID, working directory, and MCP servers.

        Example:
            %session_info
        """
        import os

        if not hasattr(self.kernel, '_session_id') or not self.kernel._session_id:
            self.kernel.Print("No active session")
            self.kernel.Print("\nUse %new_session to create a session")
            return

        self.kernel.Print("Current Session Information:")
        self.kernel.Print(f"  Session ID: {self.kernel._session_id}")
        
        cwd = getattr(self.kernel, '_session_cwd', os.getcwd())
        self.kernel.Print(f"  Working Directory: {cwd}")
        
        # Show agent info
        self.kernel.Print(f"  Agent Command: {self.kernel._agent_command}")
        if self.kernel._agent_args:
            self.kernel.Print(f"  Agent Args: {' '.join(self.kernel._agent_args)}")
        
        # Show MCP servers
        if hasattr(self.kernel, '_mcp_servers') and self.kernel._mcp_servers:
            self.kernel.Print(f"\n  MCP Servers ({len(self.kernel._mcp_servers)}):")
            for server in self.kernel._mcp_servers:
                args_str = ' '.join(server['args']) if server['args'] else ''
                self.kernel.Print(f"    - {server['name']}: {server['command']} {args_str}")
        else:
            self.kernel.Print("\n  No MCP servers configured")
        
        # Show permission mode
        mode = getattr(self.kernel, '_permission_mode', 'auto')
        self.kernel.Print(f"\n  Permission Mode: {mode}")

    def line_session_restart(self, args=''):
        """
        %session_restart - restart the current session

        This magic restarts the agent session, keeping the same
        configuration (working directory, MCP servers, etc.).

        Example:
            %session_restart
        """
        import os
        import asyncio

        cwd = getattr(self.kernel, '_session_cwd', os.getcwd())
        
        self.kernel.Print("Restarting session...")

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Stop and restart
        try:
            if hasattr(self.kernel, '_session_id') and self.kernel._session_id:
                loop.run_until_complete(self.kernel._stop_agent())
            
            loop.run_until_complete(self.kernel._start_agent())
            self.kernel.Print(f"Session restarted: {self.kernel._session_id}")
        except Exception as e:
            self.kernel.Error(f"Error restarting session: {e}")


def register_magics(kernel):
    kernel.register_magics(SessionMagic)
