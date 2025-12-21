# Copyright (c) Jim White.
# Distributed under the terms of the BSD 3-Clause License.

from metakernel import Magic
import json
import asyncio
import os


class MCPMagic(Magic):
    """Magic commands for MCP (Model Context Protocol) server configuration"""

    def _restart_session_if_active(self, action_description):
        """Restart the session if one is active to apply MCP changes"""
        if not hasattr(self.kernel, '_session_id') or not self.kernel._session_id:
            return
        
        self.kernel.Print(f"Restarting session to apply MCP changes...")
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Stop and restart the session
        try:
            loop.run_until_complete(self.kernel._stop_agent())
            loop.run_until_complete(self.kernel._start_agent())
            self.kernel.Print(f"Session restarted: {self.kernel._session_id}")
        except Exception as e:
            self.kernel.Error(f"Error restarting session: {e}")

    def line_mcp_add(self, args=''):
        """
        %mcp_add NAME COMMAND [ARGS...] - add an MCP server configuration

        This magic adds an MCP server to the current session configuration.
        If a session is active, it will be automatically restarted to apply
        the new MCP server.

        Arguments:
            NAME - human-readable name for the MCP server
            COMMAND - path to the MCP server executable
            ARGS - optional command-line arguments (space-separated)

        Examples:
            %mcp_add filesystem /usr/local/bin/mcp-server-filesystem
            %mcp_add github /path/to/mcp-github --token abc123
            %mcp_add jupyter uvx jupyter-mcp-server@latest
        """
        if not args.strip():
            self.kernel.Error("Usage: %mcp_add NAME COMMAND [ARGS...]")
            return

        parts = args.split(None, 2)
        if len(parts) < 2:
            self.kernel.Error("Usage: %mcp_add NAME COMMAND [ARGS...]")
            return

        name = parts[0]
        command = parts[1]
        server_args = parts[2].split() if len(parts) > 2 else []

        # Add to kernel's MCP server list
        if not hasattr(self.kernel, '_mcp_servers'):
            self.kernel._mcp_servers = []

        # Check if server with this name already exists
        for i, server in enumerate(self.kernel._mcp_servers):
            if server['name'] == name:
                self.kernel._mcp_servers[i] = {
                    'name': name,
                    'command': command,
                    'args': server_args,
                    'env': []
                }
                self.kernel.Print(f"Updated MCP server '{name}'")
                self._restart_session_if_active(f"Updated MCP server '{name}'")
                return

        self.kernel._mcp_servers.append({
            'name': name,
            'command': command,
            'args': server_args,
            'env': []
        })
        self.kernel.Print(f"Added MCP server '{name}'")
        self._restart_session_if_active(f"Added MCP server '{name}'")

    def line_mcp_list(self, args=''):
        """
        %mcp_list - list all configured MCP servers

        This magic displays all MCP servers currently configured for
        the session.

        Example:
            %mcp_list
        """
        if not hasattr(self.kernel, '_mcp_servers') or not self.kernel._mcp_servers:
            self.kernel.Print("No MCP servers configured")
            return

        self.kernel.Print("Configured MCP servers:")
        for server in self.kernel._mcp_servers:
            args_str = ' '.join(server['args']) if server['args'] else '(no args)'
            self.kernel.Print(f"  - {server['name']}: {server['command']} {args_str}")

    def line_mcp_remove(self, args=''):
        """
        %mcp_remove NAME - remove an MCP server configuration

        This magic removes an MCP server from the session configuration.
        If a session is active, it will be automatically restarted to apply
        the change.

        Arguments:
            NAME - name of the MCP server to remove

        Example:
            %mcp_remove filesystem
        """
        if not args.strip():
            self.kernel.Error("Usage: %mcp_remove NAME")
            return

        name = args.strip()

        if not hasattr(self.kernel, '_mcp_servers'):
            self.kernel.Error(f"No MCP server named '{name}' found")
            return

        for i, server in enumerate(self.kernel._mcp_servers):
            if server['name'] == name:
                self.kernel._mcp_servers.pop(i)
                self.kernel.Print(f"Removed MCP server '{name}'")
                self._restart_session_if_active(f"Removed MCP server '{name}'")
                return

        self.kernel.Error(f"No MCP server named '{name}' found")

    def line_mcp_clear(self, args=''):
        """
        %mcp_clear - remove all MCP server configurations

        This magic removes all configured MCP servers.
        If a session is active, it will be automatically restarted to apply
        the change.

        Example:
            %mcp_clear
        """
        if hasattr(self.kernel, '_mcp_servers'):
            count = len(self.kernel._mcp_servers)
            self.kernel._mcp_servers = []
            self.kernel.Print(f"Removed {count} MCP server(s)")
            if count > 0:
                self._restart_session_if_active(f"Cleared {count} MCP servers")
        else:
            self.kernel.Print("No MCP servers to clear")


def register_magics(kernel):
    kernel.register_magics(MCPMagic)
