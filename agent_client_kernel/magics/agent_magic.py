# Copyright (c) Jim White.
# Distributed under the terms of the BSD 3-Clause License.

from metakernel import Magic
import os


class AgentMagic(Magic):
    """Unified magic command for all agent configuration and management"""
    
    def get_help_on(self, info, level=0, none_on_fail=False):
        """Provide help for agent subcommands.
        
        This method is called by the MetaKernel help system when users type
        expressions like '%agent mcp?' or use '%help %agent mcp'.
        """
        if not info.get('code'):
            return None if none_on_fail else ''
        
        expr = info.get('obj', info.get('code', '')).strip()
        
        # Delegate to the kernel's helper method for subcommands
        if hasattr(self.kernel, '_get_agent_subcommand_help'):
            result = self.kernel._get_agent_subcommand_help(expr)
            if result:
                return result
        
        # If no specific help found, return None or default message
        if none_on_fail:
            return None
        else:
            return None

    def line_agent(self, args=''):
        """
        %agent SUBCOMMAND [ARGS...] - unified agent management command

        This magic provides a unified interface for all agent configuration,
        session management, MCP servers, and permissions.

        Subcommands:

        MCP Server Configuration:
          %agent mcp add NAME COMMAND [ARGS...]  - add MCP server
          %agent mcp list                        - list MCP servers
          %agent mcp remove NAME                 - remove MCP server
          %agent mcp clear                       - clear all MCP servers

        Permission Configuration:
          %agent permissions [MODE]              - set permission mode (auto/manual/deny)
          %agent permissions list                - show permission history

        Session Management:
          %agent session new [CWD]               - create new session
          %agent session info                    - show session information
          %agent session restart                 - restart current session

        Agent Configuration:
          %agent config [COMMAND [ARGS...]]     - configure agent command
          %agent env [KEY=VALUE]                 - set environment variables

        Examples:
            %agent mcp add filesystem /usr/local/bin/mcp-server-filesystem
            %agent permissions auto
            %agent session new /path/to/project
            %agent config codex-acp --verbose
        """
        if not args.strip():
            self._show_help()
            return

        parts = args.split(None, 1)
        subcommand = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ''

        # Route to appropriate handler
        if subcommand == 'mcp':
            self._handle_mcp(subargs)
        elif subcommand == 'permissions':
            self._handle_permissions(subargs)
        elif subcommand == 'session':
            self._handle_session(subargs)
        elif subcommand == 'config':
            self._handle_config(subargs)
        elif subcommand == 'env':
            self._handle_env(subargs)
        else:
            self.kernel.Error(f"Unknown subcommand: {subcommand}")
            self.kernel.Print("Use '%agent' without arguments to see available subcommands")

    def _show_help(self):
        """Show help for the agent magic"""
        self.kernel.Print("Agent Management Commands")
        self.kernel.Print("")
        self.kernel.Print("MCP Server Configuration:")
        self.kernel.Print("  %agent mcp add NAME COMMAND [ARGS...]")
        self.kernel.Print("  %agent mcp list")
        self.kernel.Print("  %agent mcp remove NAME")
        self.kernel.Print("  %agent mcp clear")
        self.kernel.Print("")
        self.kernel.Print("Permission Configuration:")
        self.kernel.Print("  %agent permissions [auto|manual|deny]")
        self.kernel.Print("  %agent permissions list")
        self.kernel.Print("")
        self.kernel.Print("Session Management:")
        self.kernel.Print("  %agent session new [CWD]")
        self.kernel.Print("  %agent session info")
        self.kernel.Print("  %agent session restart")
        self.kernel.Print("")
        self.kernel.Print("Agent Configuration:")
        self.kernel.Print("  %agent config [COMMAND [ARGS...]]")
        self.kernel.Print("  %agent env [KEY=VALUE]")
        self.kernel.Print("")
        self.kernel.Print("Use '%agent SUBCOMMAND' for detailed help")

    # MCP Server Management
    def _handle_mcp(self, args):
        """Handle MCP subcommands"""
        if not args.strip():
            self.kernel.Error("Usage: %agent mcp [add|list|remove|clear]")
            return

        parts = args.split(None, 1)
        action = parts[0].lower()
        actionargs = parts[1] if len(parts) > 1 else ''

        if action == 'add':
            self._mcp_add(actionargs)
        elif action == 'list':
            self._mcp_list(actionargs)
        elif action == 'remove':
            self._mcp_remove(actionargs)
        elif action == 'clear':
            self._mcp_clear(actionargs)
        else:
            self.kernel.Error(f"Unknown MCP action: {action}")
            self.kernel.Print("Available actions: add, list, remove, clear")

    def _mcp_add(self, args):
        """Add an MCP server"""
        if not args.strip():
            self.kernel.Error("Usage: %agent mcp add NAME COMMAND [ARGS...]")
            return

        parts = args.split(None, 2)
        if len(parts) < 2:
            self.kernel.Error("Usage: %agent mcp add NAME COMMAND [ARGS...]")
            return

        name = parts[0]
        command = parts[1]
        server_args = parts[2].split() if len(parts) > 2 else []

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
                return

        self.kernel._mcp_servers.append({
            'name': name,
            'command': command,
            'args': server_args,
            'env': []
        })
        self.kernel.Print(f"Added MCP server '{name}'")

    def _mcp_list(self, args):
        """List MCP servers"""
        if not hasattr(self.kernel, '_mcp_servers') or not self.kernel._mcp_servers:
            self.kernel.Print("No MCP servers configured")
            return

        self.kernel.Print("Configured MCP servers:")
        for server in self.kernel._mcp_servers:
            args_str = ' '.join(server['args']) if server['args'] else '(no args)'
            self.kernel.Print(f"  - {server['name']}: {server['command']} {args_str}")

    def _mcp_remove(self, args):
        """Remove an MCP server"""
        if not args.strip():
            self.kernel.Error("Usage: %agent mcp remove NAME")
            return

        name = args.strip()

        if not hasattr(self.kernel, '_mcp_servers'):
            self.kernel.Error(f"No MCP server named '{name}' found")
            return

        for i, server in enumerate(self.kernel._mcp_servers):
            if server['name'] == name:
                self.kernel._mcp_servers.pop(i)
                self.kernel.Print(f"Removed MCP server '{name}'")
                return

        self.kernel.Error(f"No MCP server named '{name}' found")

    def _mcp_clear(self, args):
        """Clear all MCP servers"""
        if hasattr(self.kernel, '_mcp_servers'):
            count = len(self.kernel._mcp_servers)
            self.kernel._mcp_servers = []
            self.kernel.Print(f"Removed {count} MCP server(s)")
        else:
            self.kernel.Print("No MCP servers to clear")

    # Permission Management
    def _handle_permissions(self, args):
        """Handle permissions subcommands"""
        if not args.strip() or args.strip() == 'show':
            mode = getattr(self.kernel, '_permission_mode', 'auto')
            self.kernel.Print(f"Current permission mode: {mode}")
            self.kernel.Print("\nAvailable modes:")
            self.kernel.Print("  auto   - automatically approve all requests (current default)")
            self.kernel.Print("  manual - prompt for each request (not yet implemented)")
            self.kernel.Print("  deny   - automatically deny all requests")
            return

        parts = args.split(None, 1)
        action = parts[0].lower()

        if action == 'list':
            self._permissions_list()
        elif action in ['auto', 'manual', 'deny']:
            self._permissions_set(action)
        else:
            self.kernel.Error(f"Invalid permissions argument: {action}")
            self.kernel.Print("Usage: %agent permissions [auto|manual|deny|list|show]")

    def _permissions_set(self, mode):
        """Set permission mode"""
        if mode == 'manual':
            self.kernel.Print("Warning: manual mode is not yet fully implemented")
            self.kernel.Print("Falling back to 'auto' mode")
            mode = 'auto'

        self.kernel._permission_mode = mode
        self.kernel.Print(f"Permission mode set to: {mode}")

    def _permissions_list(self):
        """List permission history"""
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

    # Session Management
    def _handle_session(self, args):
        """Handle session subcommands"""
        if not args.strip():
            self.kernel.Error("Usage: %agent session [new|info|restart]")
            return

        parts = args.split(None, 1)
        action = parts[0].lower()
        actionargs = parts[1] if len(parts) > 1 else ''

        if action == 'new':
            self._session_new(actionargs)
        elif action == 'info':
            self._session_info(actionargs)
        elif action == 'restart':
            self._session_restart(actionargs)
        else:
            self.kernel.Error(f"Unknown session action: {action}")
            self.kernel.Print("Available actions: new, info, restart")

    def _session_new(self, args):
        """Create a new session"""
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

    def _session_info(self, args):
        """Display session information"""
        if not hasattr(self.kernel, '_session_id') or not self.kernel._session_id:
            self.kernel.Print("No active session")
            self.kernel.Print("\nUse '%agent session new' to create a session")
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

    def _session_restart(self, args):
        """Restart the current session"""
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

    # Agent Configuration
    def _handle_config(self, args):
        """Handle agent config"""
        if not args.strip():
            # Display current configuration
            self.kernel.Print("Current Agent Configuration:")
            self.kernel.Print(f"  Command: {self.kernel._agent_command}")
            if self.kernel._agent_args:
                self.kernel.Print(f"  Args: {' '.join(self.kernel._agent_args)}")
            else:
                self.kernel.Print("  Args: (none)")
            
            self.kernel.Print("\nEnvironment Variables:")
            if os.environ.get('OPENAI_API_KEY'):
                self.kernel.Print("  OPENAI_API_KEY: ✓ set")
            else:
                self.kernel.Print("  OPENAI_API_KEY: ✗ not set")
            
            if os.environ.get('CODEX_API_KEY'):
                self.kernel.Print("  CODEX_API_KEY: ✓ set")
            
            self.kernel.Print("\nUse '%agent config COMMAND [ARGS...]' to change")
            return

        # Parse new configuration
        parts = args.split(None, 1)
        command = parts[0]
        agent_args = parts[1].split() if len(parts) > 1 else []

        # Update configuration
        self.kernel._agent_command = command
        self.kernel._agent_args = agent_args

        self.kernel.Print("Agent configuration updated:")
        self.kernel.Print(f"  Command: {command}")
        if agent_args:
            self.kernel.Print(f"  Args: {' '.join(agent_args)}")
        
        # Check if there's an active session
        if hasattr(self.kernel, '_session_id') and self.kernel._session_id:
            self.kernel.Print("\nNote: Session is active. Use '%agent session restart' to apply changes.")

    def _handle_env(self, args):
        """Handle environment variables"""
        if not args.strip():
            # Display relevant environment variables
            self.kernel.Print("Relevant Environment Variables:")
            env_vars = [
                'OPENAI_API_KEY', 'CODEX_API_KEY', 'ANTHROPIC_API_KEY',
                'ACP_AGENT_COMMAND', 'ACP_AGENT_ARGS', 'DEBUG'
            ]
            
            for var in env_vars:
                value = os.environ.get(var)
                if value:
                    # Mask API keys
                    if 'KEY' in var or 'TOKEN' in var:
                        display_value = value[:8] + "..." if len(value) > 8 else "***"
                    else:
                        display_value = value
                    self.kernel.Print(f"  {var}={display_value}")
                else:
                    self.kernel.Print(f"  {var}=(not set)")
            
            self.kernel.Print("\nUse '%agent env KEY=VALUE' to set a variable")
            return

        # Parse and set environment variable
        if '=' not in args:
            self.kernel.Error("Usage: %agent env KEY=VALUE")
            return

        key, value = args.split('=', 1)
        key = key.strip()
        value = value.strip()

        if not key:
            self.kernel.Error("Environment variable name cannot be empty")
            return

        os.environ[key] = value
        
        # Mask sensitive values in output
        if 'KEY' in key or 'TOKEN' in key or 'SECRET' in key:
            display_value = value[:8] + "..." if len(value) > 8 else "***"
        else:
            display_value = value
        
        self.kernel.Print(f"Set {key}={display_value}")


def register_magics(kernel):
    kernel.register_magics(AgentMagic)
