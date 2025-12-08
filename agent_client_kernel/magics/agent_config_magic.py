# Copyright (c) Jim White.
# Distributed under the terms of the BSD 3-Clause License.

from metakernel import Magic


class AgentConfigMagic(Magic):
    """Magic commands for agent configuration"""

    def line_agent_config(self, args=''):
        """
        %agent_config [COMMAND [ARGS...]] - configure the agent command

        This magic allows you to configure which ACP agent to use and
        its command-line arguments.

        Without arguments, displays the current configuration.
        With arguments, sets a new agent command.

        Arguments:
            COMMAND - path to the agent executable
            ARGS - optional command-line arguments (space-separated)

        Examples:
            %agent_config
            %agent_config codex-acp
            %agent_config /usr/local/bin/my-agent --verbose
            %agent_config python /path/to/agent.py --debug

        Note: Changes take effect when a new session is created.
        Use %session_restart or %new_session to apply changes.
        """
        if not args.strip():
            # Display current configuration
            self.kernel.Print("Current Agent Configuration:")
            self.kernel.Print(f"  Command: {self.kernel._agent_command}")
            if self.kernel._agent_args:
                self.kernel.Print(f"  Args: {' '.join(self.kernel._agent_args)}")
            else:
                self.kernel.Print("  Args: (none)")
            
            self.kernel.Print("\nEnvironment Variables:")
            import os
            if os.environ.get('OPENAI_API_KEY'):
                self.kernel.Print("  OPENAI_API_KEY: ✓ set")
            else:
                self.kernel.Print("  OPENAI_API_KEY: ✗ not set")
            
            if os.environ.get('CODEX_API_KEY'):
                self.kernel.Print("  CODEX_API_KEY: ✓ set")
            
            self.kernel.Print("\nUse %agent_config COMMAND [ARGS...] to change")
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
            self.kernel.Print("\nNote: Session is active. Use %session_restart to apply changes.")

    def line_agent_env(self, args=''):
        """
        %agent_env [KEY=VALUE] - set or display agent environment variables

        This magic allows you to set environment variables that will be
        available to the agent process.

        Without arguments, displays current environment variables.
        With KEY=VALUE, sets an environment variable.

        Examples:
            %agent_env
            %agent_env OPENAI_API_KEY=sk-...
            %agent_env DEBUG=1

        Note: Environment variables are set for the kernel process and
        inherited by the agent.
        """
        import os

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
            
            self.kernel.Print("\nUse %agent_env KEY=VALUE to set a variable")
            return

        # Parse and set environment variable
        if '=' not in args:
            self.kernel.Error("Usage: %agent_env KEY=VALUE")
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
    kernel.register_magics(AgentConfigMagic)
