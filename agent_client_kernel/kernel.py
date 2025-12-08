"""
The main file for the Agent Client Protocol Jupyter kernel
"""

import asyncio
import asyncio.subprocess as aio_subprocess
import logging
import os
import sys
from pathlib import Path

# Configure logging to output to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

from metakernel import MetaKernel

from acp import (
    Client,
    ClientSideConnection,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    RequestError,
    SessionNotification,
    text_block,
    PROTOCOL_VERSION,
)
from acp.schema import (
    RequestPermissionResponse,
    AllowedOutcome,
    DeniedOutcome,
)

# Import nest_asyncio to allow nested event loops
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from . import __version__, KERNEL_NAME, DISPLAY_NAME


class ACPClient(Client):
    """ACP Client implementation for the Jupyter kernel"""
    
    def __init__(self, kernel) -> None:
        self._kernel = kernel
        self._log = logging.getLogger(__name__)
        self._terminals = {}  # Track active terminals by ID
    
    async def requestPermission(self, params):
        """Handle permission requests from the agent"""
        self._log.info("Permission requested: %s", params)
        
        # Get permission mode from kernel (default: auto)
        mode = getattr(self._kernel, '_permission_mode', 'auto')
        
        # Record the permission request
        if not hasattr(self._kernel, '_permission_history'):
            self._kernel._permission_history = []
        
        if mode == 'deny':
            approved = False
            outcome = DeniedOutcome(outcome='cancelled')
        elif mode == 'manual':
            # TODO: Implement interactive prompting
            # For now, fall back to auto-approve
            approved = True
            # Select the first 'allow' option if available
            option_id = self._get_allow_option_id(params.options)
            outcome = AllowedOutcome(outcome='selected', optionId=option_id)
        else:  # auto mode
            approved = True
            # Select the first 'allow' option if available
            option_id = self._get_allow_option_id(params.options)
            outcome = AllowedOutcome(outcome='selected', optionId=option_id)
        
        self._kernel._permission_history.append({
            'request': str(params),
            'approved': approved
        })
        
        return RequestPermissionResponse(outcome=outcome)
    
    def _get_allow_option_id(self, options):
        """Get the first allow option ID from the permission options"""
        # Look for allow_once or allow_always options
        for option in options:
            if option.kind in ('allow_once', 'allow_always'):
                return option.optionId
        # Fallback to the first option if no allow option is found
        if options:
            return options[0].optionId
        # Ultimate fallback
        return 'approved'
    
    async def writeTextFile(self, params):
        """Handle file write requests"""
        from acp.schema import WriteTextFileResponse
        
        self._log.info("Writing file: %s", params.path)
        
        try:
            # Resolve the path relative to session CWD
            file_path = Path(params.path)
            if not file_path.is_absolute():
                file_path = Path(self._kernel._session_cwd) / file_path
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the content to the file
            file_path.write_text(params.content, encoding='utf-8')
            
            self._log.info("Successfully wrote file: %s", file_path)
            return WriteTextFileResponse()
        except Exception as e:
            self._log.error("Error writing file %s: %s", params.path, e)
            raise RequestError.internal_error(f"Failed to write file: {str(e)}")
    
    async def readTextFile(self, params):
        """Handle file read requests"""
        from acp.schema import ReadTextFileResponse
        
        self._log.info("Reading file: %s", params.path)
        
        try:
            # Resolve the path relative to session CWD
            file_path = Path(params.path)
            if not file_path.is_absolute():
                file_path = Path(self._kernel._session_cwd) / file_path
            
            # Check if file exists
            if not file_path.exists():
                raise RequestError.invalid_params(f"File not found: {params.path}")
            
            # Read the file content
            content = file_path.read_text(encoding='utf-8')
            
            # Handle line parameter (read from specific line)
            if params.line is not None:
                lines = content.splitlines(keepends=True)
                if params.line > 0 and params.line <= len(lines):
                    # Get lines starting from the specified line
                    content = ''.join(lines[params.line - 1:])
            
            # Handle limit parameter (limit number of characters)
            if params.limit is not None and params.limit > 0:
                content = content[:params.limit]
            
            self._log.info("Successfully read file: %s (%d chars)", file_path, len(content))
            return ReadTextFileResponse(content=content)
        except RequestError:
            raise
        except Exception as e:
            self._log.error("Error reading file %s: %s", params.path, e)
            raise RequestError.internal_error(f"Failed to read file: {str(e)}")
    
    async def createTerminal(self, params):
        """Handle terminal creation requests"""
        from acp.schema import CreateTerminalResponse
        import uuid
        
        self._log.info("Creating terminal: %s %s", params.command, params.args or [])
        
        try:
            # Generate a unique terminal ID
            terminal_id = str(uuid.uuid4())
            
            # Determine working directory
            cwd = params.cwd
            if cwd is None:
                cwd = self._kernel._session_cwd
            elif not Path(cwd).is_absolute():
                cwd = str(Path(self._kernel._session_cwd) / cwd)
            
            # Prepare environment variables
            env = os.environ.copy()
            if params.env:
                for env_var in params.env:
                    env[env_var.name] = env_var.value
            
            # Create the terminal process
            process = await asyncio.create_subprocess_exec(
                params.command,
                *(params.args or []),
                stdin=aio_subprocess.PIPE,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.STDOUT,  # Merge stderr into stdout
                cwd=cwd,
                env=env,
            )
            
            # Store terminal state
            self._terminals[terminal_id] = {
                'process': process,
                'output_buffer': [],
                'output_byte_limit': params.outputByteLimit or 1024 * 1024,  # Default 1MB
                'total_bytes': 0,
            }
            
            # Start reading output in the background
            asyncio.create_task(self._read_terminal_output(terminal_id))
            
            self._log.info("Created terminal %s with PID %s", terminal_id, process.pid)
            return CreateTerminalResponse(terminalId=terminal_id)
        except Exception as e:
            self._log.error("Error creating terminal: %s", e)
            raise RequestError.internal_error(f"Failed to create terminal: {str(e)}")
    
    async def _read_terminal_output(self, terminal_id):
        """Background task to read terminal output"""
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return
        
        process = terminal['process']
        try:
            while True:
                # Read available output
                chunk = await process.stdout.read(4096)
                if not chunk:
                    # Process has ended
                    break
                
                # Check byte limit
                if terminal['total_bytes'] + len(chunk) > terminal['output_byte_limit']:
                    # Truncate to limit
                    remaining = terminal['output_byte_limit'] - terminal['total_bytes']
                    if remaining > 0:
                        chunk = chunk[:remaining]
                        terminal['output_buffer'].append(chunk)
                        terminal['total_bytes'] += len(chunk)
                    break
                
                terminal['output_buffer'].append(chunk)
                terminal['total_bytes'] += len(chunk)
        except Exception as e:
            self._log.error("Error reading terminal output for %s: %s", terminal_id, e)
    
    async def terminalOutput(self, params):
        """Handle terminal output requests"""
        from acp.schema import TerminalOutputResponse, TerminalExitStatus
        
        terminal_id = params.terminalId
        self._log.info("Getting output for terminal: %s", terminal_id)
        
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            raise RequestError.invalid_params(f"Terminal not found: {terminal_id}")
        
        process = terminal['process']
        
        # Get all buffered output
        output_bytes = b''.join(terminal['output_buffer'])
        output = output_bytes.decode('utf-8', errors='replace')
        
        # Clear the buffer after reading
        terminal['output_buffer'] = []
        
        # Check if process has exited
        exit_status = None
        truncated = terminal['total_bytes'] >= terminal['output_byte_limit']
        
        if process.returncode is not None:
            exit_status = TerminalExitStatus(
                exitCode=process.returncode,
                signal=None  # Unix signals not easily accessible in asyncio
            )
        
        return TerminalOutputResponse(
            output=output,
            truncated=truncated,
            exitStatus=exit_status
        )
    
    async def releaseTerminal(self, params):
        """Handle terminal release requests"""
        from acp.schema import ReleaseTerminalResponse
        
        terminal_id = params.terminalId
        self._log.info("Releasing terminal: %s", terminal_id)
        
        terminal = self._terminals.get(terminal_id)
        if terminal:
            process = terminal['process']
            
            # Close stdin if still open
            if process.stdin and not process.stdin.is_closing():
                process.stdin.close()
            
            # Remove from tracking
            del self._terminals[terminal_id]
            
            self._log.info("Released terminal %s", terminal_id)
        
        return ReleaseTerminalResponse()
    
    async def waitForTerminalExit(self, params):
        """Handle terminal exit wait requests"""
        from acp.schema import WaitForTerminalExitResponse
        
        terminal_id = params.terminalId
        self._log.info("Waiting for terminal exit: %s", terminal_id)
        
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            raise RequestError.invalid_params(f"Terminal not found: {terminal_id}")
        
        process = terminal['process']
        
        # Wait for the process to complete
        await process.wait()
        
        self._log.info("Terminal %s exited with code %s", terminal_id, process.returncode)
        
        return WaitForTerminalExitResponse(
            exitCode=process.returncode,
            signal=None  # Unix signals not easily accessible in asyncio
        )
    
    async def killTerminal(self, params):
        """Handle terminal kill requests"""
        from acp.schema import KillTerminalCommandResponse
        
        terminal_id = params.terminalId
        self._log.info("Killing terminal: %s", terminal_id)
        
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            raise RequestError.invalid_params(f"Terminal not found: {terminal_id}")
        
        process = terminal['process']
        
        try:
            # Try graceful termination first
            process.terminate()
            
            # Wait a bit for graceful shutdown
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill if termination didn't work
                process.kill()
                await process.wait()
            
            self._log.info("Killed terminal %s", terminal_id)
        except Exception as e:
            self._log.error("Error killing terminal %s: %s", terminal_id, e)
            raise RequestError.internal_error(f"Failed to kill terminal: {str(e)}")
        
        return KillTerminalCommandResponse()
    
    async def sessionUpdate(self, params: SessionNotification) -> None:
        """Handle session updates from the agent"""
        update = params.update
        if isinstance(update, dict):
            kind = update.get("sessionUpdate")
            content = update.get("content")
        else:
            kind = getattr(update, "sessionUpdate", None)
            content = getattr(update, "content", None)
        
        if kind != "agent_message_chunk" or content is None:
            return
        
        if isinstance(content, dict):
            text = content.get("text", "")
        else:
            text = getattr(content, "text", "")
        
        if text:
            # Send output to notebook
            self._kernel._agent_output.append(text)
    
    async def extMethod(self, method: str, params: dict) -> dict:
        """Handle extension method calls"""
        raise RequestError.method_not_found(method)
    
    async def extNotification(self, method: str, params: dict) -> None:
        """Handle extension notifications"""
        pass


class ACPKernel(MetaKernel):
    """Jupyter kernel for Agent Client Protocol"""
    
    implementation = 'Agent Client Protocol'
    implementation_version = __version__
    banner = "Agent Client Protocol Kernel - interact with ACP agents"
    language = 'text'
    language_version = '0.1'
    language_info = {
        'name': 'agent',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
        'help_links': MetaKernel.help_links,
    }
    
    kernel_json = {
        'argv': [sys.executable, '-m', 'agent_client_kernel', '-f', '{connection_file}'],
        'display_name': DISPLAY_NAME,
        'language': 'agent',
        'name': KERNEL_NAME
    }
    
    def __init__(self, *args, **kwargs):
        """Initialize the kernel"""
        super(ACPKernel, self).__init__(*args, **kwargs)
        self._log = logging.getLogger(__name__)
        self._log.info("Starting ACP kernel %s", __version__)
        
        # ACP connection tracking
        self._session_id = None
        self._conn = None
        self._proc = None
        self._agent_output = []
        self._event_loop = None
        
        # Agent configuration - can be overridden via environment variables
        self._agent_command = os.environ.get('ACP_AGENT_COMMAND', 'codex-acp')
        self._agent_args = os.environ.get('ACP_AGENT_ARGS', '').split() if os.environ.get('ACP_AGENT_ARGS') else []
        
        # Session configuration
        self._session_cwd = os.getcwd()
        self._mcp_servers = []
        
        # Permission configuration
        self._permission_mode = 'auto'
        self._permission_history = []
        
        # Load custom magics
        self._load_magics()
    
    def _load_magics(self):
        """Load custom magic commands"""
        import importlib
        
        # Load the unified agent magic module
        try:
            module = importlib.import_module('agent_client_kernel.magics.agent_magic')
            if hasattr(module, 'register_magics'):
                module.register_magics(self)
                self._log.info("Loaded unified agent magic")
        except Exception as e:
            self._log.error(f"Failed to load agent magic: {e}")
    
    def get_usage(self):
        """Return usage information"""
        return f"""Agent Client Protocol Kernel

This kernel allows interaction with ACP agents directly from Jupyter notebooks.
Simply type your prompts and execute cells to communicate with the agent.

Current agent: {self._agent_command} {' '.join(self._agent_args)}

Agent Management Command:
Use '%agent' with subcommands to manage configuration and sessions:

  MCP Server Configuration:
    %agent mcp add NAME COMMAND [ARGS...]  - add MCP server
    %agent mcp list                        - list MCP servers
    %agent mcp remove NAME                 - remove MCP server
    %agent mcp clear                       - clear all MCP servers

  Permission Configuration:
    %agent permissions [auto|manual|deny]  - set permission mode
    %agent permissions list                - show permission history

  Session Management:
    %agent session new [CWD]               - create new session
    %agent session info                    - show session information
    %agent session restart                 - restart current session

  Agent Configuration:
    %agent config [COMMAND [ARGS...]]     - configure agent command
    %agent env [KEY=VALUE]                 - set environment variables

For detailed help: %agent (shows all subcommands)
For help on any magic: %agent?

Supported agents:
- codex-acp (OpenAI Codex, requires OPENAI_API_KEY or CODEX_API_KEY)
- Any ACP-compatible agent
"""
    
    def get_kernel_help_on(self, info, level=0, none_on_fail=False):
        """Get help on an object.  Called by the help magic.
        
        This method provides context-sensitive help for expressions in the kernel.
        It is called by the MetaKernel help system when users type expressions
        followed by '?' or use the %help magic.
        
        Args:
            info: Dictionary containing parsed code information with keys:
                  'code' - the expression to get help on
                  'obj' - the object name
            level: 0 for brief help (docstring), 1 for detailed help
            none_on_fail: If True, return None on failure; otherwise return error message
        
        Returns:
            Help text for the expression, or None/error message if not found
        """
        if not info.get('code'):
            return None if none_on_fail else ''
        
        expr = info.get('obj', info.get('code', '')).strip()
        
        # For 'agent' or '%agent', return the same help as %agent?
        if expr.lower() in ['agent', '%agent']:
            # Get the agent magic's docstring (same as %agent?)
            try:
                from agent_client_kernel.magics.agent_magic import AgentMagic
                agent_magic = AgentMagic(self)
                return agent_magic.line_agent.__doc__ or "No help available for %agent"
            except:
                return "No help available for %agent"
        
        # Handle agent subcommands like 'agent mcp', 'agent session', etc.
        expr_lower = expr.lower()
        if expr_lower.startswith('agent ') or expr_lower.startswith('%agent '):
            # Extract the subcommand
            parts = expr.split(None, 1)
            if len(parts) > 1:
                subcommand = parts[1].lower()
                return self._get_agent_subcommand_help(subcommand)
        
        # Handle standalone subcommands like 'mcp', 'session', etc.
        # These are treated as agent subcommands
        subcommand_help = self._get_agent_subcommand_help(expr_lower)
        if subcommand_help:
            return subcommand_help
        
        # For anything else, return None or indicate no help available
        if none_on_fail:
            return None
        else:
            return None
    
    def _get_agent_subcommand_help(self, subcommand):
        """Get help text for agent subcommands.
        
        Args:
            subcommand: The subcommand name (e.g., 'mcp', 'session', 'permissions', 'config', 'env')
        
        Returns:
            Help text for the subcommand, or None if not recognized
        """
        subcommand = subcommand.lower().strip()
        
        if subcommand == 'mcp':
            return """MCP Server Configuration

MCP (Model Context Protocol) servers provide additional capabilities to the agent.

Commands:
  %agent mcp add NAME COMMAND [ARGS...]
      Add an MCP server to the session
      Example: %agent mcp add filesystem /usr/local/bin/mcp-server-filesystem
      
  %agent mcp list
      List all configured MCP servers
      
  %agent mcp remove NAME
      Remove a specific MCP server by name
      
  %agent mcp clear
      Remove all configured MCP servers
"""
        
        elif subcommand == 'session':
            return """Session Management

Sessions represent an active connection to an ACP agent.

Commands:
  %agent session new [CWD]
      Create a new session, optionally with a specific working directory
      Example: %agent session new /path/to/project
      
  %agent session info
      Display information about the current session
      
  %agent session restart
      Restart the current session with the same configuration
"""
        
        elif subcommand == 'permissions':
            return """Permission Configuration

Control how the kernel handles permission requests from the agent.

Commands:
  %agent permissions [auto|manual|deny]
      Set the permission mode:
      - auto: automatically approve all requests (default)
      - manual: prompt for each request (not yet implemented)
      - deny: automatically deny all requests
      
  %agent permissions list
      Show the history of permission requests
"""
        
        elif subcommand == 'config':
            return """Agent Configuration

Configure the ACP agent command and arguments.

Commands:
  %agent config [COMMAND [ARGS...]]
      Set the agent command to use
      Example: %agent config codex-acp --verbose
      
      Without arguments, displays the current configuration
"""
        
        elif subcommand == 'env':
            return """Environment Variables

Set environment variables for the agent.

Commands:
  %agent env [KEY=VALUE]
      Set an environment variable
      Example: %agent env OPENAI_API_KEY=sk-...
      
      Without arguments, displays relevant environment variables
"""
        
        else:
            return None
    
    async def _start_agent(self):
        """Start the ACP agent process"""
        if self._proc is not None:
            return
        
        self._log.info("Starting agent: %s %s", self._agent_command, ' '.join(self._agent_args))
        
        try:
            # Find the agent executable
            program_path = Path(self._agent_command)
            spawn_program = self._agent_command
            spawn_args = self._agent_args
            
            if program_path.exists() and not os.access(program_path, os.X_OK):
                spawn_program = sys.executable
                spawn_args = [str(program_path), *self._agent_args]
            
            # Start the agent process
            self._proc = await asyncio.create_subprocess_exec(
                spawn_program,
                *spawn_args,
                stdin=aio_subprocess.PIPE,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.PIPE,
            )
            
            if self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("Agent process does not expose stdio pipes")
            
            # Create client connection
            client_impl = ACPClient(self)
            self._conn = ClientSideConnection(
                lambda _agent: client_impl,
                self._proc.stdin,
                self._proc.stdout
            )
            
            # Initialize the agent
            await self._conn.initialize(
                InitializeRequest(protocolVersion=PROTOCOL_VERSION, clientCapabilities=None)
            )
            
            # Create a new session with MCP servers
            from acp.schema import StdioMcpServer
            
            mcp_servers = []
            for server_config in self._mcp_servers:
                mcp_servers.append(StdioMcpServer(
                    name=server_config['name'],
                    command=server_config['command'],
                    args=server_config['args'],
                    env=server_config.get('env', [])
                ))
            
            session = await self._conn.newSession(
                NewSessionRequest(mcpServers=mcp_servers, cwd=self._session_cwd)
            )
            self._session_id = session.sessionId
            
            self._log.info("Agent started with session ID: %s", self._session_id)
        except Exception as e:
            # Clean up on failure to prevent inconsistent state
            self._log.error("Failed to start agent: %s", e)
            await self._stop_agent()
            raise
    
    async def _stop_agent(self):
        """Stop the ACP agent process"""
        if self._proc is None:
            return
        
        self._log.info("Stopping agent")
        
        if self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        
        self._proc = None
        self._conn = None
        self._session_id = None
    
    async def _send_prompt(self, code: str) -> str:
        """Send a prompt to the agent and get the response"""
        # Ensure agent is started
        if self._conn is None or self._session_id is None:
            await self._start_agent()
        
        # Clear previous output
        self._agent_output = []
        
        # Send the prompt
        await self._conn.prompt(
            PromptRequest(
                sessionId=self._session_id,
                prompt=[text_block(code)],
            )
        )
        
        # Wait a bit for the response to accumulate
        await asyncio.sleep(0.5)
        
        # Return the accumulated output
        return ''.join(self._agent_output) if self._agent_output else "No response from agent"
    
    def do_execute_direct(self, code):
        """
        Execute code directly - this is the main entry point for metakernel
        """
        if not code.strip():
            return ""
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async prompt
        try:
            result = loop.run_until_complete(self._send_prompt(code))
            return result
        except Exception as e:
            self._log.error("Error sending prompt: %s", e, exc_info=True)
            return f"Error: {str(e)}\n\nMake sure the ACP agent is configured correctly.\nCurrent agent: {self._agent_command}"
    
    def do_shutdown(self, restart):
        """Shutdown the kernel"""
        # Stop the agent process
        if self._proc is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._stop_agent())
            except Exception as e:
                self._log.error("Error stopping agent: %s", e)
        
        return super().do_shutdown(restart)
    
    def repr(self, data):
        """Return string representation of data"""
        return str(data)

