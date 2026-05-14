#!/usr/bin/env bash
#
# Jupyter Docker Stacks "before-notebook.d" hook: launch the LM Studio
# request shim so Codex (whether spawned by the kernel, codex-acp, or run
# directly from a shell) can reach LM Studio without hitting the
# namespace-tool rejection bug. The shim is a lightweight Python HTTP
# reverse proxy living at 127.0.0.1:18234; see
# agent_client_kernel/lmstudio_shim.py.
#
# IMPORTANT: start.sh's run-hooks.sh *sources* files ending in .sh
# (rather than exec'ing them), so this script runs in the parent shell.
# Do NOT `set -e`/`set -u` or call `exit` — that would mutate or kill
# start.sh itself before it can launch Jupyter. Use `return` at the top
# level instead.
#
# Upstream LM Studio bug being patched around:
#   https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1810
#   https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1812
#
# Controls:
#   ACP_LMSTUDIO_SHIM=off                disable the shim entirely
#   HOST_GATEWAY_IP=10.0.0.1             override upstream host (default
#                                        host.docker.internal)
#   ACP_LMSTUDIO_SHIM_TARGET=http://x:y  explicit upstream override
#   ACP_LMSTUDIO_SHIM_PORT=18234         local listen port
#
# Idempotent: returns cleanly if the shim is already running or disabled.

_lmstudio_shim_hook() {
    local port log_file shim_pid

    _lmstudio_shim_log() {
        printf '[lmstudio-shim hook] %s\n' "$*" >&2
    }

    if [[ "${ACP_LMSTUDIO_SHIM:-auto}" == "off" ]]; then
        _lmstudio_shim_log "ACP_LMSTUDIO_SHIM=off; skipping shim startup."
        return 0
    fi

    port="${ACP_LMSTUDIO_SHIM_PORT:-${SHIM_PORT:-18234}}"

    # If something is already listening, leave it alone. Safe to run
    # alongside the kernel's in-process ensure_running().
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); \
sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "$port" \
            >/dev/null 2>&1; then
        _lmstudio_shim_log "port $port already in use; assuming shim is up."
        return 0
    fi

    # Background the shim. setsid detaches it from the start.sh
    # controlling terminal so it survives the hook returning.
    log_file="${ACP_LMSTUDIO_SHIM_LOG:-/tmp/lmstudio-shim.log}"
    setsid nohup python3 -m agent_client_kernel.lmstudio_shim \
        >>"$log_file" 2>&1 < /dev/null &
    shim_pid=$!
    disown "$shim_pid" 2>/dev/null || true
    _lmstudio_shim_log "started shim pid=$shim_pid on port $port (log: $log_file)"
    return 0
}

_lmstudio_shim_hook
unset -f _lmstudio_shim_hook _lmstudio_shim_log
