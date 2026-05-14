#!/usr/bin/env bash
# Launch the LM Studio request shim in the background, if it isn't already.
#
# The shim flattens Codex 0.130+ "namespace" tool envelopes into plain
# function tools so LM Studio's Responses-API server accepts requests with
# MCP tools attached. It is only used by providers whose ``base_url`` in
# ``~/.codex/config.toml`` points at it (default http://127.0.0.1:18234/v1);
# direct OpenAI / Anthropic / etc. traffic does not traverse the shim.
#
# Upstream target resolution (first match wins):
#   1. ACP_LMSTUDIO_SHIM_TARGET (or SHIM_TARGET) — explicit override.
#   2. http://${HOST_GATEWAY_IP}:1234 — HOST_GATEWAY_IP is normally injected
#      by the container host.
#   3. http://host.docker.internal:1234 — default fallback.
#
# Set ACP_LMSTUDIO_SHIM=off to skip launching.
# Override listen port via ACP_LMSTUDIO_SHIM_PORT (default 18234).

set -euo pipefail

if [[ "${ACP_LMSTUDIO_SHIM:-auto}" == "off" ]]; then
  echo "[lmstudio-shim] disabled via ACP_LMSTUDIO_SHIM=off"
  exit 0
fi

TARGET="${ACP_LMSTUDIO_SHIM_TARGET:-${SHIM_TARGET:-}}"
if [[ -z "${TARGET}" ]]; then
  HOST="${HOST_GATEWAY_IP:-host.docker.internal}"
  TARGET="http://${HOST}:1234"
fi
export ACP_LMSTUDIO_SHIM_TARGET="${TARGET}"

PORT="${ACP_LMSTUDIO_SHIM_PORT:-${SHIM_PORT:-18234}}"

# Idempotent: bail if something is already listening on the port.
if (exec 3<>/dev/tcp/127.0.0.1/"${PORT}") 2>/dev/null; then
  exec 3<&-; exec 3>&-
  echo "[lmstudio-shim] port ${PORT} already in use, not starting"
  exit 0
fi

LOG="${ACP_LMSTUDIO_SHIM_LOG:-/tmp/lmstudio-shim.log}"
# Fully detach from the postStartCommand shell so VS Code does not keep
# waiting on inherited file descriptors. setsid + </dev/null + & is the
# combination that reliably releases the parent.
setsid nohup python3 -m agent_client_kernel.lmstudio_shim \
  </dev/null >"${LOG}" 2>&1 &
disown || true
echo "[lmstudio-shim] started on 127.0.0.1:${PORT} (log: ${LOG})"
