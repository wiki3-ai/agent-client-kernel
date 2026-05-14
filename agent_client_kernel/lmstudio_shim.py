"""LM Studio request shim.

Codex 0.130 wraps MCP tools in a Responses-API ``{"type":"namespace", ...}``
envelope. LM Studio's OpenAI-compatible Responses endpoint only accepts tools
whose ``type`` is ``"function"``, so any request that includes MCP tools is
rejected with::

    {"error":{"message":"Invalid","param":"tools.<N>.type",
              "code":"invalid_string","type":"invalid_request_error"}}

This module is a tiny reverse proxy that rewrites the outgoing HTTP body:

* ``{"type":"namespace","name":NS,"tools":[...]}`` is flattened into
  individual ``{"type":"function", ...}`` entries whose names are prefixed
  with the namespace (e.g. ``mcp__jupyter__list_notebooks``). Codex's tool
  registry accepts those flat names, so tool calls round-trip without
  further translation.
* Non-function tool types Codex sometimes emits but LM Studio doesn't run
  (``web_search``, ``image_generation``, ...) are dropped with a log line.

Response streams pass through untouched.

The shim is only needed when Codex talks to a Responses-API server that
doesn't understand namespace tools (LM Studio, Ollama's OpenAI compat layer,
etc.). For the OpenAI hosted API it is unnecessary, so wire the proxy in
per-provider via that provider's ``base_url`` in ``~/.codex/config.toml``.

Run it standalone via ``scripts/lmstudio-shim.py`` or
``python -m agent_client_kernel.lmstudio_shim``.

Environment variables:

    ACP_LMSTUDIO_SHIM_TARGET (alias SHIM_TARGET)  explicit upstream LM Studio
                                                  base URL override, e.g.
                                                  http://host.docker.internal:1234.
    HOST_GATEWAY_IP                               host gateway IP/hostname,
                                                  injected by the container
                                                  host. Falls back to
                                                  host.docker.internal. Used to
                                                  build the upstream URL
                                                  ``http://${HOST_GATEWAY_IP}:1234``
                                                  when SHIM_TARGET is unset.
    ACP_LMSTUDIO_SHIM_PORT   (alias SHIM_PORT)    local port (default 18234)
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import socketserver
import sys
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PORT = 18234
DEFAULT_HOST_GATEWAY = "host.docker.internal"
DEFAULT_UPSTREAM_PORT = 1234


def _env_target() -> Optional[str]:
    explicit = os.environ.get("ACP_LMSTUDIO_SHIM_TARGET") or os.environ.get(
        "SHIM_TARGET"
    )
    if explicit:
        return explicit
    host = os.environ.get("HOST_GATEWAY_IP") or DEFAULT_HOST_GATEWAY
    if not host:
        return None
    return f"http://{host}:{DEFAULT_UPSTREAM_PORT}"


def _env_port() -> int:
    raw = os.environ.get("ACP_LMSTUDIO_SHIM_PORT") or os.environ.get(
        "SHIM_PORT", str(DEFAULT_PORT)
    )
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_PORT


def flatten_tools(payload: dict) -> dict:
    """Rewrite ``payload['tools']`` so every entry has ``type == "function"``.

    * ``{"type":"namespace","name":NS,"tools":[...]}`` is expanded into
      individual ``{"type":"function", ...}`` entries whose names are
      the namespace concatenated with the original name.
    * Tools with any other non-function type are dropped.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload
    logger.debug(
        "lmstudio-shim flatten_tools input types=%s",
        [t.get("type") for t in tools if isinstance(t, dict)],
    )
    out: list[dict] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype == "function":
            out.append(tool)
        elif ttype == "namespace":
            ns = tool.get("name", "ns")
            for nested in tool.get("tools", []) or []:
                if nested.get("type") != "function":
                    continue
                flat = dict(nested)
                flat["name"] = f"{ns}{nested.get('name', '')}"
                out.append(flat)
        else:
            logger.debug("lmstudio-shim dropping unsupported tool type=%r", ttype)
    payload["tools"] = out
    return payload


def collect_namespaces(payload: dict) -> list[str]:
    """Return the list of MCP namespace prefixes advertised in ``payload``.

    Codex sends MCP tools as ``{"type":"namespace","name":"mcp__<srv>__",...}``
    envelopes. We need those prefix strings to split flat function-call
    names returned by the model back into ``namespace`` + ``name``.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return []
    out: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            ns = tool.get("name")
            if isinstance(ns, str):
                out.append(ns)
    return out


def split_flat_name(name: str, namespaces: list[str]) -> tuple[Optional[str], str]:
    """Split a flat function-call name on a known MCP namespace prefix.

    Returns ``(namespace, tool_name)`` if ``name`` starts with one of the
    advertised ``namespaces``, otherwise ``(None, name)``. Longest match
    wins so overlapping prefixes are handled deterministically.
    """
    best: Optional[str] = None
    for ns in namespaces:
        if name.startswith(ns) and (best is None or len(ns) > len(best)):
            best = ns
    if best is None:
        return None, name
    return best, name[len(best):]


def flatten_input_history(payload: dict) -> dict:
    """Combine ``namespace`` + ``name`` on previously-emitted function calls.

    Codex's protocol type for ``function_call`` carries ``namespace`` and
    ``name`` as separate fields (codex-rs/protocol/src/models.rs). When
    that history is replayed to LM Studio, the upstream doesn't understand
    ``namespace`` and may reject it. Concatenate them back into a single
    flat ``name`` and drop the ``namespace`` key, mirroring what the model
    originally produced.
    """
    items = payload.get("input")
    if not isinstance(items, list):
        return payload
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        ns = item.get("namespace")
        name = item.get("name")
        if isinstance(ns, str) and isinstance(name, str):
            item["name"] = f"{ns}{name}"
            item.pop("namespace", None)
    return payload


# --- response-side translation ---------------------------------------------


_FUNCTION_CALL_KEYS_TRIGGER = ("type", "name")


def rewrite_response_obj(obj, namespaces: list[str]) -> None:
    """Recursively split flat ``mcp__<srv>__<tool>`` names on response items.

    Walks ``obj`` in-place. Whenever it finds a dict that looks like a
    Responses-API ``function_call`` item (``type == "function_call"`` and
    a string ``name``), it splits the ``name`` on a known namespace prefix
    and adds a ``namespace`` field. Codex's
    ``ResponseItem::FunctionCall`` deserializer (codex-rs/protocol/src/
    models.rs) reads both fields and uses
    ``ToolName::new(namespace, name)`` so the dispatch in
    ``codex-rs/core/src/tools/registry.rs`` finds the registered MCP
    handler.
    """
    if not namespaces:
        return
    if isinstance(obj, dict):
        if obj.get("type") == "function_call" and isinstance(obj.get("name"), str):
            ns, tool = split_flat_name(obj["name"], namespaces)
            if ns is not None:
                obj["name"] = tool
                obj["namespace"] = ns
        for value in obj.values():
            rewrite_response_obj(value, namespaces)
    elif isinstance(obj, list):
        for value in obj:
            rewrite_response_obj(value, namespaces)


class SSERewriter:
    """Buffer an SSE byte stream and rewrite ``function_call`` items.

    SSE events are separated by ``\\n\\n``. Each event may span multiple
    chunks, so :meth:`feed` accumulates bytes and only emits complete
    events. Each event's ``data:`` payload is parsed as JSON and passed
    through :func:`rewrite_response_obj`; non-JSON payloads (e.g.
    ``data: [DONE]``) and events without MCP function calls pass through
    unchanged.

    When ``namespaces`` is empty, the rewriter is a pure passthrough —
    it never parses the bytes — so the only-MCP cost is paid on streams
    that actually advertised MCP tools.
    """

    def __init__(self, namespaces: list[str]) -> None:
        self._namespaces = list(namespaces)
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> bytes:
        if not self._namespaces:
            return chunk
        if not chunk:
            return b""
        self._buffer.extend(chunk)
        out = bytearray()
        while True:
            # SSE event boundary is a blank line: \n\n (or \r\n\r\n).
            sep = self._find_event_boundary(self._buffer)
            if sep is None:
                break
            event_bytes = bytes(self._buffer[: sep[1]])
            del self._buffer[: sep[1]]
            out.extend(self._rewrite_event(event_bytes))
        return bytes(out)

    def flush(self) -> bytes:
        """Return any buffered bytes that never reached an event boundary."""
        if not self._buffer:
            return b""
        leftover = bytes(self._buffer)
        self._buffer.clear()
        # Best-effort: try to rewrite as if it were a complete event.
        return self._rewrite_event(leftover)

    @staticmethod
    def _find_event_boundary(buf: bytearray) -> Optional[tuple[int, int]]:
        """Return ``(start, end)`` of the first event boundary in ``buf``.

        The end index is exclusive of the boundary bytes. The returned
        ``start`` marks the position of the boundary itself for callers
        that need it (currently unused; kept symmetric with ``end``).
        """
        for sep in (b"\r\n\r\n", b"\n\n"):
            idx = buf.find(sep)
            if idx >= 0:
                return idx, idx + len(sep)
        return None

    def _rewrite_event(self, event: bytes) -> bytes:
        try:
            text = event.decode("utf-8")
        except UnicodeDecodeError:
            return event
        # Preserve original line endings so HTTP framing isn't disturbed.
        # Most SSE servers use \n; handle \r\n by detecting the first
        # newline style.
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.split(newline)
        changed = False
        for i, line in enumerate(lines):
            if not line.startswith("data:"):
                continue
            # data: <payload>  (single space after colon is conventional
            # but optional per the SSE spec).
            payload = line[5:].lstrip(" ")
            if not payload or payload.startswith("["):
                # e.g. ``data: [DONE]`` sentinel; never JSON.
                continue
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            before = json.dumps(obj, sort_keys=True)
            rewrite_response_obj(obj, self._namespaces)
            after = json.dumps(obj, sort_keys=True)
            if before == after:
                continue
            lines[i] = "data: " + json.dumps(obj)
            changed = True
        if not changed:
            return event
        return newline.join(lines).encode("utf-8")


def _make_handler(target: str) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        # Speak HTTP/1.1 so the base class auto-handles Expect: 100-continue
        # (Codex sends Expect on large Responses-API POSTs and aborts the
        # send if it never sees a 100). We still close the connection after
        # each response to keep the proxy logic simple.
        protocol_version = "HTTP/1.1"

        def _proxy(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""

            namespaces: list[str] = []
            if body and self.headers.get("Content-Type", "").startswith(
                "application/json"
            ):
                try:
                    payload = json.loads(body)
                    if "tools" in payload:
                        namespaces = collect_namespaces(payload)
                        payload = flatten_tools(payload)
                    if "input" in payload:
                        payload = flatten_input_history(payload)
                    body = json.dumps(payload).encode()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("lmstudio-shim body rewrite failed: %s", exc)

            url = target + self.path
            req = urllib.request.Request(
                url, data=body if method != "GET" else None, method=method
            )
            for key, value in self.headers.items():
                if key.lower() in ("host", "content-length", "connection"):
                    continue
                req.add_header(key, value)
            if body:
                req.add_header("Content-Length", str(len(body)))

            try:
                upstream = urllib.request.urlopen(req, timeout=600)
                self.send_response(upstream.status)
                content_type = ""
                for key, value in upstream.getheaders():
                    if key.lower() in (
                        "transfer-encoding",
                        "connection",
                        "content-encoding",
                        "content-length",
                    ):
                        continue
                    if key.lower() == "content-type":
                        content_type = value
                    self.send_header(key, value)
                # Force close: each request gets a fresh upstream connection.
                self.send_header("Connection", "close")
                self.end_headers()
                is_sse = "text/event-stream" in content_type.lower()
                rewriter: Optional[SSERewriter] = (
                    SSERewriter(namespaces) if is_sse and namespaces else None
                )
                while True:
                    chunk = upstream.read(8192)
                    if not chunk:
                        break
                    out = rewriter.feed(chunk) if rewriter is not None else chunk
                    if not out:
                        continue
                    try:
                        self.wfile.write(out)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                if rewriter is not None:
                    tail = rewriter.flush()
                    if tail:
                        try:
                            self.wfile.write(tail)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
            except urllib.error.HTTPError as err:
                data = err.read()
                self.send_response(err.code)
                self.send_header(
                    "Content-Type",
                    err.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                logger.warning(
                    "lmstudio-shim upstream %s: %r", err.code, data[:500]
                )
            except urllib.error.URLError as err:
                msg = json.dumps(
                    {"error": {"message": f"shim upstream error: {err}"}}
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def do_GET(self) -> None:  # noqa: N802
            self._proxy("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._proxy("POST")

        def log_message(self, *_args, **_kwargs) -> None:
            return

    return Handler


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_forever(
    target: Optional[str] = None, port: Optional[int] = None
) -> None:
    """Run the shim synchronously until interrupted.

    ``target`` must be provided explicitly or via the
    ``ACP_LMSTUDIO_SHIM_TARGET`` env var. There is no hardcoded fallback
    so users can't accidentally proxy to someone else's machine.
    """
    target = target or _env_target()
    if not target:
        raise SystemExit(
            "lmstudio-shim: no upstream target configured. Set "
            "HOST_GATEWAY_IP or ACP_LMSTUDIO_SHIM_TARGET "
            "(e.g. http://host.docker.internal:1234), or pass --target."
        )
    port = port or _env_port()
    server = _ThreadedServer(("127.0.0.1", port), _make_handler(target))
    sys.stderr.write(
        f"[lmstudio-shim] listening on http://127.0.0.1:{port} -> {target}\n"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    serve_forever()


# -- in-process launcher used by the kernel ---------------------------------

_BACKGROUND_THREAD: "threading.Thread | None" = None


def _port_in_use(port: int) -> bool:
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


def ensure_running() -> Optional[str]:
    """Start the shim in a background daemon thread if it isn't already.

    Returns the upstream target URL the shim is forwarding to, or ``None``
    if the shim was suppressed (``ACP_LMSTUDIO_SHIM=off``) or could not be
    started. Idempotent across calls within the same process and silently
    no-ops if some other process is already listening on the port.
    """
    import threading

    global _BACKGROUND_THREAD

    if os.environ.get("ACP_LMSTUDIO_SHIM", "auto").lower() == "off":
        return None

    target = _env_target()
    if not target:
        return None

    port = _env_port()
    if _BACKGROUND_THREAD is not None and _BACKGROUND_THREAD.is_alive():
        return target
    if _port_in_use(port):
        return target

    def _run() -> None:
        try:
            serve_forever(target=target, port=port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lmstudio-shim background thread exited: %s", exc)

    thread = threading.Thread(
        target=_run, name="lmstudio-shim", daemon=True
    )
    thread.start()
    _BACKGROUND_THREAD = thread
    return target
