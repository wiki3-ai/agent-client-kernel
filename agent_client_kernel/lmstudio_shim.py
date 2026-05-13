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

    ACP_LMSTUDIO_SHIM_TARGET (alias SHIM_TARGET)  upstream LM Studio base URL
                                                  (default http://192.168.64.1:1234)
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

DEFAULT_TARGET = "http://192.168.64.1:1234"
DEFAULT_PORT = 18234


def _env_target() -> str:
    return os.environ.get("ACP_LMSTUDIO_SHIM_TARGET") or os.environ.get(
        "SHIM_TARGET", DEFAULT_TARGET
    )


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

    * ``{"type":"namespace","name":NS,"tools":[...]}`` is expanded into its
      nested function tools with names prefixed by ``NS``.
    * Tools with any other non-function type are dropped.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload
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
            logger.debug("dropping unsupported tool type=%r", ttype)
    payload["tools"] = out
    return payload


def _make_handler(target: str) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def _proxy(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""

            if body and self.headers.get("Content-Type", "").startswith(
                "application/json"
            ):
                try:
                    payload = json.loads(body)
                    if "tools" in payload:
                        payload = flatten_tools(payload)
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
                for key, value in upstream.getheaders():
                    if key.lower() in (
                        "transfer-encoding",
                        "connection",
                        "content-encoding",
                        "content-length",
                    ):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                while True:
                    chunk = upstream.read(8192)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
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
    """Run the shim synchronously until interrupted."""
    target = target or _env_target()
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
