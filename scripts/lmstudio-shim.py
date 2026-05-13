#!/usr/bin/env python3
"""LM Studio shim for Codex >= 0.130.

Codex 0.130 dropped the ``chat`` wire API and now emits MCP tools wrapped in a
Responses-API ``{"type": "namespace", "tools": [...]}`` envelope (plus an
optional ``{"type": "web_search"}`` tool). LM Studio's OpenAI-compatible
Responses endpoint only accepts ``type: "function"`` tools, so the request is
rejected with:

    {"error":{"message":"Invalid","param":"tools.<N>.type",
              "code":"invalid_string","type":"invalid_request_error"}}

This tiny reverse proxy sits between Codex and LM Studio and rewrites the
outgoing request body:

* ``type: "namespace"`` entries are flattened into individual ``type:
  "function"`` tools whose names are prefixed with the namespace
  (``mcp__jupyter__connect_to_jupyter`` etc.). Codex's tool registry
  already accepts those flat names, so tool calls round-trip without
  further rewriting.
* Non-function tool types Codex emits but LM Studio doesn't run
  (``web_search``, ``image_generation``, ...) are dropped with a log line.

The response stream is passed through untouched.

Usage:

    # in one terminal
    python3 scripts/lmstudio-shim.py

    # then point Codex's provider at the shim (user-level
    # ~/.codex/config.toml):
    #   [model_providers.local_lmstudio]
    #   base_url = "http://127.0.0.1:18234/v1"

Environment variables:

    SHIM_TARGET   upstream LM Studio base URL (default
                  ``http://192.168.64.1:1234``)
    SHIM_PORT     local port to listen on (default ``18234``)
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import urllib.error
import urllib.request

TARGET = os.environ.get("SHIM_TARGET", "http://192.168.64.1:1234")
LISTEN_PORT = int(os.environ.get("SHIM_PORT", "18234"))


def flatten_tools(payload: dict) -> dict:
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
            sys.stderr.write(
                f"[lmstudio-shim] dropping unsupported tool type={ttype!r}\n"
            )
    payload["tools"] = out
    return payload


class Handler(http.server.BaseHTTPRequestHandler):
    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""

        if body and self.headers.get("Content-Type", "").startswith("application/json"):
            try:
                payload = json.loads(body)
                if "tools" in payload:
                    payload = flatten_tools(payload)
                    body = json.dumps(payload).encode()
            except Exception as exc:  # noqa: BLE001 - log and forward original body
                sys.stderr.write(f"[lmstudio-shim] body rewrite failed: {exc}\n")

        url = TARGET + self.path
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
                "Content-Type", err.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            sys.stderr.write(f"[lmstudio-shim] upstream {err.code}: {data[:500]!r}\n")
        except urllib.error.URLError as err:
            msg = json.dumps({"error": {"message": f"shim upstream error: {err}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self) -> None:  # noqa: N802
        self._proxy("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._proxy("POST")

    def log_message(self, *_args, **_kwargs) -> None:  # silence default access log
        return


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    sys.stderr.write(
        f"[lmstudio-shim] listening on http://127.0.0.1:{LISTEN_PORT} -> {TARGET}\n"
    )
    ThreadedServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
