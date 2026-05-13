# Codex 0.130 + LM Studio: namespace-tools shim

## The bug

Codex 0.130 unconditionally wraps MCP tools in a Responses-API "namespace"
envelope when speaking to any non-Bedrock provider:

```json
{ "type": "namespace", "name": "mcp__jupyter__", "tools": [ {"type":"function", ...}, ... ] }
```

LM Studio's OpenAI-compatible Responses endpoint validates tools with a
strict zod schema that only allows `type:"function"`. The request is
rejected before any inference happens:

```
{"error":{"message":"Invalid","param":"tools.15.type",
          "code":"invalid_string","type":"invalid_request_error"}}
```

`tools.15` is the namespace envelope produced from our
`[mcp_servers.jupyter]` entry; `tools.8` is `web_search`, another
non-function type Codex emits.

## Why we can't fix it at either end we control

- **Codex side.** `namespace_tools` is set on `ProviderCapabilities` in
  Rust (`provider.rs`, `tool_config.rs`, `spec_plan.rs`). It is the
  default for every non-Bedrock provider and has **no TOML knob**. We
  tried `wire_api = "chat"` to dodge the Responses path entirely — that
  key was removed in 0.130 (see
  <https://github.com/openai/codex/discussions/7782>).
- **MCP server side.** `jupyter-mcp-server` produces well-formed
  `type:"function"` tool specs. Codex itself is the one re-wrapping them.
- **AgentClientKernel side.** The kernel speaks ACP/JSON-RPC over stdio
  to `codex-acp`. The failing traffic is `codex-acp → LM Studio HTTPS`,
  inside a subprocess we don't intercept. Architecturally the kernel
  cannot fix this.

## What we shipped

A small Python reverse proxy that sits in front of LM Studio and rewrites
just the tools array before forwarding:

- Keeps `type:"function"` tools unchanged.
- Expands `{"type":"namespace","name":"mcp__jupyter__","tools":[...]}`
  into prefixed flat functions (`mcp__jupyter__list_notebooks`, etc.).
  Codex's tool registry round-trips those names without further
  translation.
- Drops other non-function types (`web_search`, `image_generation`) with
  a debug log — LM Studio can't execute them anyway.
- Streams Responses-API SSE through untouched.

### Files

- [`agent_client_kernel/lmstudio_shim.py`](../agent_client_kernel/lmstudio_shim.py)
  — shim module (`flatten_tools` + threaded HTTP server + `serve_forever`).
- [`scripts/lmstudio-shim.py`](../scripts/lmstudio-shim.py) — thin CLI wrapper.
- [`scripts/start-lmstudio-shim.sh`](../scripts/start-lmstudio-shim.sh) —
  idempotent background launcher (no-ops when port 18234 is already
  bound or `ACP_LMSTUDIO_SHIM=off`).
- [`.devcontainer/codex/devcontainer.json`](../.devcontainer/codex/devcontainer.json)
  — `postStartCommand` invokes the launcher; `containerEnv` exposes
  `ACP_LMSTUDIO_SHIM`, `ACP_LMSTUDIO_SHIM_TARGET`,
  `ACP_LMSTUDIO_SHIM_PORT`.
- [`.codex/config.toml`](../.codex/config.toml) — stripped of `profile` /
  `profiles` / `model_providers` (project-local configs reject those in
  0.130).
- `~/.codex/config.toml` —
  `[model_providers.local_lmstudio] base_url = "http://127.0.0.1:18234/v1"`
  routes only LM Studio traffic through the shim.

### Provider tie-in

The shim is selected purely by `base_url`. OpenAI / Anthropic / etc.
providers point straight at their hosted endpoints and never traverse
the shim, so leaving it running while using a different profile is a
no-op. To skip launching it entirely, set `ACP_LMSTUDIO_SHIM=off`.

### Verification

End-to-end test (Codex 0.130 → shim → LM Studio) listed all 15
`jupyter-mcp` tools cleanly with no validation errors.

## Reverted

An earlier attempt auto-started the shim from
`AgentClientKernel.__init__`. Rolled back: the kernel can't reach the
failing HTTP layer, so it's the wrong place. The module stays; the
in-kernel `start_background` / `maybe_start_from_env` helpers were
dropped along with the import.

## Upstream issues to file / track

1. **openai/codex — make `namespace_tools` provider-configurable.** The
   Rust capability is hardcoded per provider. Request a TOML override
   (e.g. `[model_providers.local_lmstudio] namespace_tools = false`) so
   users targeting Responses-API-compatible servers that don't speak
   namespace envelopes can opt out. This is the cleanest fix and would
   let us delete the shim. Files to point at:
   `codex-rs/.../provider.rs`, `tool_config.rs`, `spec_plan.rs`. Related
   discussion: <https://github.com/openai/codex/discussions/7782>.

2. **openai/codex — `model_providers` not honored from project-local
   `.codex/config.toml`.** 0.130 logs
   `Ignored unsupported project-local config keys: model_providers,
   profile, profiles`. Either document that these only work at user
   scope, or allow them at project scope so repos can pin provider
   config. Current behavior silently drops them, which is what bit us
   first.

3. **lmstudio-ai — accept `type:"namespace"` tools.** The OpenAI
   Responses spec includes namespace tools for hosted-MCP support.
   LM Studio's schema only allows `"function"`. They could (a) accept
   and ignore namespace tools, (b) expand them server-side, or (c) at
   minimum return a clearer error mentioning the unsupported type
   rather than a generic `invalid_string` on `tools.<N>.type`.

4. **zed-industries/agent-client-protocol (or wherever `codex-acp`
   lives) — surface upstream Responses errors verbatim.** The
   `tools.15.type` rejection only became visible after we ran a capture
   proxy; the ACP error path was opaque. Passing the upstream error
   body through would have shortened debugging from hours to minutes.

5. **datalayer/jupyter-mcp-server — no action needed**, but worth a
   courtesy note on its tracker linking back to (1)/(3) so anyone else
   hitting "MCP server breaks LM Studio with Codex 0.130" finds the
   workaround.

## Lessons worth remembering

- When Codex's HTTP request fails, the kernel can't fix it — the only
  intercept points are (a) Codex itself, (b) the model server, or (c)
  an HTTP proxy in between.
- `wire_api = "chat"` is gone in 0.130. Don't suggest it.
- Project-local `.codex/config.toml` silently drops `profile`,
  `profiles`, and `model_providers`. They have to live in
  `~/.codex/config.toml`.
- Default `ProviderCapabilities { namespace_tools: true }` applies to
  every non-Bedrock provider; there is no current way to disable it
  without source patches.
