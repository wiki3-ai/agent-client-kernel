"""Probe-diagnose-apply auditor for Codex CLI configuration.

This module inspects the local environment (network, LM Studio, MCP services,
tooling) and reconciles ``~/.codex/config.toml`` against it. The companion
notebook ``codex-configuration.ipynb`` is a thin wrapper around the functions
exposed here:

    >>> from scripts.codex_configure import probe, load_configs, diagnose, apply_fixes
    >>> env = probe()
    >>> docs = load_configs()
    >>> issues, fixes = diagnose(env, docs)
    >>> apply_fixes(env, docs, issues, fixes, apply=True)

The notebook keeps responsibility for *displaying* state; this script owns the
*logic*. Doing it this way avoids fighting the notebook cell editor when the
audit logic is iterated on.

Codex 0.130 notes
-----------------
* ``profile``, ``profiles``, ``model_providers`` are refused at project scope.
* ``wire_api = "chat"`` was removed; LM Studio uses ``"responses"``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

HOME = Path.home()
USER_CONFIG = HOME / ".codex" / "config.toml"
PROJECT_ROOT = Path(os.environ.get("ACK_PROJECT_ROOT", "/workspaces/agent-client-kernel"))
PROJECT_CONFIG = PROJECT_ROOT / ".codex" / "config.toml"
CATALOG_DIR = HOME / ".codex" / "model-catalogs"
CATALOG_FILE = CATALOG_DIR / "local-lmstudio.json"

SHIM_PORT = int(os.environ.get("ACP_LMSTUDIO_SHIM_PORT") or os.environ.get("SHIM_PORT") or 18234)
SHIM_OFF = os.environ.get("ACP_LMSTUDIO_SHIM", "").lower() == "off"
HOST_GATEWAY = os.environ.get("HOST_GATEWAY_IP") or "host.docker.internal"
_EXPLICIT_TARGET = os.environ.get("ACP_LMSTUDIO_SHIM_TARGET") or os.environ.get("SHIM_TARGET")
LMSTUDIO_DIRECT = _EXPLICIT_TARGET or f"http://{HOST_GATEWAY}:1234"
LMSTUDIO_VIA_SHIM = f"http://127.0.0.1:{SHIM_PORT}/v1"

PREFERRED_MODEL = "qwen/qwen3-coder-next"
PREFERRED_PROFILE = "lmstudio-qwen3-coder"
PREFERRED_PROVIDER = "local_lmstudio"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_json(url: str, timeout: float = 3.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
        return None, str(e)


def probe() -> dict[str, Any]:
    """Inspect the live environment. Returns a dict of facts."""
    facts: dict[str, Any] = {
        "host_gateway": HOST_GATEWAY,
        "host_gateway_resolves": False,
        "lmstudio_direct_url": LMSTUDIO_DIRECT,
        "lmstudio_shim_port": SHIM_PORT,
        "lmstudio_shim_disabled_by_env": SHIM_OFF,
    }

    try:
        socket.gethostbyname(HOST_GATEWAY)
        facts["host_gateway_resolves"] = True
    except OSError:
        pass

    facts["lmstudio_direct_reachable"] = _tcp_open(HOST_GATEWAY, 1234)
    facts["lmstudio_models"] = []
    facts["lmstudio_loaded"] = []
    facts["lmstudio_loaded_meta"] = []

    # LM Studio native endpoint reports loaded vs not-loaded; /v1/models does not.
    status, payload = _http_json(f"{LMSTUDIO_DIRECT}/api/v0/models")
    facts["lmstudio_native_models_status"] = status
    if isinstance(payload, dict):
        for m in payload.get("data") or payload.get("models") or []:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("modelKey") or m.get("name")
            if not mid:
                continue
            facts["lmstudio_models"].append(mid)
            if (m.get("state") or "").lower() == "loaded":
                facts["lmstudio_loaded"].append(mid)
                facts["lmstudio_loaded_meta"].append({
                    "id": mid,
                    "type": m.get("type"),
                    "arch": m.get("arch"),
                    "publisher": m.get("publisher"),
                    "max_context_length": m.get("max_context_length"),
                    "loaded_context_length": m.get("loaded_context_length"),
                    "capabilities": m.get("capabilities") or [],
                })
    else:
        # Fallback: older LM Studio builds — assume everything is loaded.
        facts["lmstudio_native_error"] = payload
        status, payload = _http_json(f"{LMSTUDIO_DIRECT}/v1/models")
        facts["lmstudio_direct_models_status"] = status
        if isinstance(payload, dict):
            ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict)]
            facts["lmstudio_models"] = ids
            facts["lmstudio_loaded"] = ids
            facts["lmstudio_loaded_meta"] = [{"id": i} for i in ids]
            facts["lmstudio_loaded_state_unknown"] = True
        else:
            facts["lmstudio_direct_error"] = payload

    facts["lmstudio_shim_running"] = _tcp_open("127.0.0.1", SHIM_PORT)
    if facts["lmstudio_shim_running"]:
        status, _ = _http_json(f"{LMSTUDIO_VIA_SHIM}/models")
        facts["lmstudio_shim_models_status"] = status

    facts["ollama_reachable"] = _tcp_open("127.0.0.1", 11434) or _tcp_open(HOST_GATEWAY, 11434)
    if facts["ollama_reachable"]:
        for url in ("http://127.0.0.1:11434/api/tags", f"http://{HOST_GATEWAY}:11434/api/tags"):
            status, payload = _http_json(url)
            if isinstance(payload, dict):
                facts["ollama_url"] = url.rsplit("/api", 1)[0]
                facts["ollama_models"] = [m.get("name") for m in payload.get("models", [])]
                break

    facts["codex_bin"] = shutil.which("codex")
    facts["codex_acp_bin"] = shutil.which("codex-acp") or os.environ.get("ACP_AGENT_COMMAND")
    facts["uvx_bin"] = shutil.which("uvx")
    facts["jupyter_mcp_via_uvx"] = bool(facts["uvx_bin"])
    try:
        import paper_search_mcp  # noqa: F401
        facts["paper_search_mcp_installed"] = True
    except ImportError:
        facts["paper_search_mcp_installed"] = False
    facts["jupyter_lab_reachable"] = _tcp_open(
        "127.0.0.1", int(os.environ.get("JUPYTER_PORT", "8888"))
    )
    return facts


# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------

def _load_toml(path: Path):
    if not path.exists():
        return None
    return tomlkit.parse(path.read_text())


@dataclass
class Configs:
    user_doc: Any = None
    project_doc: Any = None
    catalog: dict | None = None


def load_configs() -> Configs:
    cfg = Configs()
    cfg.user_doc = _load_toml(USER_CONFIG)
    cfg.project_doc = _load_toml(PROJECT_CONFIG)
    if CATALOG_FILE.exists():
        cfg.catalog = json.loads(CATALOG_FILE.read_text())
    return cfg


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------

def _profile_name_for(model_id: str) -> str:
    safe = model_id.replace("/", "-").replace(":", "-")
    return f"lmstudio-{safe}"


@dataclass
class DiagnoseResult:
    issues: list[tuple[str, str, Any]] = field(default_factory=list)
    fixes: dict[str, tuple[str, Any]] = field(default_factory=dict)


def diagnose(env: dict, cfg: Configs) -> DiagnoseResult:
    res = DiagnoseResult()
    issues = res.issues
    fixes = res.fixes

    lm_loaded = set(env.get("lmstudio_loaded") or [])
    lm_loaded_meta = env.get("lmstudio_loaded_meta") or []
    lm_loaded_llms = [
        m for m in lm_loaded_meta
        if (m.get("type") or "llm").lower() in {"llm", "vlm"}
    ]
    lm_loaded_llm_ids = [m["id"] for m in lm_loaded_llms]
    preferred_loaded = PREFERRED_MODEL in lm_loaded
    shim_up = env["lmstudio_shim_running"] and not SHIM_OFF
    lmstudio_up = bool(env["lmstudio_direct_reachable"])

    if not lmstudio_up:
        issues.append(("error", f"LM Studio not reachable at {LMSTUDIO_DIRECT}. Start it on the host (port 1234).", None))
    elif not lm_loaded:
        issues.append(("warn", "LM Studio reachable but no models reported as loaded.", None))
    elif not preferred_loaded:
        issues.append(("info", f"Preferred model {PREFERRED_MODEL!r} not loaded. Currently loaded: {sorted(lm_loaded)}", None))

    if not SHIM_OFF and not shim_up:
        issues.append(("error", f"LM Studio namespace-tools shim not listening on 127.0.0.1:{SHIM_PORT}.", None))

    user_doc = cfg.user_doc
    if user_doc is None:
        issues.append(("error", f"{USER_CONFIG} is missing.", "create_user_config"))
        fixes["create_user_config"] = ("create_user_config", None)
        return res

    expected_base_url = LMSTUDIO_VIA_SHIM if not SHIM_OFF else f"{LMSTUDIO_DIRECT}/v1"
    prov = user_doc.get("model_providers", {}).get(PREFERRED_PROVIDER)
    if prov is None:
        issues.append(("error", f"[model_providers.{PREFERRED_PROVIDER}] missing.", "set_provider"))
        fixes["set_provider"] = ("set_provider", expected_base_url)
    else:
        if prov.get("base_url") != expected_base_url:
            issues.append(("warn", f"model_providers.{PREFERRED_PROVIDER}.base_url is {prov.get('base_url')!r}; expected {expected_base_url!r}.", "set_provider_url"))
            fixes["set_provider_url"] = ("set_provider_url", expected_base_url)
        if prov.get("wire_api") != "responses":
            issues.append(("warn", f"model_providers.{PREFERRED_PROVIDER}.wire_api should be 'responses'.", "set_wire_api"))
            fixes["set_wire_api"] = ("set_wire_api", "responses")

    profiles = user_doc.get("profiles", {})
    if PREFERRED_PROFILE not in profiles:
        issues.append(("error", f"profile {PREFERRED_PROFILE!r} not defined.", "add_profile"))
        fixes["add_profile"] = ("add_profile", PREFERRED_PROFILE)
    else:
        prof = profiles[PREFERRED_PROFILE]
        if prof.get("model") != PREFERRED_MODEL:
            issues.append(("warn", f"profile {PREFERRED_PROFILE}.model={prof.get('model')!r}, expected {PREFERRED_MODEL!r}.", "fix_profile_model"))
            fixes["fix_profile_model"] = ("fix_profile_model", PREFERRED_MODEL)
        if prof.get("model_provider") != PREFERRED_PROVIDER:
            issues.append(("warn", f"profile {PREFERRED_PROFILE}.model_provider={prof.get('model_provider')!r}, expected {PREFERRED_PROVIDER!r}.", "fix_profile_provider"))
            fixes["fix_profile_provider"] = ("fix_profile_provider", PREFERRED_PROVIDER)

    # Pick default profile based on what's actually loaded right now.
    desired = None
    current = user_doc.get("profile")
    current_model = profiles.get(current, {}).get("model") if current else None
    lm_profiles = {n: p for n, p in profiles.items() if p.get("model_provider") == PREFERRED_PROVIDER}

    if preferred_loaded:
        desired = PREFERRED_PROFILE
    elif current_model and current_model in lm_loaded:
        desired = current
    if desired is None:
        for name, p in lm_profiles.items():
            if p.get("model") in lm_loaded:
                desired = name
                break

    # Nothing matches — auto-provision a profile and a catalog entry for
    # whatever LM Studio actually has loaded.
    if desired is None and lm_loaded_llms:
        target = lm_loaded_llms[0]
        new_profile = _profile_name_for(target["id"])
        desired = new_profile
        fixes["auto_provision_loaded"] = ("auto_provision_loaded", target)
        issues.append((
            "warn",
            f"LM Studio has {sorted(lm_loaded_llm_ids)} loaded but no existing profile matches. "
            f"Will auto-provision profile {new_profile!r} (model {target['id']!r}, "
            f"ctx={target.get('max_context_length')}) and a catalog entry on apply.",
            "auto_provision_loaded",
        ))

    if desired and current != desired:
        if "auto_provision_loaded" not in fixes:
            issues.append((
                "warn",
                f"top-level `profile` is {current!r} (model={current_model!r}); "
                f"LM Studio currently has {sorted(lm_loaded)} loaded — best matching profile is {desired!r}.",
                "set_default_profile",
            ))
        fixes["set_default_profile"] = ("set_default_profile", desired)

    # Catalog
    if not CATALOG_FILE.exists():
        issues.append(("error", f"Model catalog {CATALOG_FILE} missing.", "create_catalog"))
        fixes["create_catalog"] = ("create_catalog", None)
    elif cfg.catalog is not None:
        slugs = {m["slug"] for m in cfg.catalog.get("models", [])}
        if PREFERRED_MODEL not in slugs:
            issues.append(("warn", f"Catalog has no entry for {PREFERRED_MODEL!r}.", "add_catalog_entry"))
            fixes["add_catalog_entry"] = ("add_catalog_entry", PREFERRED_MODEL)

    # Project config sanity
    if cfg.project_doc is not None:
        for forbidden in ("profile", "profiles", "model_providers"):
            if forbidden in cfg.project_doc:
                issues.append(("warn", f"project .codex/config.toml has `{forbidden}` — Codex 0.130+ ignores it at project scope.", None))
        mcp = cfg.project_doc.get("mcp_servers", {})
        if "jupyter" in mcp and not env["jupyter_lab_reachable"]:
            issues.append(("warn", "mcp_servers.jupyter configured but Jupyter Lab not on 127.0.0.1:8888.", None))
        if "paper-search" in mcp and not env["paper_search_mcp_installed"]:
            issues.append(("warn", "mcp_servers.paper-search configured but `paper_search_mcp` not importable.", None))
        if not env["uvx_bin"] and mcp.get("jupyter", {}).get("command") == "uvx":
            issues.append(("error", "mcp_servers.jupyter uses `uvx` but uvx is not on PATH.", None))

    sev_order = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda x: sev_order.get(x[0], 99))
    return res


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

USER_CONFIG_TEMPLATE = '''# Codex user-level config — managed by scripts/codex_configure.py.
# Holds the keys Codex 0.130+ refuses to read from a project-local config
# (profile, profiles, model_providers). Project-local overrides
# (mcp_servers, sandbox_mode) stay in .codex/config.toml.

profile = "{profile}"
'''

PROFILE_DEFAULTS = {
    "lmstudio-qwen3-coder": {
        "model": "qwen/qwen3-coder-next",
        "model_provider": "local_lmstudio",
        "model_catalog_json": str(CATALOG_FILE),
        "model_context_window": 262144,
        "model_auto_compact_token_limit": 240000,
        "model_reasoning_summary": "none",
        "model_supports_reasoning_summaries": False,
        "sandbox_mode": "danger-full-access",
        "personality": "pragmatic",
    },
}

PROVIDER_DEFAULTS = {
    "local_lmstudio": {
        "name": "LM Studio",
        "base_url": LMSTUDIO_VIA_SHIM if not SHIM_OFF else f"{LMSTUDIO_DIRECT}/v1",
        "wire_api": "responses",
        "requires_openai_auth": False,
        "supports_websockets": False,
        "stream_idle_timeout_ms": 300000,
        "request_max_retries": 1,
        "stream_max_retries": 0,
    },
}


def _backup(path: Path) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = path.with_name(path.name + f".bak.{ts}")
    bak.write_bytes(path.read_bytes())
    return bak


def _ensure_table(doc, key):
    if key not in doc:
        doc[key] = tomlkit.table()
    return doc[key]


def _set_provider(doc, base_url=None):
    providers = _ensure_table(doc, "model_providers")
    if PREFERRED_PROVIDER not in providers:
        providers[PREFERRED_PROVIDER] = tomlkit.table()
    prov = providers[PREFERRED_PROVIDER]
    defaults = dict(PROVIDER_DEFAULTS["local_lmstudio"])
    if base_url:
        defaults["base_url"] = base_url
    for k, v in defaults.items():
        if k not in prov:
            prov[k] = v
    if base_url:
        prov["base_url"] = base_url


def _profile_template(name: str) -> dict:
    return dict(PROFILE_DEFAULTS.get(name, PROFILE_DEFAULTS[PREFERRED_PROFILE]))


def _add_profile(doc, name, *, model=None, provider=None, context_window=None):
    profiles = _ensure_table(doc, "profiles")
    if name not in profiles:
        profiles[name] = tomlkit.table()
    prof = profiles[name]
    defaults = _profile_template(name)
    if model:
        defaults["model"] = model
    if provider:
        defaults["model_provider"] = provider
    if context_window:
        defaults["model_context_window"] = int(context_window)
        defaults["model_auto_compact_token_limit"] = max(1024, int(context_window) * 9 // 10)
    for k, v in defaults.items():
        if k not in prof:
            prof[k] = v
    if model:
        prof["model"] = model
    if provider:
        prof["model_provider"] = provider
    if context_window:
        prof["model_context_window"] = int(context_window)
        prof["model_auto_compact_token_limit"] = max(1024, int(context_window) * 9 // 10)


def _catalog_entry_for(model_id: str, max_ctx: int | None, display_name: str | None = None) -> dict:
    ctx = int(max_ctx) if max_ctx else 32768
    return {
        "slug": model_id,
        "display_name": display_name or f"{model_id} via LM Studio",
        "description": f"Local model served by LM Studio ({model_id}).",
        "default_reasoning_level": "low",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Minimal local-model reasoning control"}
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "minimal_client_version": "0.0.1",
        "supported_in_api": True,
        "availability_nux": None,
        "upgrade": None,
        "priority": 100,
        "base_instructions": "You are Codex, a coding agent. Use the provided tool-calling interface for tool use.",
        "supports_reasoning_summaries": False,
        "reasoning_summary_format": "none",
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": "low",
        "apply_patch_tool_type": "function",
        "input_modalities": ["text"],
        "supports_image_detail_original": False,
        "truncation_policy": {"mode": "tokens", "limit": max(1024, ctx * 9 // 10)},
        "supports_parallel_tool_calls": False,
        "context_window": ctx,
        "max_context_window": ctx,
        "auto_compact_token_limit": max(1024, ctx * 9 // 10),
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
    }


CATALOG_ENTRY_TEMPLATE = {
    PREFERRED_MODEL: _catalog_entry_for(PREFERRED_MODEL, 262144, "Qwen 3 Coder Next via LM Studio"),
}


def _apply_to_doc(doc, key: str, value: Any) -> None:
    if key in {"set_provider", "set_provider_url"}:
        _set_provider(doc, base_url=value)
    elif key == "set_wire_api":
        _set_provider(doc)
        doc["model_providers"][PREFERRED_PROVIDER]["wire_api"] = value
    elif key == "add_profile":
        _add_profile(doc, value)
    elif key == "fix_profile_model":
        _add_profile(doc, PREFERRED_PROFILE, model=value)
    elif key == "fix_profile_provider":
        _add_profile(doc, PREFERRED_PROFILE, provider=value)
    elif key == "set_default_profile":
        doc["profile"] = value
    elif key == "auto_provision_loaded":
        meta = value
        new_profile = _profile_name_for(meta["id"])
        _set_provider(doc)
        _add_profile(
            doc,
            new_profile,
            model=meta["id"],
            provider=PREFERRED_PROVIDER,
            context_window=meta.get("max_context_length"),
        )
        doc["profile"] = new_profile


def apply_fixes(env: dict, cfg: Configs, diag: DiagnoseResult, *, apply: bool = False) -> dict:
    """Apply planned fixes. Returns a report dict.

    With ``apply=False`` (default), this is a dry run: nothing is written.
    """
    planned = [(sev, msg, key) for sev, msg, key in diag.issues if key]
    report = {"planned": planned, "written": []}
    if not planned:
        return report
    if not apply:
        return report

    # 1. Catalog
    catalog = cfg.catalog
    cat_changed = False
    if catalog is None:
        CATALOG_DIR.mkdir(parents=True, exist_ok=True)
        catalog = {"models": []}
        cat_changed = True
    catalog_slugs = {m["slug"] for m in catalog.get("models", [])}
    for sev, msg, key in planned:
        if key == "add_catalog_entry":
            entry = CATALOG_ENTRY_TEMPLATE.get(PREFERRED_MODEL)
            if entry and PREFERRED_MODEL not in catalog_slugs:
                catalog["models"].append(entry)
                catalog_slugs.add(PREFERRED_MODEL)
                cat_changed = True
        elif key == "auto_provision_loaded":
            meta = diag.fixes[key][1]
            slug = meta["id"]
            if slug not in catalog_slugs:
                catalog["models"].append(_catalog_entry_for(slug, meta.get("max_context_length")))
                catalog_slugs.add(slug)
                cat_changed = True

    if cat_changed:
        if CATALOG_FILE.exists():
            _backup(CATALOG_FILE)
        CATALOG_FILE.write_text(json.dumps(catalog, indent=2) + "\n")
        cfg.catalog = catalog
        report["written"].append(str(CATALOG_FILE))

    # 2. User config
    user_doc = cfg.user_doc
    if user_doc is None:
        USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        user_doc = tomlkit.parse(USER_CONFIG_TEMPLATE.format(profile=PREFERRED_PROFILE))
        _set_provider(user_doc)
        _add_profile(user_doc, PREFERRED_PROFILE)
    else:
        _backup(USER_CONFIG)

    # Apply auto_provision before set_default so the profile exists.
    ordered = sorted(planned, key=lambda x: 0 if x[2] == "auto_provision_loaded" else 1)
    for _sev, _msg, key in ordered:
        if key in {"create_user_config", "create_catalog", "add_catalog_entry"}:
            continue
        fix = diag.fixes.get(key)
        val = fix[1] if isinstance(fix, tuple) and len(fix) > 1 else None
        _apply_to_doc(user_doc, key, val)
    USER_CONFIG.write_text(tomlkit.dumps(user_doc))
    cfg.user_doc = user_doc
    report["written"].append(str(USER_CONFIG))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(apply: bool = False) -> None:
    env = probe()
    cfg = load_configs()
    diag = diagnose(env, cfg)

    print("--- environment ---")
    print(json.dumps(env, indent=2, default=str))
    print()
    print("--- issues ---")
    if not diag.issues:
        print("All checks passed.")
    for sev, msg, key in diag.issues:
        tag = {"error": "[ERROR]", "warn": "[WARN] ", "info": "[INFO] "}[sev]
        suffix = f"  (fix: {key})" if key else ""
        print(f"{tag} {msg}{suffix}")

    report = apply_fixes(env, cfg, diag, apply=apply)
    print()
    print("--- plan ---")
    if not report["planned"]:
        print("No fixable issues.")
    for sev, msg, key in report["planned"]:
        print(f"  - [{sev}] {key}: {msg}")
    if apply and report["written"]:
        print()
        print("--- wrote ---")
        for p in report["written"]:
            print(" ", p)
        print()
        print("--- new ~/.codex/config.toml ---")
        print(USER_CONFIG.read_text())
    elif report["planned"]:
        print()
        print("Pass apply=True to write changes.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes to ~/.codex/config.toml and catalog")
    args = ap.parse_args()
    main(apply=args.apply)
