#!/usr/bin/env python3
"""Update pinned ACP agent versions and SHA-256 checksums.

Reads .devcontainer/codex/agent-versions.json, fetches the requested release
tarballs from GitHub for every supported target triple, computes their
SHA-256 checksums, and writes the manifest back in place.

Usage:
    # Refresh checksums for the versions currently pinned in the manifest:
    python scripts/update-agent-versions.py

    # Set a new version for one or more agents (release_tag is inferred from
    # the existing template unless --tag is given):
    python scripts/update-agent-versions.py \\
        --set codex=0.130.0 \\
        --set codex-acp=0.13.0

    # Set explicit release tags too (rarely needed):
    python scripts/update-agent-versions.py \\
        --set codex=0.130.0 --tag codex=rust-v0.130.0

    # Just check that pinned checksums still match upstream (CI-friendly):
    python scripts/update-agent-versions.py --check

The script fails loudly on any download error or checksum mismatch and never
silently overwrites a verified pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / ".devcontainer" / "codex" / "agent-versions.json"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "agent-client-kernel-update"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (https github)
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        return resp.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expand(template: str, *, tag: str, version: str, target: str) -> str:
    return template.format(tag=tag, version=version, target=target)


def _parse_kv_args(values: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values or []:
        if "=" not in v:
            sys.exit(f"--{flag} expects NAME=VALUE, got: {v!r}")
        k, val = v.split("=", 1)
        out[k.strip()] = val.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to agent-versions.json")
    parser.add_argument(
        "--set", dest="set_version", action="append", default=[],
        metavar="NAME=VERSION",
        help="Set a new version for an agent (repeatable).",
    )
    parser.add_argument(
        "--tag", dest="set_tag", action="append", default=[],
        metavar="NAME=TAG",
        help="Override the release tag for an agent (default: derived from existing tag pattern).",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify that pinned checksums still match upstream; do not modify the manifest.",
    )
    args = parser.parse_args()

    new_versions = _parse_kv_args(args.set_version, "set")
    new_tags = _parse_kv_args(args.set_tag, "tag")

    manifest = json.loads(args.manifest.read_text())
    targets = sorted(set(manifest["docker_arch_to_target"].values()))

    failures: list[str] = []
    changed = False

    for name, agent in manifest["agents"].items():
        old_version = agent["version"]
        old_tag = agent["release_tag"]

        version = new_versions.get(name, old_version)
        if name in new_tags:
            tag = new_tags[name]
        elif name in new_versions:
            # Derive new tag by replacing the version inside the old tag.
            if old_version in old_tag:
                tag = old_tag.replace(old_version, version)
            else:
                sys.exit(
                    f"Cannot infer new release_tag for {name}: old tag {old_tag!r} does not "
                    f"contain old version {old_version!r}. Pass --tag {name}=...",
                )
        else:
            tag = old_tag

        print(f"==> {name}: version={version} tag={tag}")
        new_sha: dict[str, str] = {}
        for target in targets:
            url = _expand(agent["url_template"], tag=tag, version=version, target=target)
            print(f"    fetching {url}")
            try:
                data = _download(url)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name} {target}: download failed: {exc}")
                continue
            digest = _sha256(data)
            new_sha[target] = digest
            print(f"      sha256: {digest}")

            if args.check:
                pinned = agent.get("sha256", {}).get(target)
                if pinned and pinned != digest:
                    failures.append(f"{name} {target}: pinned {pinned} != upstream {digest}")

        if not args.check:
            if (
                version != old_version
                or tag != old_tag
                or new_sha != agent.get("sha256", {})
            ):
                changed = True
            agent["version"] = version
            agent["release_tag"] = tag
            agent["sha256"] = new_sha

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    if args.check:
        print("\nAll pinned checksums match upstream.")
        return 0

    if changed:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nWrote {args.manifest}")
    else:
        print("\nNo changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
