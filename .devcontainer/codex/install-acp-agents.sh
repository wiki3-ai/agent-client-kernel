#!/usr/bin/env bash
# install-acp-agents.sh
#
# Install pinned native-binary ACP agents based on agent-versions.json.
# Used by the Codex devcontainer Dockerfile. Verifies SHA-256 checksums for
# every download and refuses to proceed on mismatch.
#
# Inputs:
#   $1 - path to agent-versions.json
#   $2 - install prefix (binaries go to "$prefix/bin")
#
# Environment:
#   TARGETARCH - Docker buildx platform arch (amd64|arm64). Falls back to
#                detecting the current arch with `uname -m`.

set -euo pipefail

manifest="${1:-/tmp/agent-versions.json}"
prefix="${2:-/usr/local}"

if [[ ! -f "$manifest" ]]; then
    echo "ERROR: manifest not found: $manifest" >&2
    exit 1
fi

# Resolve build target arch.
docker_arch="${TARGETARCH:-}"
if [[ -z "$docker_arch" ]]; then
    case "$(uname -m)" in
        x86_64)        docker_arch=amd64 ;;
        aarch64|arm64) docker_arch=arm64 ;;
        *) echo "ERROR: unsupported arch $(uname -m)" >&2; exit 1 ;;
    esac
fi

target=$(python3 -c "
import json, sys
m = json.load(open('$manifest'))
arch = m['docker_arch_to_target'].get('$docker_arch')
if not arch:
    sys.exit(f'unsupported docker arch: $docker_arch')
print(arch)
")

echo "==> Installing ACP agents for target: $target (docker arch: $docker_arch)"

mkdir -p "$prefix/bin"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Iterate agents listed in the manifest.
agents=$(python3 -c "import json; print('\n'.join(json.load(open('$manifest'))['agents']))")

for agent in $agents; do
    eval "$(python3 - <<PY
import json
m = json.load(open('$manifest'))
a = m['agents']['$agent']
sha = a['sha256'].get('$target')
if not sha:
    raise SystemExit(f"No SHA-256 pinned for $agent on $target")
url = a['url_template'].format(tag=a['release_tag'], version=a['version'], target='$target')
extracted = a['extracted_binary'].format(target='$target')
print(f"VERSION={a['version']}")
print(f"URL={url}")
print(f"SHA256={sha}")
print(f"EXTRACTED={extracted}")
print(f"INSTALL_AS={a['install_as']}")
PY
)"

    echo "==> $agent $VERSION"
    echo "    url:    $URL"
    echo "    sha256: $SHA256"

    archive="$tmp/$agent.tar.gz"
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$archive" "$URL"
    echo "$SHA256  $archive" | sha256sum -c -
    tar -xzf "$archive" -C "$tmp"
    install -m 0755 "$tmp/$EXTRACTED" "$prefix/bin/$INSTALL_AS"
    rm -f "$archive" "$tmp/$EXTRACTED"
    echo "    installed: $prefix/bin/$INSTALL_AS"
done

echo "==> All agents installed successfully."
