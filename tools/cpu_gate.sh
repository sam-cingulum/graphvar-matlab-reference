#!/usr/bin/env bash
# Record CPU provenance and refuse anything but the approved AMD EPYC models.
set -euo pipefail
out="${1:?provenance file}"
{ echo "GitHub run=$GITHUB_RUN_ID attempt=$GITHUB_RUN_ATTEMPT sha=$GITHUB_SHA"; lscpu; uname -a; } > "$out"
model="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
if ! printf '%s' "$model" | grep -Eq '^AMD EPYC .*(7763|9V45)'; then
  echo "CPU rejected: $model" >&2
  exit 86
fi
echo "CPU approved: $model"
