#!/usr/bin/env bash
# Run every probe item in its own MATLAB process with a timeout.
set -uo pipefail
gv="${1:?GraphVar root}"; out="${2:?results dir}"
log="$out/probe_log.jsonl"; list="$out/items.txt"
mkdir -p "$out"
ml() { timeout "${2:-180}" matlab -batch "addpath('drivers'); $1"; }
ml "probe_mex('$gv','$list','list','')" 600 || { echo "listing failed" >&2; exit 1; }
ml "probe_mex('$gv','$log','env','')" 600
while IFS= read -r label; do
  before=$(wc -l < "$log")
  ml "probe_mex('$gv','$log','one','$label')" 180
  rc=$?
  after=$(wc -l < "$log")
  if [ "$after" -eq "$before" ]; then
    status=CRASHED; [ "$rc" -eq 124 ] && status=TIMEOUT
    printf '{"kind":"RESULT","item":"%s","data":{"status":"%s","exit_code":%d}}\n' "$label" "$status" "$rc" >> "$log"
  fi
  echo "$label -> rc=$rc"
done < "$list"
echo '{"kind":"END","item":""}' >> "$log"
