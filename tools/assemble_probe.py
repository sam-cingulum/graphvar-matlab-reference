"""Fold the probe's JSON-lines log into one report with a status tally."""
import json, sys, collections
log, out = sys.argv[1], sys.argv[2]
results, ended = {}, False
for line in open(log):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if e["kind"] == "RESULT":
        results[e["item"]] = e["data"]
    elif e["kind"] == "END":
        ended = True
tally = collections.Counter(v.get("status") for k, v in results.items() if k != "env")
report = {"complete": ended, "tally": dict(tally), "results": results}
json.dump(report, open(out, "w"), indent=1)
print(json.dumps({"complete": ended, "tally": tally}, default=dict))
for k, v in sorted(results.items()):
    if k != "env":
        print(f'{v.get("status"):15s} {k}  {v.get("identifier","")} {str(v.get("message",""))[:100]}')
