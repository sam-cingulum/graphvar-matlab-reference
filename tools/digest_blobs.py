"""Turn the driver's JSON-lines output into the reference format.

Arrays the driver wrote as big-endian blobs become SHA-256 digests with nnz
and the first 64 words (see graphvar-spec/P0-reference-format.md).
"""
import hashlib, json, os, sys

src, blobdir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
POS0, NEG0 = bytes(8), bytes.fromhex("8000000000000000")


def fix(o):
    if isinstance(o, dict):
        if "blob" in o:
            data = open(os.path.join(blobdir, o["blob"]), "rb").read()
            words = [data[i:i + 8] for i in range(0, len(data), 8)]
            return {"size": o["size"], "sha256": hashlib.sha256(data).hexdigest(),
                    "nnz": sum(1 for w in words if w not in (POS0, NEG0)),
                    "head": [w.hex() for w in words[:64]]}
        return {k: fix(v) for k, v in o.items()}
    if isinstance(o, list):
        return [fix(v) for v in o]
    return o


records = []
for line in open(src):
    line = line.strip()
    if line:
        r = fix(json.loads(line))
        if r.get("error") == []:
            r["error"] = None
        records.append(r)
json.dump(records, open(dst, "w"), indent=0)
print(f"{dst}: {len(records)} records")
