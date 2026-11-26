"""P2-03 battery: path transitivity, M76 (binary) and M77 (weighted).

Writes <out>/cmeasures.json and <out>/matrices.json (a copy of the P2-02
hand-built graphs, so {"p2_matrix": name} resolves here). Reads cases/phase1,
cases/p103 and cases/p2b next to <out>; run gen_p103.py and gen_p2b.py first.
"""
import json, os, sys

OUT = sys.argv[1]


def main():
    os.makedirs(OUT, exist_ok=True)
    base = os.path.dirname(OUT.rstrip("/")) or "."
    p1cases = json.load(open(os.path.join(base, "phase1", "construct.json")))
    p1mats = json.load(open(os.path.join(base, "phase1", "matrices.json")))
    p103mats = json.load(open(os.path.join(base, "p103", "matrices.json")))
    p2bmats = json.load(open(os.path.join(base, "p2b", "matrices.json")))
    json.dump(p2bmats, open(os.path.join(OUT, "matrices.json"), "w"))
    square = {k for k, v in p1mats.items() if v["size"][0] == v["size"][1] and k not in ("nan_5", "inf_5")}
    mc = []
    for k in p1mats:
        if k in square and k.startswith("g_"):
            mc.append({"graph": {"phase1_matrix": k}, "measures": ["M76"]})
        elif k in square and k.startswith("w_"):
            mc.append({"graph": {"phase1_matrix": k}, "measures": ["M77"]})
    for k in p103mats:
        mc.append({"graph": {"matrix": k}, "measures": ["M76" if k.startswith("g_") else "M77"]})
    for k in p2bmats:
        mc.append({"graph": {"p2_matrix": k}, "measures": ["M76"]})
    binary = [c["case"] for c in p1cases if c["mode"] == "binary" and c["matrix"] in square]
    weighted = [c["case"] for c in p1cases if c["mode"] == "weighted" and c["matrix"] in square]
    for cid in binary[::32]:
        mc.append({"graph": {"construct": cid}, "measures": ["M76"]})
    for cid in weighted[::32]:
        mc.append({"graph": {"construct": cid}, "measures": ["M77"]})
    for i, c in enumerate(mc):
        c["case"] = f"p2c-measure-{i + 1:04d}"
    json.dump(mc, open(os.path.join(OUT, "cmeasures.json"), "w"))
    print(f"P2-03 battery: {len(mc)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
