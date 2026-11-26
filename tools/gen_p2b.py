#!/usr/bin/env python3
"""Phase-2b inputs: one-operand Gram products X.'*X and X*X.', distinct-operand
X.'*Y, and short inner products, for checking the P2-00 MM5/MM6/MM7 rules.

- square: Gaussian A, B (n x n) for n in SQ_N.
- rect:   Gaussian X, Y (r x w) for r in RECT_R, w in RECT_W (short K, r < 256 mostly).
- inner:  Gaussian column vectors a, b of every length 1..400.
Python's random with fixed seeds; every value is an exact binary64 word.

It also writes the P2-02 binary betweenness battery (M75): bmeasures.json
(cases) and matrices.json (new hand-built graphs), reading cases/phase1 and
cases/p103 (run tools/gen_p103.py first).
"""
import json, os, random, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "cases/p2b"
SQ_N = sorted(set([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 23, 24, 25, 31, 32, 33, 44, 47, 48, 49,
                   63, 64, 65, 90, 96, 116, 128, 129, 192, 200, 256, 257, 379]))
RECT_R = [3, 8, 16, 31, 50, 64, 65, 100, 128, 200, 255, 256, 300]
RECT_W = [2, 5, 12, 13, 40, 100]
INNER_MAX = 400


def hx(x):
    return struct.pack(">d", float(x)).hex()


def mat(rows):
    r, c = len(rows), len(rows[0])
    return {"size": [r, c], "hex": [hx(rows[i][j]) for j in range(c) for i in range(r)]}


def gauss(rnd, r, c):
    return [[rnd.gauss(0, 1) for _ in range(c)] for _ in range(r)]


def layered(widths, rnd=None, p=1.0, directed=False):
    n = sum(widths); W = [[0.0] * n for _ in range(n)]; start = 0
    for a, b in zip(widths, widths[1:]):
        for i in range(start, start + a):
            for j in range(start + a, start + a + b):
                if rnd is None or rnd.random() < p:
                    W[i][j] = 1.0
                    if not directed: W[j][i] = 1.0
        start += a
    return W


def rand_graph(rnd, n, p, directed):
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (directed or j > i) and rnd.random() < p:
                W[i][j] = 1.0
                if not directed: W[j][i] = 1.0
    return W


def b_battery():
    base = os.path.dirname(OUT.rstrip("/")) or "."
    p1cases = json.load(open(os.path.join(base, "phase1", "construct.json")))
    p1mats = json.load(open(os.path.join(base, "phase1", "matrices.json")))
    p103mats = json.load(open(os.path.join(base, "p103", "matrices.json")))
    rnd = random.Random(202611274)
    m = {}
    m["b_n1"] = [[0.0]]
    m["b_edge_2"] = [[0.0, 1.0], [1.0, 0.0]]
    m["b_diredge_2"] = [[0.0, 1.0], [0.0, 0.0]]
    m["b_complete_5"] = [[0.0 if i == j else 1.0 for j in range(5)] for i in range(5)]
    m["b_dircycle_5"] = [[1.0 if j == (i + 1) % 5 else 0.0 for j in range(5)] for i in range(5)]
    m["b_layer3_363"] = layered([3] * 121)
    m["b_layer19_361"] = layered([19] * 19)
    m["b_layer3dir_150"] = layered([3] * 50, directed=True)
    widths = [1] + [rnd.randint(1, 6) for _ in range(85)]
    m["b_layermix"] = layered(widths, rnd, 0.7)
    m["b_layermixdir"] = layered(widths, rnd, 0.7, directed=True)
    m["b_rand_und_379"] = rand_graph(rnd, 379, 0.03, False)
    m["b_rand_dir_200"] = rand_graph(rnd, 200, 0.02, True)
    m["b_rand_und_116"] = rand_graph(rnd, 116, 0.1, False)
    for k, W in m.items():
        assert len(W) <= 379, k
    mc = []
    for k in p1mats:
        if k.startswith("g_"):
            mc.append({"graph": {"phase1_matrix": k}, "measures": ["M75"]})
    for k in p103mats:
        if k.startswith("g_"):
            mc.append({"graph": {"matrix": k}, "measures": ["M75"]})
    square = {k for k, v in p1mats.items() if v["size"][0] == v["size"][1]}
    binary = [c["case"] for c in p1cases if c["mode"] == "binary" and c["matrix"] in square
              and c["matrix"] not in ("nan_5", "inf_5")]
    for cid in binary[::8]:
        mc.append({"graph": {"construct": cid}, "measures": ["M75"]})
    for k in m:
        mc.append({"graph": {"p2_matrix": k}, "measures": ["M75"]})
    for i, c in enumerate(mc):
        c["case"] = f"p2b-measure-{i + 1:04d}"
    json.dump({k: mat(v) for k, v in m.items()}, open(os.path.join(OUT, "matrices.json"), "w"))
    json.dump(mc, open(os.path.join(OUT, "bmeasures.json"), "w"))
    print(f"P2-02 battery: {len(m)} matrices, {len(mc)} M75 cases")


def main():
    os.makedirs(OUT, exist_ok=True)
    b_battery()
    rnd = random.Random(202611273)
    out = {"square": [], "rect": [], "inner": []}
    # P1-00 PR8a: where one-thread sum switches from four lanes (<= 81,920 measured) to
    # 4,096-blocks (>= 90,000 measured; run 45830964237 narrowed it to 88,805..88,999, now swept). Squares n^2 for n = 287..299 are GraphVar sizes.
    out["sums"] = sorted(set([81921, 81924, 81925, 82000, 82944, 83000, 83968, 84000, 84992, 85000, 86000, 86016,
                              87000, 87040, 88000, 88064, 89000, 89088, 89999, 90000, 90112]
                             + [n * n for n in range(287, 300)] + list(range(88805, 89000))))
    for n in SQ_N:
        out["square"].append({"n": n, "A": mat(gauss(rnd, n, n)), "B": mat(gauss(rnd, n, n))})
    for r in RECT_R:
        for w in RECT_W:
            out["rect"].append({"r": r, "w": w, "X": mat(gauss(rnd, r, w)), "Y": mat(gauss(rnd, r, w))})
    for L in range(1, INNER_MAX + 1):
        out["inner"].append({"L": L, "a": mat(gauss(rnd, L, 1)), "b": mat(gauss(rnd, L, 1))})
    with open(os.path.join(OUT, "gram.json"), "w") as f:
        json.dump(out, f)
    print(f"square {len(SQ_N)}; rect {len(RECT_R) * len(RECT_W)}; inner {INNER_MAX}")


if __name__ == "__main__":
    main()
