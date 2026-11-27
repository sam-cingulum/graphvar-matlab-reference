"""Inputs for null-model battery 5 (drivers/ref_nm5.m): Louvain and community detection. Writes <out>/nm5.json.
Synthetic only (seeded).

  graphs  [{"name", "dir", "sign", "n", "W"}]: W column-major as 16-hex binary64 words, zero diagonal. "dir" = asymmetric,
          "sign" = has negative weights.
  cl      [{"case", "graph", "gamma", "seed"}]: community_louvain(W, gamma) after rng(seed, 'twister'), BCT 2019 first on the
          path, plus a line-for-line traced copy from the same seed (every pass's node order, moves and every level's B and Q).
  clgv    [{"case", "graph", "seed"}]: GraphVar 2.03a's own ext/BCT community_louvain(W) (default objective negative_asym).
  old     [{"case", "graph", "seed"}]: GraphVar 2.03a's ext/BCT modularity_louvain_und(W) (the menu's "Louvain" item).
  items   [{"case", "fn", "graph", "seed", "order"}]: a GraphVar menu helper fn(W) after rng(seed); order "gv" = GraphVar's
          ext/BCT first on the path (what GraphVar runs), "bct" = BCT 2019 first.
  prim    [{"case", "kind", "n", "seed"}]: kind "trace": X = rand(n) - 0.5 scaled as in gen_nm4 sum3, trace(X);
          kind "lsum": the same X and L = rand(n) < 0.5, sum(X(L)).
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def und(n, val):
    W = [0.0] * (n * n)
    for j in range(n):
        for i in range(j + 1, n):
            W[i + j * n] = W[j + i * n] = val(i, j)
    return W


def dirg(n, val):
    W = [0.0] * (n * n)
    for j in range(n):
        for i in range(n):
            if i != j:
                W[i + j * n] = val(i, j)
    return W


def main(out):
    os.makedirs(out, exist_ok=True)
    R = random.Random(5505)
    doc = {"graphs": [], "cl": [], "clgv": [], "old": [], "items": [], "prim": []}
    seed = lambda: R.randrange(1, 2**32)
    wv = lambda: R.randrange(1, 2**20) / 2**20
    sv = lambda: R.randrange(-2**20, 2**20) / 2**20

    def add(name, directed, n, W):
        assert all(W[i + i * n] == 0 for i in range(n)), name
        sym = all(W[i + j * n] == W[j + i * n] for i in range(n) for j in range(n))
        assert sym != directed, name
        doc["graphs"].append({"name": name, "dir": directed, "sign": any(x < 0 for x in W), "n": n,
                              "W": [hx(x) for x in W]})

    for n, p in [(6, 0.5), (10, 0.3), (10, 0.6), (20, 0.2), (40, 0.15), (60, 0.1), (100, 0.06)]:
        add(f"und-bin-n{n}-p{p}", False, n, und(n, lambda i, j: 1.0 if R.random() < p else 0.0))
        add(f"und-wei-n{n}-p{p}", False, n, und(n, lambda i, j: wv() if R.random() < p else 0.0))
    # planted communities: 4 blocks of 8, dense inside, sparse between
    add("und-blocks-n32", False, 32, und(32, lambda i, j: (wv() if R.random() < (0.7 if i // 8 == j // 8 else 0.05) else 0.0)))
    add("und-bblocks-n32", False, 32, und(32, lambda i, j: (1.0 if R.random() < (0.7 if i // 8 == j // 8 else 0.05) else 0.0)))
    add("und-ring-n20", False, 20, und(20, lambda i, j: 1.0 if (i - j) % 20 in (1, 19) else 0.0))
    add("und-full-n8", False, 8, und(8, lambda i, j: 1.0))
    add("und-twocomp-n16", False, 16, und(16, lambda i, j: 0.0 if (i < 8) != (j < 8) else (wv() if R.random() < 0.5 else 0.0)))
    add("und-isolated-n12", False, 12, und(12, lambda i, j: 0.0 if j == 0 else (1.0 if R.random() < 0.5 else 0.0)))
    add("und-wide-n20", False, 20, und(20, lambda i, j: (2.0 ** R.randrange(-30, 30) * wv()) if R.random() < 0.4 else 0.0))
    add("dir-wei-n20-p0.25", True, 20, dirg(20, lambda i, j: wv() if R.random() < 0.25 else 0.0))
    add("dir-bin-n15-p0.3", True, 15, dirg(15, lambda i, j: 1.0 if R.random() < 0.3 else 0.0))
    for n, p in [(10, 0.6), (20, 0.4), (40, 0.3)]:
        add(f"und-sgn-n{n}-p{p}", False, n, und(n, lambda i, j: sv() if R.random() < p else 0.0))
    add("und-sgnblocks-n24", False, 24,
        und(24, lambda i, j: (wv() if i // 8 == j // 8 else -wv()) if R.random() < 0.5 else 0.0))

    nonneg = [g["name"] for g in doc["graphs"] if not g["sign"]]
    signed = [g["name"] for g in doc["graphs"] if g["sign"]]
    for g in nonneg:
        for k in range(2):
            doc["cl"].append({"case": f"nm5-cl-{g}-s{k}", "graph": g, "gamma": 1.0, "seed": seed()})
    for g in ["und-wei-n20-p0.2", "und-blocks-n32", "und-bin-n40-p0.15", "dir-wei-n20-p0.25"]:
        for gam in (0.5, 2.0):
            doc["cl"].append({"case": f"nm5-cl-{g}-g{gam}", "graph": g, "gamma": gam, "seed": seed()})
    for g in signed + ["und-wei-n20-p0.2", "und-blocks-n32", "und-bin-n10-p0.3"]:
        doc["clgv"].append({"case": f"nm5-clgv-{g}", "graph": g, "seed": seed()})
    for g in ["und-wei-n10-p0.3", "und-bin-n20-p0.2", "und-wei-n40-p0.15", "und-blocks-n32", "und-bblocks-n32",
              "und-ring-n20", "dir-wei-n20-p0.25"]:
        doc["old"].append({"case": f"nm5-old-{g}", "graph": g, "seed": seed()})
    for fn, gs in [("modularity_louvain_QOut_und", ["und-wei-n10-p0.3", "und-blocks-n32", "und-sgn-n10-p0.6"]),
                   ("modularity_louvain_und", ["und-wei-n10-p0.3", "und-blocks-n32"]),
                   ("hpos_Louvain", ["und-sgn-n10-p0.6"]), ("hneg_Louvain", ["und-sgn-n10-p0.6"]),
                   ("modularity_louvain_cOut_und", ["und-wei-n10-p0.3"]),
                   ("modularity_cOut_und", ["und-wei-n10-p0.3", "und-sgn-n10-p0.6"]),
                   ("hpos_Newman", ["und-sgn-n10-p0.6", "und-sgnblocks-n24"]),
                   ("hneg_Newman", ["und-sgn-n10-p0.6"]),
                   ("modularity_QOut_und", ["und-wei-n10-p0.3"])]:
        for g in gs:
            for order in ("gv", "bct"):
                doc["items"].append({"case": f"nm5-item-{fn}-{g}-{order}", "fn": fn, "graph": g, "seed": seed(),
                                     "order": order})
    for kind in ("trace", "lsum"):
        for n in (1, 2, 5, 43, 44, 45, 64, 100, 211, 300):
            doc["prim"].append({"case": f"nm5-prim-{kind}-n{n}", "kind": kind, "n": n, "seed": seed()})
    json.dump(doc, open(os.path.join(out, "nm5.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1])
