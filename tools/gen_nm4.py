"""Inputs for null-model battery 4 (drivers/ref_nm4.m). Writes <out>/nm4.json. Synthetic only (seeded).

  sum3     [{"case", "a", "b", "c", "seed", "kind"}]: the driver builds M (a x b x c) from rng(seed, 'twister'):
           U1 = rand(a, b, c); U2 = rand(a, b, c); M = (U1 - 0.5) .* 2.^(floor(64 * U2) - 32), every step exact.
           kind "negz" then sets M(1, 1, :) = -0. Records sum(M, 3) and mean(M, 3) (GraphVar's normalisation).
  graphs   [{"name", "dir", "n", "W"}] as in nm.json (W column-major, 16-hex binary64 words, zero diagonal).
  sw       [{"case", "graph", "src", "nr", "iter", "seed"}]: rng(seed); RW = nr calls of BCT 2019
           randmio_und(W, iter); then smallworldness three ways. src "nm" = a graph of nm.json, "nm4" = of this file.
  swpa     [{"case", "graph", "args", "seed"}]: small_world_propensity (option b) on asymmetric graphs.
  shuffle  [{"case", "n", "R", "seed"}]: GraphVar's random_shuffle(R, 1) after rng(seed); R column-major hex.
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def disjoint_pair(n, W):
    E = [(i, j) for j in range(n) for i in range(n) if W[i + j * n] != 0 and i > j]
    for a in range(len(E)):
        for b in range(a + 1, len(E)):
            if len({E[a][0], E[a][1], E[b][0], E[b][1]}) == 4:
                return True
    return False


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
    R = random.Random(4404)
    doc = {"sum3": [], "graphs": [], "sw": [], "swpa": [], "shuffle": []}
    seed = lambda: R.randrange(1, 2**32)

    # sum(M, 3) / mean(M, 3)
    C = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 43, 44, 45, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 1000]
    for a, b in [(1, 1), (7, 1), (1, 7), (3, 3), (10, 10), (44, 1), (1, 44), (45, 45)]:
        for c in C:
            doc["sum3"].append({"case": f"nm4-sum3-{a}x{b}x{c}", "a": a, "b": b, "c": c, "seed": seed(), "kind": "gen"})
    for c in [2, 3, 44, 45, 100, 257, 1000]:
        doc["sum3"].append({"case": f"nm4-sum3-100x100x{c}", "a": 100, "b": 100, "c": c, "seed": seed(), "kind": "gen"})
    for c in [2, 17]:
        doc["sum3"].append({"case": f"nm4-sum3-200x200x{c}", "a": 200, "b": 200, "c": c, "seed": seed(), "kind": "gen"})
    for a, b in [(1, 1), (5, 5)]:
        for c in [1, 2, 3, 44, 45, 100]:
            doc["sum3"].append({"case": f"nm4-sum3-negz-{a}x{b}x{c}", "a": a, "b": b, "c": c, "seed": seed(), "kind": "negz"})

    # graphs for small-worldness (disconnected ones) and asymmetric graphs for small-world propensity
    def add(name, directed, n, W, need_pair=True):
        if need_pair:
            assert disjoint_pair(n, W), name
        if not directed:
            assert all(W[i + j * n] == W[j + i * n] for i in range(n) for j in range(n)), name
        else:
            assert any(W[i + j * n] != W[j + i * n] for i in range(n) for j in range(n)), name
        doc["graphs"].append({"name": name, "dir": directed, "n": n, "W": [hx(x) for x in W]})

    wv = lambda: R.randrange(1, 2**20) / 2**20
    add("und-wtwocomp-n20", False, 20,
        und(20, lambda i, j: 0.0 if (i < 10) != (j < 10) else (wv() if R.random() < 0.4 else 0.0)))
    add("und-isolated-n12", False, 12, und(12, lambda i, j: 0.0 if j == 0 else (1.0 if R.random() < 0.5 else 0.0)))
    add("und-wisolated-n12", False, 12, und(12, lambda i, j: 0.0 if j == 0 else (wv() if R.random() < 0.5 else 0.0)))
    add("und-bin-sparse-n30", False, 30, und(30, lambda i, j: 1.0 if R.random() < 0.07 else 0.0))
    add("und-wei-sparse-n30", False, 30, und(30, lambda i, j: wv() if R.random() < 0.07 else 0.0))
    for n, p in [(10, 0.3), (20, 0.2), (30, 0.3)]:
        add(f"dir-bin-n{n}-p{p}", True, n, dirg(n, lambda i, j: 1.0 if R.random() < p else 0.0), False)
        add(f"dir-wei-n{n}-p{p}", True, n, dirg(n, lambda i, j: wv() if R.random() < p else 0.0), False)
    add("dir-oneway-n12", True, 12, dirg(12, lambda i, j: (wv() if R.random() < 0.6 else 0.0) if i < j else 0.0), False)

    # small-worldness: battery-1 graphs (same calls as nm-*-smallworldness) and the new disconnected ones
    old = ["und-bin-n10-p0.2", "und-wei-n10-p0.2", "und-bin-n10-p0.5", "und-wei-n10-p0.5", "und-bin-n30-p0.2",
           "und-wei-n30-p0.2", "und-bin-n30-p0.5", "und-wei-n30-p0.5", "und-twocomp-n20", "und-ring-n30", "und-wring-n30"]
    for g in old:
        doc["sw"].append({"case": f"nm4-sw-{g}", "graph": g, "src": "nm", "nr": 3, "iter": 1, "seed": seed()})
    for g in ["und-wtwocomp-n20", "und-isolated-n12", "und-wisolated-n12", "und-bin-sparse-n30", "und-wei-sparse-n30"]:
        doc["sw"].append({"case": f"nm4-sw-{g}", "graph": g, "src": "nm4", "nr": 3, "iter": 1, "seed": seed()})
    for g, nr, it in [("und-bin-n10-p0.5", 50, 1), ("und-wei-n10-p0.5", 50, 1), ("und-twocomp-n20", 5, 2),
                      ("und-wtwocomp-n20", 45, 1)]:
        doc["sw"].append({"case": f"nm4-sw-{g}-nr{nr}-it{it}", "graph": g, "src": "nm" if "wtwo" not in g else "nm4",
                          "nr": nr, "iter": it, "seed": seed()})

    # small-world propensity on asymmetric input (symm_matrix path), option b
    for g in doc["graphs"]:
        if g["dir"]:
            for args in ([], ["bin"]):
                doc["swpa"].append({"case": f"nm4-swpa-{g['name']}{'-bin' if args else ''}", "graph": g["name"],
                                    "args": args, "seed": seed()})

    # random_shuffle (raw-matrix menu)
    for n, sym in [(2, True), (3, True), (5, True), (10, True), (30, True), (6, False)]:
        M = [0.0] * (n * n)
        for j in range(n):
            for i in range(n):
                if i == j:
                    M[i + j * n] = 1.0 if sym else 0.5
                elif not sym or i > j:
                    v = R.randrange(-2**20, 2**20) / 2**20
                    M[i + j * n] = v
                    if sym:
                        M[j + i * n] = v
        doc["shuffle"].append({"case": f"nm4-shuffle-n{n}{'' if sym else '-asym'}", "n": n,
                               "R": [hx(x) for x in M], "seed": seed()})

    json.dump(doc, open(os.path.join(out, "nm4.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1])
