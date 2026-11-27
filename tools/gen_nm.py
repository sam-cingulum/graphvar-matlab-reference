"""Inputs for the null-model battery (drivers/ref_nm.m). Writes <out>/nm.json.

  graphs  [{"name", "dir", "n", "W"}]: W column-major as 16-hex binary64 words. Undirected graphs are exactly
          symmetric; every graph has a zero diagonal and at least two vertex-disjoint edges (so BCT's edge-pair
          search always ends).
  calls   [{"case", "fn", "graph", "args", "seed"}]: the driver runs rng(seed, 'twister') and then the call;
          args are the arguments after W (numbers, or strings for small_world_propensity's option).
  rng     {"randperm_k": [{"n", "k", "seed"}], "randperm": [{"n", "seed"}], "randi": [{"lo", "hi", "count", "seed"}],
           "rand": [{"seed", "count"}]}: MATLAB random-number primitives.
Synthetic only (seeded). Needs no other input.
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def disjoint_pair(n, W, directed):
    E = [(i, j) for j in range(n) for i in range(n) if W[i + j * n] != 0 and (directed or i > j)]
    for a in range(len(E)):
        for b in range(a + 1, len(E)):
            if len({E[a][0], E[a][1], E[b][0], E[b][1]}) == 4:
                return True
    return False


def und(n, val):
    """val(i, j) for i > j, called column by column (j outer, i inner)"""
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
    G = []
    R = random.Random(7700)

    def add(name, directed, n, W):
        assert disjoint_pair(n, W, directed), name
        if not directed:
            assert all(W[i + j * n] == W[j + i * n] for i in range(n) for j in range(n)), name
        G.append({"name": name, "dir": directed, "n": n, "W": [hx(x) for x in W]})

    def bern(p, f):
        return lambda i, j: (f() if R.random() < p else 0.0)

    for n in (10, 30, 60):
        for p in (0.2, 0.5):
            add(f"und-bin-n{n}-p{p}", False, n, und(n, bern(p, lambda: 1.0)))
            add(f"und-wei-n{n}-p{p}", False, n, und(n, bern(p, lambda: 1.0 - R.random())))
    for n in (10, 30):
        for p in (0.5, 0.9):
            add(f"und-sig-n{n}-p{p}", False, n, und(n, bern(p, lambda: R.gauss(0.0, 1.0))))
    add("und-bin-n20-p0.8", False, 20, und(20, bern(0.8, lambda: 1.0)))
    add("und-full-n8", False, 8, und(8, lambda i, j: 1.0))
    add("und-wfull-n8", False, 8, und(8, lambda i, j: 1.0 - R.random()))
    # node 0 linked to every other node, the rest sparse
    add("und-fullnode-n20", False, 20, und(20, lambda i, j: 1.0 if j == 0 else (1.0 if R.random() < 0.2 else 0.0)))
    # two components of 10 nodes each
    add("und-twocomp-n20", False, 20, und(20, lambda i, j: 0.0 if (i < 10) != (j < 10) else (1.0 if R.random() < 0.4 else 0.0)))
    # ring lattice, each node linked to its 2 nearest neighbours on each side
    add("und-ring-n30", False, 30, und(30, lambda i, j: 1.0 if min((i - j) % 30, (j - i) % 30) <= 2 else 0.0))
    # weighted ring with distinct weights
    add("und-wring-n30", False, 30, und(30, lambda i, j: (1.0 - R.random()) if min((i - j) % 30, (j - i) % 30) <= 2 else 0.0))
    for n in (10, 30):
        for p in (0.2, 0.5):
            add(f"dir-bin-n{n}-p{p}", True, n, dirg(n, bern(p, lambda: 1.0)))
            add(f"dir-wei-n{n}-p{p}", True, n, dirg(n, bern(p, lambda: 1.0 - R.random())))
        add(f"dir-sig-n{n}-p0.6", True, n, dirg(n, bern(0.6, lambda: R.gauss(0.0, 1.0))))

    calls = []
    S = random.Random(7701)

    def call(fn, g, args, reps=1):
        for _ in range(reps):
            calls.append({"case": f"nm-{len(calls):04d}-{fn}-{g}", "fn": fn, "graph": g, "args": args,
                          "seed": S.randrange(1, 1 << 31)})

    for g in G:
        nm = g["name"]
        if not g["dir"]:
            if "-sig-" not in nm:
                call("randmio_und", nm, [1], 2)
                call("randmio_und", nm, [5])
                call("randmio_und_connected", nm, [1], 2)
                call("randmio_und_connected", nm, [5])
                call("randomizer_bin_und", nm, [0.5], 2)
                call("randomizer_bin_und", nm, [5])
                call("small_world_propensity", nm, [], 1)
                call("small_world_propensity", nm, ["bin"], 1)
                call("smallworldness", nm, [3, 1], 1)
            call("randmio_und_signed", nm, [1], 2)
            call("randmio_und_signed", nm, [5])
            call("null_model_und_sign", nm, [5], 2)
            call("null_model_und_sign", nm, [5, 1])
            call("null_model_und_sign", nm, [1, 0.5])
        else:
            if "-sig-" not in nm:
                call("randmio_dir", nm, [1], 2)
                call("randmio_dir", nm, [5])
                call("randmio_dir_connected", nm, [1], 2)
                call("randmio_dir_connected", nm, [5])
            call("randmio_dir_signed", nm, [1], 2)
            call("null_model_dir_sign", nm, [5], 2)
            call("null_model_dir_sign", nm, [5, 1])
    # expected errors: signed input to randomizer_bin_und, a 3-node graph for randmio_und_signed (randperm(3, 4))
    call("randomizer_bin_und", "und-sig-n10-p0.5", [1])
    G.append({"name": "und-tri-n3", "dir": False, "n": 3, "W": [hx(x) for x in und(3, lambda i, j: 1.0)]})
    call("randmio_und_signed", "und-tri-n3", [1])

    rp = []
    for seed in (1, 2024):
        for n in list(range(1, 13)) + [16, 20, 31, 32, 33, 50, 64, 100, 379, 1000, 4096, 65536, 1000000]:
            ks = {1, 2, 3, 4, 5, 6, 8, 10, n // 4, n // 2, n - 1, n}
            for k in sorted(k for k in ks if 1 <= k <= n and k <= 4096):
                rp.append({"n": n, "k": k, "seed": seed})
    rpf = [{"n": n, "seed": s} for s in (1, 2024) for n in (1, 2, 3, 10, 100, 1000, 4005, 71631)]
    ri = []
    for seed in (1, 2024):
        for lo, hi in ((1, 1), (1, 2), (1, 3), (1, 5), (1, 7), (1, 10), (1, 100), (1, 379), (1, 1000),
                       (0, 9), (5, 12), (1, 2 ** 31 - 1), (1, 2 ** 32), (1, 2 ** 40), (-3, 3)):
            ri.append({"lo": lo, "hi": hi, "count": 6, "seed": seed})
    doc = {"graphs": G, "calls": calls,
           "rng": {"randperm_k": rp, "randperm": rpf, "randi": ri, "rand": [{"seed": s, "count": 16} for s in (0, 1, 5489, 2024)]}}
    json.dump(doc, open(os.path.join(out, "nm.json"), "w"))
    print(f"nm: {len(G)} graphs, {len(calls)} calls, {len(rp)} randperm_k, {len(rpf)} randperm, {len(ri)} randi",
          file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
