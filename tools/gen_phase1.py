"""Generate the synthetic phase-1 inputs (pure Python, fixed seeds).

Writes cases/phase1/{matrices,construct,measures1,prim}.json. Every number is
stored as a 16-hex-digit IEEE-754 binary64 word (big-endian, like num2hex),
arrays column-major. No patient data: everything here is generated.
"""
import json, math, os, random, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "cases/phase1"


def hx(x):
    return struct.pack(">d", float(x)).hex()


def arr(rows, cols, get):
    """Column-major array record from get(i, j) with 0-based indices."""
    return {"size": [rows, cols], "hex": [hx(get(i, j)) for j in range(cols) for i in range(rows)]}


def mat(M):
    n, m = len(M), len(M[0])
    return arr(n, m, lambda i, j: M[i][j])


def vec(v):
    return arr(1, len(v), lambda i, j: v[j])


def corr_like(n, T, k, seed, common=0.0, decimals=None):
    rnd = random.Random(seed)
    L = [[rnd.gauss(0, 1) for _ in range(k)] for _ in range(n)]
    # T samples with shared factor scores per sample
    rnd = random.Random(seed + 1000)
    F = [[rnd.gauss(0, 1) for _ in range(k)] for _ in range(T)]
    C = [rnd.gauss(0, 1) for _ in range(T)]
    S = [[sum(L[i][f] * F[t][f] for f in range(k)) + common * C[t] + rnd.gauss(0, 1.5) for t in range(T)]
         for i in range(n)]
    mu = [sum(s) / T for s in S]
    sd = [math.sqrt(sum((x - mu[i]) ** 2 for x in S[i])) for i in range(n)]
    R = [[0.0] * n for _ in range(n)]
    for i in range(n):
        R[i][i] = 1.0
        for j in range(i + 1, n):
            r = sum((S[i][t] - mu[i]) * (S[j][t] - mu[j]) for t in range(T)) / (sd[i] * sd[j])
            if decimals is not None:
                r = round(r * 10 ** decimals) / 10 ** decimals
            R[i][j] = R[j][i] = r
    return R


def uniform(n, m, seed, lo=-1.0, hi=1.0):
    rnd = random.Random(seed)
    return [[rnd.uniform(lo, hi) for _ in range(m)] for _ in range(n)]


def sym_uniform(n, seed, lo=-1.0, hi=1.0, diag=0.0):
    rnd = random.Random(seed)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = diag
        for j in range(i + 1, n):
            M[i][j] = M[j][i] = rnd.uniform(lo, hi)
    return M


def binary_from_edges(n, edges, directed=False):
    M = [[0.0] * n for _ in range(n)]
    for i, j in edges:
        M[i][j] = 1.0
        if not directed:
            M[j][i] = 1.0
    return M


def main():
    os.makedirs(OUT, exist_ok=True)
    m = {}
    print("building 379-region matrices ...", file=sys.stderr)
    A = corr_like(379, 150, 6, 11)
    m["corrA_379"] = A
    m["corrB_379"] = corr_like(379, 150, 4, 12, common=2.0)
    m["corrT_379"] = corr_like(379, 150, 6, 13, decimals=2)
    m["asym_379"] = uniform(379, 379, 14)
    near = [row[:] for row in A]
    rnd = random.Random(15)
    for i in range(379):
        for j in range(i + 1, 379):
            if rnd.random() < 0.05:
                near[i][j] += rnd.choice((-1, 1)) * 2.0 ** -45  # below 1e-10 asymmetry
    m["near_379"] = near
    # p-value matrix for corrA: symmetric, with exact hits on list values
    rnd = random.Random(16)
    P = [[0.0] * 379 for _ in range(379)]
    hits = [0.05, 0.01, 0.001, 0.0001]
    for i in range(379):
        for j in range(i + 1, 379):
            u = rnd.random()
            p = hits[int(u * 4)] if u < 0.02 else rnd.random() ** 3
            P[i][j] = P[j][i] = p
    m["pA_379"] = P
    # small matrices
    m["s2"] = [[0.0, 0.5], [0.5, 0.0]]
    m["s3"] = sym_uniform(3, 21, diag=1.0)
    m["s5"] = sym_uniform(5, 22, diag=1.0)
    m["s44"] = sym_uniform(44, 23, diag=1.0)
    nz = sym_uniform(44, 24, diag=0.7)
    for i, j in ((0, 3), (5, 9), (10, 40), (2, 2)):
        nz[i][j] = -0.0
        nz[j][i] = -0.0
    m["negzero_44"] = nz
    ties = [[0.0] * 12 for _ in range(12)]
    vals = [0.5, 0.5, 0.25, 0.25, 0.25, -0.5, 0.75, 0.75, 0.1]
    rnd = random.Random(25)
    for i in range(12):
        for j in range(i + 1, 12):
            ties[i][j] = ties[j][i] = rnd.choice(vals)
    m["ties_12"] = ties
    m["zero_5"] = [[0.0] * 5 for _ in range(5)]
    one = [[0.0] * 5 for _ in range(5)]
    one[1][3] = one[3][1] = 0.4
    m["single_edge_5"] = one
    m["asym_12"] = uniform(12, 12, 26)
    m["nonsquare_3x4"] = uniform(3, 4, 27)
    bad = sym_uniform(5, 28)
    bad[1][2] = float("nan")
    m["nan_5"] = bad
    bad = sym_uniform(5, 29)
    bad[0][4] = float("inf")
    m["inf_5"] = bad
    m["p_s5"] = sym_uniform(5, 30, lo=0.0, hi=0.1)

    # hand-built graphs for measures (already valid W: zero diagonal, finite)
    g = {}
    g["empty_6"] = binary_from_edges(6, [])
    g["complete_6"] = binary_from_edges(6, [(i, j) for i in range(6) for j in range(i + 1, 6)])
    g["star_7"] = binary_from_edges(7, [(0, j) for j in range(1, 7)])
    g["path_8"] = binary_from_edges(8, [(i, i + 1) for i in range(7)])
    g["ring_9"] = binary_from_edges(9, [(i, (i + 1) % 9) for i in range(9)])
    g["twocomp_10"] = binary_from_edges(10, [(0, 1), (1, 2), (2, 0), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (5, 9), (6, 8)])
    g["isolated_8"] = binary_from_edges(8, [(0, 1), (1, 2), (2, 3), (0, 2), (4, 5)])
    rnd = random.Random(31)
    g["dag_7"] = binary_from_edges(7, [(i, j) for i in range(7) for j in range(i + 1, 7) if rnd.random() < 0.5], directed=True)
    g["recip_6"] = binary_from_edges(6, [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2), (4, 5)], directed=True)
    g["uppertri_8"] = binary_from_edges(8, [(i, j) for i in range(8) for j in range(i + 1, 8) if (i + j) % 3 == 0], directed=True)
    g["regular_8"] = binary_from_edges(8, [(i, (i + d) % 8) for i in range(8) for d in (1, 2)])
    g["degties_9"] = binary_from_edges(9, [(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6), (2, 3), (5, 6)])
    rnd = random.Random(32)
    g["rand_und_12"] = binary_from_edges(12, [(i, j) for i in range(12) for j in range(i + 1, 12) if rnd.random() < 0.3])
    g["rand_dir_12"] = binary_from_edges(12, [(i, j) for i in range(12) for j in range(12) if i != j and rnd.random() < 0.3], directed=True)
    clique, tail = 120, 150
    edges = [(i, j) for i in range(clique) for j in range(i + 1, clique)]
    edges += [(clique - 1 + t, clique + t) for t in range(tail)]
    g["lollipop_270"] = binary_from_edges(clique + tail, edges)
    w = {}
    w["signed_10"] = sym_uniform(10, 33)
    pos = uniform(10, 10, 34, lo=0.0, hi=1.0)
    for i in range(10):
        pos[i][i] = 0.0
    w["posasym_10"] = pos
    w["onlyneg_6"] = sym_uniform(6, 35, lo=-1.0, hi=-0.1)
    w["onlypos_6"] = sym_uniform(6, 36, lo=0.1, hi=1.0)

    json.dump({k: mat(v) for k, v in {**m, **{"g_" + k: v for k, v in g.items()}, **{"w_" + k: v for k, v in w.items()}}.items()},
              open(os.path.join(OUT, "matrices.json"), "w"))

    # construction cases
    prop = [f"0.{k:02d}" if k < 100 else "1" for k in range(10, 51)]
    absl = ["0.05", "0.1", "0.2", "0.25", "0.3", "0.5", "0.75", "1"]
    sig = ["0.05", "0.01", "0.005", "0.001", "0.0001", "0.00000001"]
    rules = [("binary", "keep"), ("weighted", "keep"), ("weighted", "absolute"), ("weighted", "zero")]
    cases = []

    def add(mid, typ, val, mode, rule, mask=None, pid=None):
        cases.append({"case": f"p1-construct-{len(cases) + 1:04d}", "matrix": mid, "type": typ,
                      "value": val, "mode": mode, "neg_rule": rule, "mask": mask, "pvalues": pid})

    for mid in ("corrA_379", "corrB_379", "corrT_379", "asym_379", "near_379"):
        for v in prop:
            for mode, rule in rules:
                add(mid, "proportional", v, mode, rule)
    for mid in ("corrA_379", "asym_379"):
        for v in absl:
            for mode, rule in rules:
                add(mid, "absolute", v, mode, rule)
        for mode, rule in rules:
            add(mid, "none", None, mode, rule)
    for v in sig:
        for mode, rule in rules:
            add("corrA_379", "significance", v, mode, rule, pid="pA_379")
    for mid in ("s2", "s3", "s5", "s44", "negzero_44", "ties_12", "zero_5", "single_edge_5", "asym_12"):
        for v in ("0.1", "0.25", "0.5", "1"):
            for mode, rule in rules:
                add(mid, "proportional", v, mode, rule)
        for mode, rule in rules:
            add(mid, "absolute", "0.3", mode, rule)
            add(mid, "none", None, mode, rule)
    for mode, rule in rules:
        add("s5", "significance", "0.05", mode, rule, pid="p_s5")
    rnd = random.Random(40)
    masks = [[True] * 379, [i % 7 != 3 for i in range(379)], [i in (10, 200) for i in range(379)],
             [rnd.random() < 0.8 for _ in range(379)]]
    for mk in masks:
        for mode, rule in rules[:2]:
            add("corrA_379", "proportional", "0.2", mode, rule, mask=mk)
    # error cases
    add("nonsquare_3x4", "proportional", "0.2", "binary", "keep")
    add("nan_5", "proportional", "0.2", "binary", "keep")
    add("inf_5", "absolute", "0.3", "weighted", "keep")
    add("s5", "significance", "0.05", "binary", "keep", pid=None)
    add("s5", "proportional", "0.2", "binary", "keep", mask=[True, False, False, False, False])
    json.dump(cases, open(os.path.join(OUT, "construct.json"), "w"))

    # measure cases: graphs from construction plus the hand-built graphs
    bin_measures = [f"M{k:02d}" for k in range(1, 34)]
    wei_measures = [f"M{k:02d}" for k in range(34, 43)]
    mcases = []
    by_key = {(c["matrix"], c["type"], c["value"], c["mode"], c["neg_rule"], json.dumps(c["mask"])): c["case"] for c in cases}
    for mid in ("corrA_379", "corrB_379", "corrT_379", "asym_379", "near_379"):
        for v in ("0.10", "0.20", "0.30", "0.40", "0.50"):
            mcases.append({"graph": {"construct": by_key[(mid, "proportional", v, "binary", "keep", "null")]}, "measures": bin_measures})
            for rule in ("keep", "absolute", "zero"):
                mcases.append({"graph": {"construct": by_key[(mid, "proportional", v, "weighted", rule, "null")]}, "measures": wei_measures})
    for mid in ("s5", "s44", "negzero_44", "ties_12", "zero_5", "single_edge_5", "asym_12"):
        mcases.append({"graph": {"construct": by_key[(mid, "proportional", "0.5", "binary", "keep", "null")]}, "measures": bin_measures})
        mcases.append({"graph": {"construct": by_key[(mid, "none", None, "weighted", "keep", "null")]}, "measures": wei_measures})
    for k in g:
        mcases.append({"graph": {"matrix": "g_" + k}, "measures": bin_measures})
    for k in w:
        mcases.append({"graph": {"matrix": "w_" + k}, "measures": wei_measures})
    for i, c in enumerate(mcases):
        c["case"] = f"p1-measure-{i + 1:04d}"
    json.dump(mcases, open(os.path.join(OUT, "measures1.json"), "w"))

    # primitive batteries
    rnd = random.Random(50)
    prim = {"sum1d": [], "matsum": [], "mean": [], "maxmin": [], "square": None, "round": None, "sortrows": []}
    for L in (0, 1, 2, 3, 43, 44, 45, 100, 379, 1025, 1026, 2048, 4097, 143641):
        prim["sum1d"].append(vec([rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-20, 20) for _ in range(L)]) if L else {"size": [1, 0], "hex": []})
    for L in (1, 2, 5, 44, 100):
        prim["sum1d"].append(vec([-0.0] * L))
        prim["sum1d"].append(vec([(-0.0 if (i % 3) else 0.0) for i in range(L)]))
        prim["sum1d"].append(vec([(-0.0 if i else 0.0) for i in range(L)]))
    for (r, c) in ((3, 3), (44, 44), (45, 7), (7, 45), (379, 379), (100, 1), (1, 100)):
        M = [[rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-10, 10) for _ in range(c)] for _ in range(r)]
        prim["matsum"].append(mat(M))
    Z = [[(-0.0 if (i + j) % 2 else 0.0) if j % 3 == 0 else (-0.0) for j in range(9)] for i in range(50)]
    prim["matsum"].append(mat(Z))
    for L in (1, 2, 43, 44, 45, 379, 1025):
        prim["mean"].append(vec([rnd.uniform(0, 1) for _ in range(L)]))
    prim["maxmin"] = [vec([-0.0, 0.0]), vec([0.0, -0.0]), vec([-0.0, -0.0, 0.0]), vec([1.0, float("nan"), 1.0]),
                      vec([float("nan")] * 3), mat([[-0.0, 0.0, float("nan")], [float("nan")] * 3, [2.0, 2.0, 1.0]])]
    prim["square"] = vec([rnd.uniform(-1e3, 1e3) for _ in range(2000)] + [rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-500, 500) for _ in range(2000)])
    prim["round"] = vec([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4999999999999996, 4503599627370495.5, 10744.65, 21489.3, 0.49999999999999994])
    prim["sortrows"].append(mat([[float(i + 1), v] for i, v in enumerate([0.5, 0.25, 0.5, 0.75, 0.25, -0.0, 0.0, 0.5])]))
    json.dump(prim, open(os.path.join(OUT, "prim.json"), "w"))
    print(f"{len(cases)} construction cases, {len(mcases)} measure cases", file=sys.stderr)


if __name__ == "__main__":
    main()
