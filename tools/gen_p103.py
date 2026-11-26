"""Generate the synthetic P1-03 inputs (pure Python, fixed seeds).

Writes cases/p103/{matrices,measures,prim}.json. Graphs built by phase-1
construction are referenced by their phase-1 case id; hand-built graphs live
in cases/p103/matrices.json. Every number is a 16-hex-digit binary64 word
(big-endian, like num2hex), arrays column-major. No patient data.
"""
import json, os, random, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "cases/p103"
PHASE1 = os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", "phase1")

W_ITEMS = [f"M{k}" for k in range(43, 62)]
B_ITEMS = ["M33", "M62", "M63", "M64"]


def hx(x):
    return struct.pack(">d", float(x)).hex()


def arr(rows, cols, get):
    return {"size": [rows, cols], "hex": [hx(get(i, j)) for j in range(cols) for i in range(rows)]}


def mat(M):
    return arr(len(M), len(M[0]), lambda i, j: M[i][j])


def vec(v):
    return arr(1, len(v), lambda i, j: v[j])


def zeros(n):
    return [[0.0] * n for _ in range(n)]


def sym(n, edges):
    M = zeros(n)
    for i, j, w in edges:
        M[i][j] = M[j][i] = float(w)
    return M


def main():
    os.makedirs(OUT, exist_ok=True)
    m = {}
    # ties: grid and rings with equal weights
    grid = []
    for r in range(3):
        for c in range(3):
            k = 3 * r + c
            if c < 2: grid.append((k, k + 1, 0.5))
            if r < 2: grid.append((k, k + 3, 0.5))
    m["w_lattice_9"] = sym(9, grid)
    m["w_ring_8"] = sym(8, [(i, (i + 1) % 8, 1.0) for i in range(8)])
    m["w_ring_frac_7"] = sym(7, [(i, (i + 1) % 7, 0.1) for i in range(7)])
    # exact tie 0.25 + 0.5 = 0.75 and near tie 0.1 + 0.2 vs 0.3
    m["w_tie_exact_4"] = sym(4, [(0, 1, 4.0), (1, 3, 2.0), (0, 3, 4.0 / 3.0), (0, 2, 1.0), (2, 3, 1.0)])
    m["w_tie_near_4"] = sym(4, [(0, 1, 10.0), (1, 3, 5.0), (0, 3, 1.0 / 0.3), (0, 2, 1.0), (2, 1, 1.0)])
    rnd = random.Random(61)
    A = zeros(10)
    for i in range(10):
        for j in range(10):
            if i != j and rnd.random() < 0.45:
                A[i][j] = rnd.uniform(0.05, 1.0)
    m["w_asym_10"] = A
    m["w_disc_10"] = sym(10, [(0, 1, .9), (1, 2, .4), (0, 2, .7), (3, 4, .2), (4, 5, .8), (5, 6, .3), (6, 3, .6), (7, 8, .5), (8, 9, .5)])
    m["w_isolated_8"] = sym(8, [(0, 1, .3), (1, 2, .6), (2, 3, .9), (0, 3, .1), (4, 5, .7), (5, 6, .2)])
    m["w_star_7"] = sym(7, [(0, k, 0.1 * k) for k in range(1, 7)])
    # an absorbed length: 1 + 2^-60 == 1 in binary64
    m["w_absorb_5"] = sym(5, [(0, 1, 1.0), (1, 2, 2.0 ** 60), (2, 3, 1.0), (0, 4, 0.5), (4, 3, 0.25)])
    m["w_subnormal_4"] = sym(4, [(0, 1, 1e-310), (1, 2, 0.5), (2, 3, 0.5)])
    m["w_negative_5"] = sym(5, [(0, 1, 0.5), (1, 2, -0.25), (2, 3, 0.5), (3, 4, 0.75)])
    rnd = random.Random(62)
    m["w_span_8"] = sym(8, [(i, j, 10.0 ** rnd.uniform(-300, 300)) for i in range(8) for j in range(i + 1, 8) if rnd.random() < 0.6])
    for name, n, p, seed in (("w_complete_6", 6, 1.01, 63), ("w_dense_40", 40, 0.7, 64), ("w_dense_60", 60, 0.85, 65), ("w_rc_12", 12, 0.35, 66)):
        rnd = random.Random(seed)
        m[name] = sym(n, [(i, j, rnd.uniform(0.01, 1.0)) for i in range(n) for j in range(i + 1, n) if rnd.random() < p])
    rnd = random.Random(67)
    D = zeros(30)
    for i in range(30):
        for j in range(30):
            if i != j and rnd.random() < (0.9 if i < 6 else 0.15):
                D[i][j] = rnd.uniform(0.01, 1.0)
    m["w_hubdir_30"] = D  # hubs whose total degree exceeds n (rich club levels > n)
    # binary: walk-count overflow case for M05 (clique 120 + tail 170)
    clique, tail = 120, 170
    L = zeros(clique + tail)
    for i in range(clique):
        for j in range(i + 1, clique):
            L[i][j] = L[j][i] = 1.0
    for t in range(tail):
        L[clique - 1 + t][clique + t] = L[clique + t][clique - 1 + t] = 1.0
    m["g_lollipop_290"] = L
    rnd = random.Random(68)
    H = zeros(20)
    for i in range(20):
        for j in range(20):
            if i != j and rnd.random() < (0.95 if i < 4 or j < 4 else 0.1):
                H[i][j] = 1.0
    m["g_hubdir_20"] = H
    # inverse-distance overflow (M49/M50): 1/W is subnormal, so the length check passes, but 1/length = +Inf
    m["w_maxweight_3"] = sym(3, [(0, 1, 1.0), (1, 2, 1.7976931348623157e308), (0, 2, 1.0)])
    # boundary: W = 2^1023, 1/(1/W) = W exactly, no overflow
    m["w_halfmax_3"] = sym(3, [(0, 1, 1.0), (1, 2, 2.0 ** 1023), (0, 2, 1.0)])
    json.dump({k: mat(v) for k, v in m.items()}, open(os.path.join(OUT, "matrices.json"), "w"))

    # measure cases
    p1cases = json.load(open(os.path.join(PHASE1, "construct.json")))
    p1mats = json.load(open(os.path.join(PHASE1, "matrices.json")))
    by_key = {(c["matrix"], c["type"], c["value"], c["mode"], c["neg_rule"], json.dumps(c["mask"])): c["case"] for c in p1cases}
    mc = []
    for mid in ("corrA_379", "corrB_379", "corrT_379", "asym_379", "near_379"):
        for v in ("0.10", "0.30", "0.50"):
            mc.append({"graph": {"construct": by_key[(mid, "proportional", v, "binary", "keep", "null")]}, "measures": B_ITEMS})
            for rule in ("keep", "absolute", "zero"):
                mc.append({"graph": {"construct": by_key[(mid, "proportional", v, "weighted", rule, "null")]}, "measures": W_ITEMS})
    for mid in ("s5", "s44", "ties_12", "zero_5", "single_edge_5", "asym_12"):
        mc.append({"graph": {"construct": by_key[(mid, "proportional", "0.5", "binary", "keep", "null")]}, "measures": B_ITEMS})
        for rule in ("keep", "absolute"):
            mc.append({"graph": {"construct": by_key[(mid, "none", None, "weighted", rule, "null")]}, "measures": W_ITEMS})
    for k in p1mats:
        if k.startswith("g_"):
            mc.append({"graph": {"phase1_matrix": k}, "measures": B_ITEMS})
        elif k.startswith("w_"):
            mc.append({"graph": {"phase1_matrix": k}, "measures": W_ITEMS})
    for k in m:
        if k == "g_lollipop_290":
            mc.append({"graph": {"matrix": k}, "measures": ["M05", "M64"]})
        elif k.startswith("g_"):
            mc.append({"graph": {"matrix": k}, "measures": B_ITEMS})
        else:
            mc.append({"graph": {"matrix": k}, "measures": W_ITEMS})
    for i, c in enumerate(mc):
        c["case"] = f"p103-measure-{i + 1:04d}"
    json.dump(mc, open(os.path.join(OUT, "measures.json"), "w"))

    # primitive batteries
    rnd = random.Random(70)
    xs = [0.0, 1.0, 8.0, 27.0, 0.125, 2.0 ** -1074, 2.0 ** -1022, 2.0 ** 1023, 1.7976931348623157e308]
    xs += [float(k) for k in range(2, 2001)]
    xs += [float(k ** 3) for k in range(2, 1001)]
    xs += [2.0 ** e for e in range(-1074, 1024, 3)]
    while len(xs) < 379 * 80:
        r = rnd.random()
        if r < 0.4:
            xs.append(rnd.uniform(0.0, 1.0))
        elif r < 0.6:
            xs.append(1.0 + rnd.uniform(-1e-6, 1e-6))
        elif r < 0.9:
            xs.append(rnd.uniform(1.0, 2.0) * 2.0 ** rnd.randint(-1074, 1023))
        else:
            xs.append(rnd.uniform(1.0, 100.0))
    xs = [x if x != float("inf") else 1.0 for x in xs][:379 * 80]
    prim = {"pow": vec(xs), "outer": [], "sum1d": [], "rowsum": []}
    for k in (1, 2, 3, 7, 44, 45, 100, 379):
        prim["outer"].append(vec([rnd.uniform(0.0, 2.0) * 2.0 ** rnd.randint(-30, 30) for _ in range(k)]))
    for L in (4096 + 10, 4096 + 43, 8192 + 44, 4096 * 44, 4096 * 44 + 5, 4096 * 45 + 3, 200000):
        prim["sum1d"].append(vec([rnd.uniform(-1, 1) * 2.0 ** rnd.randint(-20, 20) for _ in range(L)]))
    prim["sum1d"].append(vec([-0.0] * (4096 + 10)))
    prim["rowsum"].append(mat([[-0.0, -0.0, -0.0], [1.0, -0.0, 2.0], [0.0, -0.0, -0.0]]))
    prim["rowsum"].append(mat([[-0.0] * 50, [(-0.0 if j else 0.0) for j in range(50)], [rnd.uniform(-1, 1) for _ in range(50)]]))
    json.dump(prim, open(os.path.join(OUT, "prim.json"), "w"))
    print(f"{len(m)} matrices, {len(mc)} measure cases, {len(xs)} pow values", file=sys.stderr)


if __name__ == "__main__":
    main()
