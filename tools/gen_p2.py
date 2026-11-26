"""Generate the phase-2 inputs: probes of MATLAB's real matrix multiply (MKL dgemm).

Writes cases/p2/gemm.json, and the P2-01 / P1-01 v2 battery:
cases/p2/{construct,matrices,measures}.json. Run tools/gen_p103.py first:
measure cases also reuse the P1-03 graphs (cases/p103) and phase-1 cases.
Pure Python, fixed seeds, no patient data.
Every number is a 16-hex-digit binary64 word (big-endian, like num2hex),
arrays column-major.

Batteries
- tree:  for inner length k, every pair (a, b) of term positions gets one row of
         A with +2^80 at a, -2^80 at b and 1 elsewhere; B = ones(k). Each output
         entry is then k - |subtree of the reduction tree at LCA(a, b)|, which pins
         the order the k products are added in. Calls are square k x k times
         k x k, as GraphVar uses them; k rows per call.
- fma:   rows of A hold 2^53 at q and (1+2^-26) at p; B rows are 2^-53 at even
         positions and 2^-53 (1-2^-26+2^-52) at odd ones. The two products are 1
         and 2^-53 (1+2^-78). Rounded separately, 1 + 2^-53 ties to 1; fused, it
         gives 1 + 2^-52. q is even, p odd, both orders.
- zero:  signs of zero results: rows of -0, and rows whose products cancel.
- rand:  Gaussian A, B (twister-free: Python random, seed fixed) for model checks.
- mpow:  non-negative symmetric weights for A^2, A^3, A*A.', A.'*A.
"""
import json, os, random, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "cases/p2"
M = 2.0 ** 80
X26 = 1.0 + 2.0 ** -26
Y26 = (1.0 - 2.0 ** -26 + 2.0 ** -52) * 2.0 ** -53
from fractions import Fraction as _F
assert _F(X26) * _F(Y26) == _F(2) ** -53 * (1 + _F(2) ** -78)  # the exact product the fma probe relies on

TREE_K = list(range(1, 17)) + [20, 24, 31, 32, 33, 44, 60, 63, 64, 65, 100, 116, 128, 160, 200, 255, 256, 257, 300, 379]
RAND_N = [2, 3, 5, 8, 16, 20, 44, 64, 65, 90, 116, 200, 256, 257, 379]
MPOW_N = [3, 5, 44, 116, 379]


def hx(x):
    return struct.pack(">d", float(x)).hex()


def mat(rows):
    r, c = len(rows), len(rows[0])
    return {"size": [r, c], "hex": [hx(rows[i][j]) for j in range(c) for i in range(r)]}


def tree_calls(k):
    pairs = [(a, b) for a in range(k) for b in range(k) if a != b] if k <= 16 else \
            [(a, b) for a in range(k) for b in range(a + 1, k)]
    calls = []
    for s in range(0, len(pairs), k):
        chunk = pairs[s:s + k]
        while len(chunk) < k:
            chunk.append(pairs[0])
        # The driver builds A itself: ones(k), then A(i, a) = 2^80 and A(i, b) = -2^80.
        calls.append({"pairs": [[a + 1, b + 1] for a, b in chunk]})
    return calls


def fma_call(k):
    qs = list(range(0, k, 2))
    ps = list(range(1, k, 2))
    cand = [(q, p) for q in qs for p in ps]
    if len(cand) > k:
        rnd = random.Random(1000 + k)
        must = [(0, 1), (0, ps[-1]), (qs[-1], ps[-1]), (qs[-1], 1)]
        if k > 2:
            must += [(2, 1)]
        rest = [c for c in cand if c not in must]
        rnd.shuffle(rest)
        cand = (must + rest)[:k]
    while len(cand) < k:
        cand.append(cand[0])
    rows = []
    for q, p in cand:
        row = [0.0] * k
        row[q], row[p] = 2.0 ** 53, X26
        rows.append(row)
    B = [[(Y26 if l % 2 else 2.0 ** -53) for _ in range(k)] for l in range(k)]
    return {"qp": [[q + 1, p + 1] for q, p in cand], "A": mat(rows), "B": mat(B)}


def zero_call(k):
    rnd = random.Random(2000 + k)
    rows = [[-0.0] * k, [0.0] * k]
    if k >= 2:
        rows.append([1.0, -1.0] + [-0.0] * (k - 2))
        rows.append([-0.0] * (k - 2) + [1.0, -1.0])
        rows.append([-1.0, 1.0] + [0.0] * (k - 2))
    while len(rows) < k:
        rows.append([rnd.choice([-0.0, 0.0]) for _ in range(k)])
    rows = rows[:k]
    return {"A": mat(rows)}


P2_ITEMS = [f"M{k}" for k in range(65, 75)]


def p2_battery():
    phase1 = os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", "phase1")
    p103 = os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", "p103")
    p1cases = json.load(open(os.path.join(phase1, "construct.json")))
    p1mats = json.load(open(os.path.join(phase1, "matrices.json")))
    p103mats = json.load(open(os.path.join(p103, "matrices.json")))
    by_key = {(c["matrix"], c["type"], c["value"], c["mode"], c["neg_rule"], json.dumps(c["mask"])): c["case"] for c in p1cases}
    # P1-01 v2: rank = magnitude (every case carries the field, so jsondecode gives a struct array)
    con = []
    for mid in ("corrA_379", "corrB_379", "corrT_379", "asym_379", "near_379"):
        for v in ("0.10", "0.20", "0.30", "0.40", "0.50"):
            for mode in ("binary", "weighted"):
                con.append({"matrix": mid, "type": "proportional", "value": v, "mode": mode, "neg_rule": "keep",
                            "mask": None, "pvalues": None, "rank": "magnitude"})
    for mid in ("s5", "s44", "ties_12", "asym_12"):
        for mode in ("binary", "weighted"):
            con.append({"matrix": mid, "type": "proportional", "value": "0.5", "mode": mode, "neg_rule": "keep",
                        "mask": None, "pvalues": None, "rank": "magnitude"})
    for i, c in enumerate(con):
        c["case"] = f"p2-construct-{i + 1:04d}"
    # hand-built graphs for clustering and transitivity
    rnd = random.Random(202611272)
    def zeros(n):
        return [[0.0] * n for _ in range(n)]
    m = {}
    W = zeros(3); W[0][1], W[1][2], W[2][0] = 0.5, 2.0, 3.0; m["w_dir3cycle_3"] = W
    W = zeros(6); W[0][1], W[1][2], W[2][0] = 0.25, 0.5, 0.75; W[3][4] = W[4][3] = 0.6; W[4][5] = 0.9; W[2][3] = 0.1; m["w_dir3cycle_6"] = W
    m["w_empty_6"] = zeros(6)
    W = zeros(4); W[0][1] = W[1][0] = 0.7; m["w_recip_4"] = W
    W = zeros(3); W[0][1] = W[1][0] = 5e-324; W[1][2] = W[2][1] = 1e-310; W[0][2] = W[2][0] = 2.2e-308; m["w_tri_subnormal_3"] = W
    W = zeros(5)
    for i in range(5):
        for j in range(i + 1, 5):
            W[i][j] = W[j][i] = 1.7976931348623157e308 if (i + j) % 2 else 1e300
    m["w_huge_5"] = W
    W = zeros(5); W[0][1] = W[1][0] = 0.4; W[1][2] = W[2][1] = -0.3; W[0][2] = W[2][0] = 0.8; m["w_negative_5b"] = W
    W = zeros(4)
    for i in range(4):
        for j in range(4):
            if i != j: W[i][j] = 1.0
    m["w_complete_4"] = W
    W = zeros(30)
    for i in range(30):
        for j in range(30):
            if i != j and rnd.random() < 0.3: W[i][j] = rnd.random()
    m["w_dir_30"] = W
    W = zeros(40)
    for i in range(40):
        for j in range(i + 1, 40):
            if rnd.random() < 0.4: W[i][j] = W[j][i] = rnd.random()
    m["w_sym_40"] = W
    W = zeros(97)
    for i in range(97):
        for j in range(i + 1, 97):
            if rnd.random() < 0.5: W[i][j] = W[j][i] = rnd.random() * 2.0 ** rnd.randint(-8, 8)
    m["w_sym_97"] = W
    mc = []
    for mid in ("corrA_379", "corrB_379", "corrT_379", "asym_379", "near_379"):
        for v in ("0.10", "0.30", "0.50"):
            for rule in ("keep", "absolute", "zero"):
                mc.append({"graph": {"construct": by_key[(mid, "proportional", v, "weighted", rule, "null")]}, "measures": P2_ITEMS})
    for mid in ("s5", "s44", "ties_12", "zero_5", "single_edge_5", "asym_12"):
        for rule in ("keep", "absolute"):
            mc.append({"graph": {"construct": by_key[(mid, "none", None, "weighted", rule, "null")]}, "measures": P2_ITEMS})
    for k in p1mats:
        if k.startswith("w_"):
            mc.append({"graph": {"phase1_matrix": k}, "measures": P2_ITEMS})
    for k in p103mats:
        if not k.startswith("g_"):
            mc.append({"graph": {"matrix": k}, "measures": P2_ITEMS})
    for c in con:
        if c["mode"] == "weighted":
            mc.append({"graph": {"p2_construct": c["case"]}, "measures": P2_ITEMS})
    for k in m:
        mc.append({"graph": {"p2_matrix": k}, "measures": P2_ITEMS})
    for i, c in enumerate(mc):
        c["case"] = f"p2-measure-{i + 1:04d}"
    json.dump(con, open(os.path.join(OUT, "construct.json"), "w"))
    json.dump({k: mat(v) for k, v in m.items()}, open(os.path.join(OUT, "matrices.json"), "w"))
    json.dump(mc, open(os.path.join(OUT, "measures.json"), "w"))
    print(f"P2-01 battery: {len(con)} constructions, {len(m)} matrices, {len(mc)} measure cases")


def main():
    os.makedirs(OUT, exist_ok=True)
    p2_battery()
    out = {"tree": [], "fma": [], "zero": [], "rand": [], "mpow": []}
    # Long sums: the driver builds v(i) = (mod(i*7919, 10007) - 5003) * 2^(mod(i, 41) - 20), i = 1..L,
    # every value exact, and sums it with 1 and with 4 computational threads.
    out["sums"] = [65536, 65537, 72900, 73000, 80000, 81920, 90000, 98304, 100000, 100001, 102400, 110000,
                   114688, 120000, 122880, 125000, 131072, 131073, 135000, 140000, 143641, 150000, 163840,
                   250000, 1000000, 1000003]
    for k in TREE_K:
        out["tree"].append({"k": k, "calls": tree_calls(k)})
        if k >= 2:
            out["fma"].append({"k": k, **fma_call(k)})
        out["zero"].append({"k": k, **zero_call(k)})
    rnd = random.Random(20261127)
    for n in RAND_N:
        A = [[rnd.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        B = [[rnd.gauss(0, 1) for _ in range(n)] for _ in range(n)]
        out["rand"].append({"n": n, "A": mat(A), "B": mat(B)})
    for n in MPOW_N:
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if rnd.random() < 0.3:
                    W[i][j] = W[j][i] = rnd.random()
        out["mpow"].append({"n": n, "W": mat(W)})
    with open(os.path.join(OUT, "gemm.json"), "w") as f:
        json.dump(out, f)
    nt = sum(len(t["calls"]) for t in out["tree"])
    print(f"tree k values {len(TREE_K)}, calls {nt}; fma {len(out['fma'])}; zero {len(out['zero'])}; rand {len(RAND_N)}; mpow {len(MPOW_N)}")


if __name__ == "__main__":
    main()
