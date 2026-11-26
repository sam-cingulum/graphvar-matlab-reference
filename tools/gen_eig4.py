"""Inputs for eig probe 4 (drivers/probe_eig4.m). Writes <out>/eig4.json:
  nonfinite  [{"case", "A"}]: NaN / Inf / -Inf placed in symmetric and asymmetric positions (P2-04 question 2,
             and question 1 through any LAPACK non-convergence that MATLAB passes on)
  tiny       [{"case", "A"}]: finite symmetric matrices whose Householder columns have |beta| < 2^-969
             (dlarfg's rescale loop, P2-04 question 3)
  larfg      [{"case", "alpha", "x"}]: MKL dlarfg inputs around and below 2^-969, subnormals, signed zeros
  walk       [{"case", "W"}]: P2-03 graphs where BCT's hop count and next-node walk disagree by rounding
  steqr      [{"case", "kind", "d", "e"}]: tridiagonals for MKL dsteqr('I'), n = 2..64
  ormtr      [{"case", "n", "A"}]: gauss matrices for the dormtr crossover (n = 34..47) and n >= 257
Synthetic only (seeded). Needs no other input.
"""
import json, math, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def vec(v):
    return {"size": [len(v), 1], "hex": [hx(t) for t in v]}


def matl(n, M):
    return {"size": [n, n], "hex": [hx(M[i][j]) for j in range(n) for i in range(n)]}


def gauss_sym(n, rng):
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            M[i][j] = M[j][i] = rng.gauss(0, 1)
    return M


WALK = [["p203w-early-min", [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 0.0, 1.0, 3.0], [1.0, 3.0, 1.0, 0.0, 0.0], [0.0, 3.0, 3.0, 0.0, 0.0]]], ["p203w-late-min", [[0.0, 3.0, 1.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 0.0, 3.0, 0.0], [1.0, 0.0, 0.0, 1.0, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 3.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], ["p203w-early-1", [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 3.0, 1.0, 1.0], [0.0, 1.0, 3.0, 0.0, 0.0, 3.0, 1.0], [0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0]]], ["p203w-early-2", [[0.0, 0.0, 10.0, 3.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 3.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [3.0, 10.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 3.0, 0.0]]], ["p203w-early-3", [[0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0], [2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 3.0], [3.0, 0.0, 0.0, 1.0, 1.0, 3.0, 0.0]]], ["p203w-early-4", [[0.0, 3.0, 0.0, 1.0, 1.0], [3.0, 0.0, 3.0, 0.0, 1.0], [0.0, 3.0, 0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 3.0, 0.0, 0.0]]], ["p203w-early-5", [[0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0], [3.0, 2.0, 0.0, 3.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 3.0, 3.0, 1.0, 0.0]]], ["p203w-late-1", [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 10.0, 0.0], [0.0, 1.0, 0.0, 0.0, 3.0, 10.0, 0.0, 0.0], [1.0, 10.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 10.0, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 10.0, 0.0, 0.0, 3.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]]], ["p203w-late-2", [[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 1.0], [0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0], [0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0, 3.0, 0.0]]], ["p203w-late-3", [[0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 3.0, 3.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]], ["p203w-late-4", [[0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 0.0, 3.0, 3.0, 0.0], [5.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0]]], ["p203w-late-5", [[0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0], [10.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0], [10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]]]


def main(out):
    rng = random.Random(4404)
    doc = {}
    # ---- non-finite inputs -------------------------------------------------------------
    nf = []
    nan, inf = float("nan"), float("inf")
    for n in (1, 2, 5, 26, 201):
        base = gauss_sym(n, rng)
        def put(tag, cells):
            M = [r[:] for r in base]
            for (i, j, v) in cells:
                M[i][j] = v
            nf.append({"case": f"eig4-nonfinite-{tag}-n{n:03d}", "A": matl(n, M)})
        put("nan-diag", [(0, 0, nan)])
        put("inf-diag", [(0, 0, inf)])
        put("minf-diag", [(n - 1, n - 1, -inf)])
        if n >= 2:
            put("nan-pair", [(0, n - 1, nan), (n - 1, 0, nan)])
            put("inf-pair", [(0, n - 1, inf), (n - 1, 0, inf)])
            put("inf-minf-pair", [(0, 1, inf), (1, 0, -inf)])
            put("nan-upper", [(0, n - 1, nan)])
            put("inf-upper", [(0, n - 1, inf)])
            put("inf-and-nan", [(0, 0, inf), (n - 1, n - 1, nan)])
            put("inf-diag2", [(0, 0, inf), (n - 1, n - 1, -inf)])
        put("all-nan", [(i, j, nan) for i in range(n) for j in range(n)])
        put("all-inf", [(i, j, inf) for i in range(n) for j in range(n)])
    doc["nonfinite"] = nf
    # ---- tiny Householder columns --------------------------------------------------------
    tiny = []
    for n in (3, 5, 12, 26, 65):
        for ex in (-975, -1000, -1030, -1060, -1074):
            M = gauss_sym(n, rng)
            for i in range(n - 1):          # last column (the first reduced, uplo U) is tiny
                v = rng.gauss(0, 1) * 2.0 ** (ex + rng.randint(-2, 2))
                if ex == -1074: v = rng.choice([1, -1, 2, -3]) * 2.0 ** -1074
                M[i][n - 1] = M[n - 1][i] = v
            tiny.append({"case": f"eig4-tiny-last-e{-ex}-n{n:03d}", "A": matl(n, M)})
        M = gauss_sym(n, rng)
        for i in range(n):
            for j in range(i + 1, n):
                M[i][j] = M[j][i] = rng.gauss(0, 1) * 2.0 ** rng.randint(-1060, -975)
        tiny.append({"case": f"eig4-tiny-alloff-n{n:03d}", "A": matl(n, M)})
        M = gauss_sym(n, rng)
        for i in range(n):
            for j in range(i + 1, n):
                if (i + j) % 3 == 0: M[i][j] = M[j][i] = rng.gauss(0, 1) * 2.0 ** -1000
        tiny.append({"case": f"eig4-tiny-mixed-n{n:03d}", "A": matl(n, M)})
    doc["tiny"] = tiny
    # ---- dlarfg kernel -------------------------------------------------------------------
    lf = []
    for t in range(300):
        m = rng.choice([1, 2, 3, 4, 5, 8, 13, 24, 31])
        e = rng.choice([-500, -960, -969, -970, -980, -1000, -1022, -1030, -1060, -1074])
        def g():
            if rng.random() < 0.1: return rng.choice([0.0, -0.0])
            if e == -1074: return rng.choice([1, -1, 3, -2]) * 2.0 ** -1074
            return rng.gauss(0, 1) * 2.0 ** (e + rng.randint(-6, 6))
        x = [g() for _ in range(m)]
        alpha = rng.choice([g(), 0.0, -0.0, rng.gauss(0, 1) * 2.0 ** e, rng.gauss(0, 1)])
        lf.append({"case": f"eig4-larfg-{t:03d}", "alpha": vec([alpha]), "x": vec(x)})
    doc["larfg"] = lf
    # ---- P2-03 walk graphs -----------------------------------------------------------------
    doc["walk"] = [{"case": c, "W": matl(len(W), W)} for c, W in WALK]
    # ---- dsteqr stress -----------------------------------------------------------------------
    st = []
    kinds = ["gauss", "int", "dzero", "ties", "graded", "wilk", "tinye", "huge", "tiny", "mixed", "pm", "sparse_e"]
    for t in range(600):
        k = kinds[t % len(kinds)]
        n = rng.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 16, 20, 24, 25, 26, 32, 33, 40, 64])
        g = lambda: rng.gauss(0, 1)
        if k == "gauss": d = [g() for _ in range(n)]; e = [g() for _ in range(n - 1)]
        elif k == "int": d = [float(rng.randint(-3, 3)) for _ in range(n)]; e = [float(rng.randint(-2, 2)) for _ in range(n - 1)]
        elif k == "dzero": d = [0.0] * n; e = [g() for _ in range(n - 1)]
        elif k == "ties": c = g(); d = [c] * n; e = [g() * 10 ** rng.randint(-12, 0) for _ in range(n - 1)]
        elif k == "graded": d = [g() * 10.0 ** (-i) for i in range(n)]; e = [g() * 10.0 ** (-i - 0.5) for i in range(n - 1)]
        elif k == "wilk": m = (n - 1) / 2; d = [abs(i - m) for i in range(n)]; e = [1.0] * (n - 1)
        elif k == "tinye": d = [g() for _ in range(n)]; e = [g() * 2.0 ** rng.randint(-80, -20) for _ in range(n - 1)]
        elif k == "huge": s = 2.0 ** rng.randint(480, 509); d = [g() * s for _ in range(n)]; e = [g() * s for _ in range(n - 1)]
        elif k == "tiny": s = 2.0 ** rng.randint(-1000, -400); d = [g() * s for _ in range(n)]; e = [g() * s for _ in range(n - 1)]
        elif k == "mixed": d = [g() * 2.0 ** rng.randint(-300, 300) for _ in range(n)]; e = [g() * 2.0 ** rng.randint(-300, 300) for _ in range(n - 1)]
        elif k == "pm": d = [(-1.0) ** i for i in range(n)]; e = [rng.choice([1.0, -1.0, 0.5]) for _ in range(n - 1)]
        else: d = [g() for _ in range(n)]; e = [g() if rng.random() < 0.5 else 0.0 for _ in range(n - 1)]
        st.append({"case": f"eig4-steqr-{k}-n{n:03d}-{t:03d}", "kind": k, "d": vec(d), "e": vec(e)})
    doc["steqr"] = st
    # ---- dormtr sizes ----------------------------------------------------------------------------
    orm = []
    for n in list(range(34, 48)) + [257, 264, 280, 288, 300, 320, 350, 379, 400]:
        orm.append({"case": f"eig4-ormtr-gauss-n{n:03d}", "n": n, "A": matl(n, gauss_sym(n, rng))})
    doc["ormtr"] = orm
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eig4.json"), "w") as f:
        json.dump(doc, f, allow_nan=False)
    print(f"eig4: {len(nf)} nonfinite, {len(tiny)} tiny, {len(lf)} larfg, {len(doc['walk'])} walk, {len(st)} steqr, {len(orm)} ormtr", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cases/eig")
