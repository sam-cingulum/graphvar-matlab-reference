"""Inputs for eig probe 3 (drivers/probe_eig3.m). Writes <out>/eig3.json:
  blocked     [{"case", "n", "A"}]: extra gauss sizes for the blocked dsytrd (n > 200); the probe also
              takes every symmetric eig.json case with n > 200
  snap_n      sizes whose blocked dsytrd panels are recorded in full (A after each dlatrd and dsyr2k)
  latrd_n     gauss sizes (eig.json or "blocked") for the raw dlatrd records, nb in latrd_nb
  ormtr_snap_n  sizes whose dormtr blocks are recorded (T and C after each block)
  steqr       [{"case", "kind", "d", "e"}]: tridiagonals for MKL dsteqr('I')
  scale       [{"case", "kind", "A"}]: matrices whose max |entry| sits at or beyond 2^+-485
  gemm_nn     [{"m", "n", "k", "ldc"}]: random-word shapes (dstedc's dlaed3 calls, reference LAPACK)
  fprev_gemm  [{"m", "n", "L", "rows", "cols"}]: dgemm('N','N') FPRev shapes
  fprev_gemv_t  [{"m", "n"}], fprev_gemv_n [{"m", "n"}]
  fprev_trmm  [{"uplo", "trans", "diag", "m", "k"}]
  fprev_syr2k [{"n", "k", "d"}]
  kern        random-word kernel records (see probe_eig3.m)
Synthetic only (seeded). Needs no other input; probe_eig3 also reads eig.json from tools/gen_eig.py.
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def vec(v):
    return {"size": [len(v), 1], "hex": [hx(t) for t in v]}


def mat(n, M):
    return {"size": [n, n], "hex": [hx(M[i][j]) for j in range(n) for i in range(n)]}


def sym(n, upper):
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            M[i][j] = M[j][i] = upper(i, j)
    return M


def rows_sample(m):
    s = set(range(1, 15)) | set(range(23, 27)) | set(range(35, 39)) | set(range(47, 51)) | {m // 2, m // 2 + 1} | set(range(m - 13, m + 1))
    return sorted(x for x in s if 1 <= x <= m)


def cols_sample(n):
    s = set(range(1, 10)) | {12, 13, 14, n // 2, n // 2 + 1} | set(range(n - 8, n + 1))
    return sorted(x for x in s if 1 <= x <= n)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    doc = {}

    # blocked dsytrd: same generator as gen_eig.py's gauss kind (seed 7000 + n)
    doc["blocked"] = []
    for n in (201, 224, 225, 248):
        r = random.Random(7000 + n)
        doc["blocked"].append({"case": f"eig3-gauss-n{n:03d}", "n": n, "A": mat(n, sym(n, lambda i, j: r.gauss(0.0, 1.0)))})
    doc["snap_n"] = [225, 379]
    doc["latrd_n"] = [26, 65, 201, 225, 379]
    doc["latrd_nb"] = [1, 2, 3, 24]
    doc["ormtr_snap_n"] = [65, 129, 379]

    # dsteqr targets: splits (e = 0), zero diagonals, tiny off-diagonals at the split threshold, ties
    st = []
    def addst(kind, d, e):
        st.append({"case": f"eig3-steqr-{kind}-n{len(d):03d}-{len(st):03d}", "kind": kind, "d": vec(d), "e": vec(e)})
    for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 25):
        r = random.Random(13000 + n)
        d = [r.gauss(0.0, 1.0) for _ in range(n)]
        e = [r.gauss(0.0, 1.0) for _ in range(n - 1)]
        addst("gauss", d, e)
        addst("dzero", [0.0] * n, e)
        for p in sorted({0, (n - 1) // 2, n - 2}):
            if n < 2: continue
            e2 = list(e); e2[p] = 0.0
            addst(f"esplit{p + 1}", d, e2)
            addst(f"dzero-esplit{p + 1}", [0.0] * n, e2)
            # tiny e(p): |e(p)| just below and just above eps2 * sqrt(|d(p)| |d(p+1)|) (dsteqr's split test)
            eps2 = 2.0 ** -106
            t = (eps2 * abs(d[p] * d[p + 1])) ** 0.5
            for f, nm in ((0.5, "tinylo"), (2.0, "tinyhi")):
                e3 = list(e); e3[p] = t * f
                addst(f"{nm}{p + 1}", d, e3)
        addst("ties", [1.0] * n, [0.0] * (n - 1))
        addst("ties-smalle", [1.0] * n, [2.0 ** -30 * x for x in e])
        addst("graded", [2.0 ** (-3 * i) for i in range(n)], [2.0 ** (-3 * i - 1) for i in range(n - 1)])
        addst("binlike", [0.0] * n, [1.0 if r.random() < 0.5 else 0.0 for _ in range(n - 1)])
    doc["steqr"] = st

    # scaling bounds (dsyevd: scale when 0 < max|a| < 2^-485 or max|a| > 2^485)
    sc = []
    for n in (8, 30, 100):
        r = random.Random(14000 + n)
        A0 = sym(n, lambda i, j: r.uniform(-0.999, 0.999))
        A0[0][n - 1] = A0[n - 1][0] = 1.0       # the max |entry| is exactly 1 before scaling
        for nm, s in (("p485", 2.0 ** 485), ("p485up", 2.0 ** 485 * (1 + 2.0 ** -52)), ("p600", 2.0 ** 600),
                      ("m485", 2.0 ** -485), ("m485dn", 2.0 ** -485 * (1 - 2.0 ** -53)), ("m600", 2.0 ** -600)):
            M = [[x * s for x in row] for row in A0]
            if nm in ("p485up", "m485dn"):      # only the max pair moves; the rest stay scaled by the power of two
                p2 = 2.0 ** 485 if nm == "p485up" else 2.0 ** -485
                M = [[x * p2 for x in row] for row in A0]
                M[0][n - 1] = M[n - 1][0] = s
            sc.append({"case": f"eig3-scale-{nm}-n{n:03d}", "kind": nm, "A": mat(n, M)})
    doc["scale"] = sc

    # dgemm('N','N') shapes of reference dstedc's dlaed3 on the eig.json tridiagonals (m, n, k, ldc)
    doc["gemm_nn"] = [{"m": m, "n": nn, "k": k, "ldc": ldc} for (m, nn, k, ldc) in [
        (13, 26, 13, 26), (15, 31, 15, 31), (16, 31, 16, 31), (16, 32, 16, 32), (16, 33, 16, 33), (17, 33, 17, 33),
        (24, 48, 24, 48), (15, 31, 15, 63), (31, 63, 31, 63), (32, 63, 32, 63), (32, 64, 32, 64), (32, 65, 32, 65),
        (33, 65, 33, 65), (25, 50, 25, 100), (50, 100, 50, 100), (25, 49, 24, 100), (50, 99, 49, 100),
        (64, 128, 64, 128), (64, 128, 64, 129), (65, 128, 64, 129), (100, 194, 94, 200), (100, 194, 100, 200),
        (64, 127, 63, 256), (128, 246, 118, 256), (128, 246, 128, 256), (18, 37, 18, 300), (19, 38, 19, 300),
        (37, 75, 37, 300), (38, 75, 38, 300), (75, 148, 73, 300), (75, 150, 75, 300), (150, 282, 132, 300),
        (150, 282, 150, 300), (100, 195, 95, 400), (100, 198, 100, 400), (200, 370, 170, 400), (200, 370, 200, 400),
        (190, 360, 189, 379), (189, 379, 189, 379)]]
    fg = [{"m": 37, "n": 13, "L": L, "rows": list(range(1, 38)), "cols": list(range(1, 14))}
          for L in list(range(2, 51)) + [63, 64, 65, 94, 95, 96, 98, 100, 118, 128, 132, 150, 170, 200, 256, 257, 300]]
    for (m, nn, L) in [(13, 26, 13), (16, 32, 16), (17, 33, 17), (24, 48, 24), (25, 50, 25), (32, 64, 32), (33, 65, 33),
                       (50, 100, 50), (64, 128, 64), (75, 150, 75), (100, 198, 100), (128, 246, 118), (150, 282, 132),
                       (200, 370, 170), (200, 370, 200), (1, 9, 20), (2, 9, 20), (3, 9, 20), (4, 9, 20), (5, 9, 20),
                       (8, 9, 20), (11, 9, 20), (12, 9, 20), (13, 9, 20), (25, 1, 20), (25, 2, 20), (25, 3, 20),
                       (25, 4, 20), (25, 5, 20), (25, 6, 20), (25, 7, 20), (25, 8, 20)]:
        fg.append({"m": m, "n": nn, "L": L, "rows": rows_sample(m), "cols": cols_sample(nn)})
    doc["fprev_gemm"] = fg
    doc["fprev_gemv_t"] = ([{"m": m, "n": k} for m in range(2, 49) for k in (1, 2, 3, 4, 5, 6, 7, 8, 23)] +
                           [{"m": m, "n": k} for m in (63, 64, 65, 183, 231, 299, 399) for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 20, 23)])
    doc["fprev_gemv_n"] = [{"m": m, "n": k} for m in (5, 37, 399) for k in range(1, 25)]
    doc["fprev_trmm"] = [{"uplo": u, "trans": t, "diag": dg, "m": m, "k": k}
                         for (u, t, dg) in (("U", "N", "U"), ("L", "T", "N"), ("U", "T", "U"))
                         for m in (37, 400) for k in (2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32)]
    doc["fprev_syr2k"] = ([{"n": 41, "k": k, "d": d} for k in (1, 2, 3, 4, 5, 8, 24) for d in (1, 2, 3, 4, 5, 8, 12, 13)] +
                          [{"n": 232, "k": 24, "d": d} for d in (1, 2, 3, 4, 5, 8, 12, 13, 24, 25, 100)])

    # random-word kernel records
    kern = []
    r = random.Random(15000)
    g = lambda: r.gauss(0.0, 1.0)
    for (ta, tb, m, nn, k) in [("T", "N", 65, 16, 48), ("T", "N", 129, 32, 96), ("T", "N", 379, 32, 346), ("T", "N", 379, 27, 351),
                               ("T", "N", 26, 8, 17), ("N", "T", 48, 65, 16), ("N", "T", 96, 129, 32), ("N", "T", 346, 379, 32),
                               ("N", "T", 351, 379, 27), ("N", "T", 17, 26, 8)]:
        kern.append({"kind": "gemm", "ta": ta, "tb": tb, "m": m, "n": nn, "k": k, "seed": r.randrange(1 << 30)})
    for (u, t, dg) in (("U", "N", "U"), ("L", "T", "N"), ("U", "T", "U")):
        for (m, k) in ((26, 8), (65, 16), (129, 32), (379, 32), (379, 27), (11, 3)):
            kern.append({"kind": "trmm", "uplo": u, "trans": t, "diag": dg, "m": m, "k": k, "seed": r.randrange(1 << 30)})
    for m in (183, 231, 299, 378, 399):
        for k in (1, 5, 12, 23):
            kern.append({"kind": "gemvT", "m": m, "n": k, "seed": r.randrange(1 << 30)})
            kern.append({"kind": "gemvNs", "m": m, "n": k, "seed": r.randrange(1 << 30)})
    for (m, k) in ((10, 9), (5, 5), (26, 26)):
        kern.append({"kind": "ger", "m": m, "n": k, "seed": r.randrange(1 << 30)})
    kern.append({"kind": "scal", "m": 50, "n": 1, "seed": r.randrange(1 << 30)})
    for (n, k) in ((41, 24), (232, 24), (184, 24), (376, 24), (41, 5)):
        kern.append({"kind": "syr2k", "n": n, "k": k, "seed": r.randrange(1 << 30)})
    doc["kern"] = kern
    json.dump(doc, open(os.path.join(out, "eig3.json"), "w"))
    print(f"eig3: {len(doc['steqr'])} steqr, {len(sc)} scale, {len(fg)} gemm trees, {len(kern)} kernel records", file=sys.stderr)


if __name__ == "__main__":
    main()
