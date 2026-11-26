"""Inputs for eig probe 5 (drivers/probe_eig5.m). Writes <out>/eig5.json:
  panel   [{"case", "n", "A"}]: gauss matrices, n = 256..999, for dormtr's column-panel test (MKL applies Q to
          Z in column panels; the probe compares panel widths against eig's V)
  orm2l   [{"case", "n", "A", "Z"}]: zero-heavy symmetric A and zero-heavy Z (signed zeros), n = 2..47, for
          MKL dormtr's unblocked path (dorm2l) word for word, signed zeros included
  stedc   [{"case", "kind", "d", "e"}]: tridiagonals for MKL dstedc('I'), n = 26..200 (divide and conquer)
  kern5   [{"kind", ...shape, "seed"}]: random-word records (MATLAB rng(seed)) for DLARFB's kernels at the
          back-transform's shapes: dgemm 'T','N' and 'N','T', dtrmm (R,U,N,U), (R,L,T,N), (R,U,T,U)
  larft5  [{"rows", "ib", "seed"}]: random-word records for MKL dlarft('B','C')
Synthetic only (seeded). Needs no other input.
"""
import json, os, random, struct, sys

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


def zword(rng, pz):
    if rng.random() < pz:
        return 0.0 if rng.random() < 0.5 else -0.0
    return rng.gauss(0, 1) * 2.0 ** rng.randint(-20, 20)


def main(out):
    rng = random.Random(5505)
    doc = {}
    # ---- dormtr column panels ------------------------------------------------------------------
    pan = []
    for n in (256, 257, 258, 264, 280, 288, 300, 320, 350, 379, 384, 385, 400, 450, 512, 513, 600, 768, 999):
        pan.append({"case": f"eig5-panel-gauss-n{n:03d}", "n": n, "A": matl(n, gauss_sym(n, rng))})
    doc["panel"] = pan
    # ---- dorm2l with signed zeros --------------------------------------------------------------------
    o2 = []
    for t in range(48):
        n = 2 + t % 46
        pz = (0.5, 0.8, 0.95)[t % 3]
        pz2 = (0.4, 0.8)[t % 2]
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                A[i][j] = A[j][i] = zword(rng, pz)
        Z = [[zword(rng, pz2) for _ in range(n)] for _ in range(n)]
        o2.append({"case": f"eig5-orm2l-{t:02d}-n{n:03d}", "n": n, "A": matl(n, A), "Z": matl(n, Z)})
    doc["orm2l"] = o2
    # ---- dstedc tridiagonals -----------------------------------------------------------------------------
    st = []
    sizes = (26, 27, 30, 40, 48, 49, 50, 51, 63, 64, 75, 100, 101, 128, 129, 150, 200)
    for n in sizes:
        d = [rng.gauss(0, 1) for _ in range(n)]
        e = [rng.gauss(0, 1) for _ in range(n - 1)]
        st.append({"case": f"eig5-stedc-gauss-n{n:03d}", "kind": "gauss", "d": vec(d), "e": vec(e)})
    for n in (30, 50, 64, 100, 129):
        # graph-like: integer diagonal, some exact zero off-diagonals (splits), repeated values
        d = [float(rng.randint(0, 6)) for _ in range(n)]
        e = [0.0 if rng.random() < 0.08 else -float(rng.randint(1, 3)) for _ in range(n - 1)]
        st.append({"case": f"eig5-stedc-graph-n{n:03d}", "kind": "graph", "d": vec(d), "e": vec(e)})
    for n in (40, 100):
        # clustered eigenvalues (deflation in the merges)
        d = [1.0 + 1e-9 * rng.random() for _ in range(n)]
        e = [1e-7 * rng.gauss(0, 1) for _ in range(n - 1)]
        st.append({"case": f"eig5-stedc-cluster-n{n:03d}", "kind": "cluster", "d": vec(d), "e": vec(e)})
    doc["stedc"] = st
    # ---- DLARFB kernels at back-transform shapes --------------------------------------------------------
    r = random.Random(15500)
    kern = []
    for k in (26, 32):
        for nc in (1, 3, 5, 13, 123, 256):
            for L in (32, 200, 256, 257, 346, 600):
                # W(nc x k) += C1(L x nc)' * V1(L x k)
                kern.append({"kind": "gemm", "ta": "T", "tb": "N", "m": nc, "n": k, "k": L, "seed": r.randrange(1 << 30)})
                # C1(L x nc) -= V1(L x k) * W(nc x k)'
                kern.append({"kind": "gemm", "ta": "N", "tb": "T", "m": L, "n": nc, "k": k, "seed": r.randrange(1 << 30)})
            for (u, t, dg) in (("U", "N", "U"), ("L", "T", "N"), ("U", "T", "U")):
                kern.append({"kind": "trmm", "uplo": u, "trans": t, "diag": dg, "m": nc, "k": k, "seed": r.randrange(1 << 30)})
    doc["kern5"] = kern
    lf = []
    for ib in (8, 16, 17, 24, 31, 32):
        for rows in (ib, ib + 1, 48, 64, 100, 200, 256, 257, 378, 600):
            if rows >= ib:
                lf.append({"rows": rows, "ib": ib, "seed": r.randrange(1 << 30)})
    doc["larft5"] = lf
    os.makedirs(out, exist_ok=True)
    json.dump(doc, open(os.path.join(out, "eig5.json"), "w"))
    print(f"eig5: {len(pan)} panel, {len(o2)} orm2l, {len(st)} stedc, {len(kern)} kern5, {len(lf)} larft5", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
