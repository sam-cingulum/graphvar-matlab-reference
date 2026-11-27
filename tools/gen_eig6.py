"""Inputs for eig probe 6 (drivers/probe_eig6.m). Writes <out>/eig6.json and <out>/eig6/*.bin.

Arrays with more than 4096 words are written as big-endian IEEE doubles (column-major) to <out>/eig6/<name>.bin
and appear in the JSON as {"size": [r, c], "bin": "<name>.bin"}; smaller ones as {"size", "hex"}.

  full     [{"case", "kind", "n", "A"}]: symmetric matrices with n = 600..1000 for [V, D] = eig(A) with every
           stage's output (dsytrd, dstedc, dormtr), across the 128/256/512 block boundaries and n = 1000
  larft6   [{"case", "V", "tau"}]: zero-heavy V (both signs of zero) and zero tau for MKL dlarft('B','C')
  gemmtn0  [{"case", "A", "B"}]: zero-heavy inputs for dgemm('T','N') with beta = 0 (dlarft's cross products)
  gemmnn6  [{"m", "n", "k", "seed"}]: random-word dgemm('N','N') shapes with inner dimension 256..1000
  fprev6   [{"m", "n", "L", "rows", "cols"}]: dgemm('N','N') FPRev shapes with L = 256..1000
  trmm6    [{"uplo", "trans", "diag", "m", "k", "ldb", "seed"}]: dtrmm side 'R' with m = 1..5 rows and ldb = m or
           a large leading dimension
Synthetic only (seeded). Needs no other input.
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


class Out:
    def __init__(self, out):
        self.dir = os.path.join(out, "eig6")
        os.makedirs(self.dir, exist_ok=True)

    def arr(self, name, rows, cols, colmajor):
        """colmajor: flat list, column-major, rows*cols words"""
        assert len(colmajor) == rows * cols
        if rows * cols > 4096:
            fn = name + ".bin"
            with open(os.path.join(self.dir, fn), "wb") as f:
                f.write(struct.pack(">%dd" % len(colmajor), *colmajor))
            return {"size": [rows, cols], "bin": fn}
        return {"size": [rows, cols], "hex": [hx(t) for t in colmajor]}


def sym_flat(n, upper):
    """upper(i, j) for i <= j, mirrored; column-major flat list"""
    F = [0.0] * (n * n)
    for j in range(n):
        for i in range(j + 1):
            F[i + j * n] = F[j + i * n] = upper(i, j)
    return F


def zword(rng, pz, lo=-5, hi=5):
    if rng.random() < pz:
        return 0.0 if rng.random() < 0.5 else -0.0
    return rng.gauss(0, 1) * 2.0 ** rng.randint(lo, hi)


def main(out):
    o = Out(out)
    doc = {}
    # ---- full eig with every stage's output --------------------------------------------------------
    full = []
    def add(kind, n, F):
        name = f"eig6-full-{kind}-n{n:04d}"
        full.append({"case": name, "kind": kind, "n": n, "A": o.arr(name + "__A", n, n, F)})
    for n in (600, 640, 700, 767, 768, 769, 770, 772, 800, 896, 999, 1000):
        r = random.Random(66000 + n)
        add("gauss", n, sym_flat(n, lambda i, j: r.gauss(0.0, 1.0)))
    for n in (600, 1000):
        r = random.Random(66100 + n)
        add("bin30", n, sym_flat(n, lambda i, j: 0.0 if i == j else (1.0 if r.random() < 0.3 else 0.0)))
    for n in (769, 1000):
        r = random.Random(66200 + n)
        add("wei30", n, sym_flat(n, lambda i, j: 0.0 if i == j else (r.random() if r.random() < 0.3 else 0.0)))
    # two disconnected weighted blocks (zero reflectors, exact zeros in Z)
    n = 640; r = random.Random(66300 + n)
    add("twoblocks", n, sym_flat(n, lambda i, j: 0.0 if (i == j or (i < n // 2) != (j < n // 2)) else
                                 (r.random() if r.random() < 0.3 else 0.0)))
    # weighted star plus a sparse ring (many repeated eigenvalues: deflation in the merges)
    n = 700; r = random.Random(66400 + n)
    star = [r.random() for _ in range(n)]
    add("starring", n, sym_flat(n, lambda i, j: 0.0 if i == j else
                                (star[j] if i == 0 else (1.0 if j == i + 1 else 0.0))))
    doc["full"] = full
    # ---- zero-heavy dlarft ---------------------------------------------------------------------------
    R = random.Random(66500)
    lf = []
    def addlf(rows, k, pz, pt):
        V = [zword(R, pz) for _ in range(rows * k)]
        t = [(0.0 if R.random() < 0.5 else -0.0) if R.random() < pt else 1 + R.random() for _ in range(k)]
        name = f"eig6-larft-{len(lf):03d}-r{rows:04d}-k{k:02d}"
        lf.append({"case": name, "V": o.arr(name + "__V", rows, k, V), "tau": o.arr(name + "__tau", k, 1, t)})
    for _ in range(400):
        k = R.randint(1, 32)
        rows = R.choice([k, k + 1]) if k > 16 else R.randint(k, k + 60)
        addlf(rows, k, R.choice([0.3, 0.7, 0.95]), R.choice([0.0, 0.3]))
    for k in (8, 16, 17, 24, 32):
        for rows in (k + 2, 100, 257, 700, 999):
            for pz in (0.7, 0.95):
                addlf(rows, k, pz, 0.3)
    doc["larft6"] = lf
    # ---- zero-heavy dgemm('T','N'), beta = 0 -------------------------------------------------------------
    gz = []
    for t in range(240):
        M = R.choice([1, 2, 3, 4, 8, 12, 15, 16])
        N = R.choice([1, 2, 4, 7, 8, 15, 16])
        K = R.choice([1, 2, 3, 5, 8, 9, 17, 40, 100, 257, 600])
        pz = R.choice([0.5, 0.8, 0.95, 1.0])
        A = [zword(R, pz) for _ in range(K * M)]
        B = [zword(R, pz) for _ in range(K * N)]
        name = f"eig6-gemmtn0-{t:03d}-M{M:02d}-N{N:02d}-K{K:03d}"
        gz.append({"case": name, "A": o.arr(name + "__A", K, M, A), "B": o.arr(name + "__B", K, N, B)})
    doc["gemmtn0"] = gz
    # ---- dgemm('N','N') random words: dstedc's DLAED3 shapes past inner dimension 256 -----------------
    doc["gemmnn6"] = [{"m": m, "n": nn, "k": k, "seed": 66600 + i} for i, (m, nn, k) in enumerate([
        (256, 512, 256), (250, 500, 257), (257, 513, 257), (260, 520, 260), (300, 600, 300), (320, 640, 320),
        (350, 700, 350), (383, 767, 384), (384, 768, 384), (384, 769, 385), (400, 800, 400), (448, 896, 448),
        (499, 999, 500), (500, 1000, 500), (500, 960, 480), (500, 600, 300), (384, 700, 350), (128, 256, 512),
        (129, 257, 513), (37, 26, 520), (13, 8, 1000), (25, 9, 700), (12, 4, 600), (11, 3, 600)])]
    # ---- dgemm('N','N') FPRev, long inner sums --------------------------------------------------------------
    fp = [{"m": 13, "n": 5, "L": L, "rows": [1, 6, 12, 13], "cols": [1, 4, 5]}
          for L in (256, 257, 300, 301, 384, 385, 400, 511, 512, 513, 520, 640, 768, 1000)]
    fp += [{"m": 100, "n": 40, "L": L, "rows": [1, 12, 13, 96, 97, 100], "cols": [1, 4, 5, 37, 40]} for L in (300, 520)]
    doc["fprev6"] = fp
    # ---- dtrmm side 'R' with few rows and a large leading dimension --------------------------------------------
    tr = []
    for (u, t, dg) in (("U", "N", "U"), ("L", "T", "N"), ("U", "T", "U")):
        for m in (1, 2, 3, 4, 5):
            for k in (8, 9, 16, 17, 31, 32):
                for ldb in (m, 770):
                    tr.append({"uplo": u, "trans": t, "diag": dg, "m": m, "k": k, "ldb": ldb, "seed": R.randrange(1 << 30)})
    doc["trmm6"] = tr
    json.dump(doc, open(os.path.join(out, "eig6.json"), "w"))
    print(f"eig6: {len(full)} full, {len(lf)} larft6, {len(gz)} gemmtn0, {len(doc['gemmnn6'])} gemmnn6, "
          f"{len(fp)} fprev6, {len(tr)} trmm6", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
