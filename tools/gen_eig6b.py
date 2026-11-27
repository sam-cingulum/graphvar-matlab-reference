"""Inputs for eig probe 6b: drivers/probe_eig6.m run on <out>/eig6.json and <out>/eig6/*.bin.

Probe 6b fills the gap between probe 1 (n <= 400) and probe 6 (n >= 600) with full-output cases:
[V, D] = eig(A), d = eig(A), and every stage's output (MKL dsytrd d0, e0, tau, Ar; MKL dstedc w, Z), at
n = 401, 450, 512, 513, 514 and 599, each for gauss, bin30 and wei30 (18 cases). Only the "full" set is
filled; the other probe-6 sets are empty lists, so the driver writes the ilaenv record and the 18 full records.

Same file format as tools/gen_eig6.py: arrays with more than 4096 words are big-endian IEEE doubles
(column-major) in <out>/eig6/<name>.bin, referenced from the JSON as {"size": [r, c], "bin": "<name>.bin"}.
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
    """upper(i, j) for i <= j, called column by column (j outer, i inner), mirrored; column-major flat list"""
    F = [0.0] * (n * n)
    for j in range(n):
        for i in range(j + 1):
            F[i + j * n] = F[j + i * n] = upper(i, j)
    return F


SIZES = (401, 450, 512, 513, 514, 599)


def main(out):
    o = Out(out)
    full = []

    def add(kind, n, F):
        name = f"eig6b-full-{kind}-n{n:04d}"
        full.append({"case": name, "kind": kind, "n": n, "A": o.arr(name + "__A", n, n, F)})

    for n in SIZES:
        r = random.Random(66700 + n)
        add("gauss", n, sym_flat(n, lambda i, j: r.gauss(0.0, 1.0)))
        r = random.Random(66800 + n)
        add("bin30", n, sym_flat(n, lambda i, j: 0.0 if i == j else (1.0 if r.random() < 0.3 else 0.0)))
        r = random.Random(66900 + n)
        add("wei30", n, sym_flat(n, lambda i, j: 0.0 if i == j else (r.random() if r.random() < 0.3 else 0.0)))
    doc = {"full": full, "larft6": [], "gemmtn0": [], "gemmnn6": [], "fprev6": [], "trmm6": []}
    json.dump(doc, open(os.path.join(out, "eig6.json"), "w"))
    print(f"eig6b: {len(full)} full", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
