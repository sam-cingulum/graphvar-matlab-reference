"""Inputs for eig probe 2 (drivers/probe_eig2.m). Writes <out>/eig2.json:
  ilaenv_n       sizes for the ilaenv queries
  fprev_symv_n   sizes for the dsymv('U') FPRev trees
  fprev_dot_n    sizes for the ddot FPRev trees
  kernel         [{"n", "alpha", "x", "y"}]: random vectors for the kernel
                 records; the matrix is the eig.json "gauss" case of the same n
Needs no other input; probe_eig2 also reads eig.json from tools/gen_eig.py.
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()
vec = lambda v: {"size": [len(v), 1], "hex": [hx(t) for t in v]}

GAUSS_N = [1, 2, 3, 4, 5, 8, 16, 24, 25, 26, 31, 32, 33, 48, 63, 64, 65, 100, 128, 129, 200, 256, 300, 379, 400]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    kernel = []
    for n in GAUSS_N:
        r = random.Random(12000 + n)
        x = [r.gauss(0.0, 1.0) for _ in range(n)]
        y = [r.gauss(0.0, 1.0) for _ in range(n)]
        alpha = r.uniform(1.0, 2.0)
        kernel.append({"n": n, "alpha": {"size": [1, 1], "hex": [hx(alpha)]}, "x": vec(x), "y": vec(y)})
    doc = {
        "ilaenv_n": list(range(1, 131)) + [160, 192, 200, 256, 300, 379, 400],
        "fprev_symv_n": list(range(2, 49)) + [63, 64, 65, 96, 127, 128, 129, 200, 256, 400],
        "fprev_dot_n": list(range(2, 131)) + [200, 256, 300, 379, 400],
        "kernel": kernel,
    }
    json.dump(doc, open(os.path.join(out, "eig2.json"), "w"))
    print(f"eig2: {len(kernel)} kernel records, {len(doc['fprev_symv_n'])} symv trees, {len(doc['fprev_dot_n'])} dot trees", file=sys.stderr)


if __name__ == "__main__":
    main()
