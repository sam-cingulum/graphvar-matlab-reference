"""Inputs for the eig probe (which LAPACK path MATLAB's eig takes on real
symmetric matrices, and its stages). Writes <out>/eig.json:
{"cases": [{"case", "kind", "n", "A": {"size", "hex"}}]}. Every A is exactly
symmetric (upper triangle mirrored), except kind "asym1ulp".
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def mat(n, get):
    return {"size": [n, n], "hex": [hx(get(i, j)) for j in range(n) for i in range(n)]}


def sym(n, upper):
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            M[i][j] = M[j][i] = upper(i, j)
    return M


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cases = []
    sizes = [1, 2, 3, 4, 5, 8, 16, 24, 25, 26, 31, 32, 33, 48, 63, 64, 65, 100, 128, 129, 200, 256, 300, 379, 400]

    def add(kind, M):
        n = len(M)
        cases.append({"case": f"eig-{kind}-n{n:03d}", "kind": kind, "n": n, "A": mat(n, lambda i, j: M[i][j])})

    for n in sizes:
        r = random.Random(7000 + n)
        add("gauss", sym(n, lambda i, j: r.gauss(0.0, 1.0)))
        r = random.Random(8000 + n)
        add("bin30", sym(n, lambda i, j: 0.0 if i == j else (1.0 if r.random() < 0.3 else 0.0)))
        r = random.Random(9000 + n)
        add("wei30", sym(n, lambda i, j: 0.0 if i == j else (r.random() if r.random() < 0.3 else 0.0)))
    for n in (5, 32, 100):
        add("complete", sym(n, lambda i, j: 0.0 if i == j else 1.0))
        add("star", sym(n, lambda i, j: 1.0 if i == 0 and j > 0 else 0.0))
        add("twoblocks", sym(n, lambda i, j: 1.0 if i != j and (i < n // 2) == (j < n // 2) else 0.0))
        add("zero", sym(n, lambda i, j: 0.0))
    r = random.Random(9500)
    M = sym(64, lambda i, j: r.gauss(0.0, 1.0))
    M[3][7] = struct.unpack(">d", (struct.unpack(">Q", struct.pack(">d", M[3][7]))[0] + 1).to_bytes(8, "big"))[0]
    cases.append({"case": "eig-asym1ulp-n064", "kind": "asym1ulp", "n": 64, "A": mat(64, lambda i, j: M[i][j])})
    json.dump({"cases": cases}, open(os.path.join(out, "eig.json"), "w"))
    print(f"{len(cases)} eig cases", file=sys.stderr)


if __name__ == "__main__":
    main()
