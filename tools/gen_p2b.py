#!/usr/bin/env python3
"""Phase-2b inputs: one-operand Gram products X.'*X and X*X.', distinct-operand
X.'*Y, and short inner products, for checking the P2-00 MM5/MM6/MM7 rules.

- square: Gaussian A, B (n x n) for n in SQ_N.
- rect:   Gaussian X, Y (r x w) for r in RECT_R, w in RECT_W (short K, r < 256 mostly).
- inner:  Gaussian column vectors a, b of every length 1..400.
Python's random with fixed seeds; every value is an exact binary64 word.
"""
import json, os, random, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "cases/p2b"
SQ_N = sorted(set([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 23, 24, 25, 31, 32, 33, 44, 47, 48, 49,
                   63, 64, 65, 90, 96, 116, 128, 129, 192, 200, 256, 257, 379]))
RECT_R = [3, 8, 16, 31, 50, 64, 65, 100, 128, 200, 255, 256, 300]
RECT_W = [2, 5, 12, 13, 40, 100]
INNER_MAX = 400


def hx(x):
    return struct.pack(">d", float(x)).hex()


def mat(rows):
    r, c = len(rows), len(rows[0])
    return {"size": [r, c], "hex": [hx(rows[i][j]) for j in range(c) for i in range(r)]}


def gauss(rnd, r, c):
    return [[rnd.gauss(0, 1) for _ in range(c)] for _ in range(r)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(202611273)
    out = {"square": [], "rect": [], "inner": []}
    for n in SQ_N:
        out["square"].append({"n": n, "A": mat(gauss(rnd, n, n)), "B": mat(gauss(rnd, n, n))})
    for r in RECT_R:
        for w in RECT_W:
            out["rect"].append({"r": r, "w": w, "X": mat(gauss(rnd, r, w)), "Y": mat(gauss(rnd, r, w))})
    for L in range(1, INNER_MAX + 1):
        out["inner"].append({"L": L, "a": mat(gauss(rnd, L, 1)), "b": mat(gauss(rnd, L, 1))})
    with open(os.path.join(OUT, "gram.json"), "w") as f:
        json.dump(out, f)
    print(f"square {len(SQ_N)}; rect {len(RECT_R) * len(RECT_W)}; inner {INNER_MAX}")


if __name__ == "__main__":
    main()
