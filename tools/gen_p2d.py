"""MM7 battery: distinct-operand transposed products C = A.'*B (A is K x M, B is K x N), word for word.

Writes <out>/tn.json: {"tn": [{"case", "M", "N", "K", "seed"}]}. No matrix words are stored: both A and B come
from one Park-Miller stream that MATLAB (drivers/ref_p2d.m) and Python reproduce exactly in double arithmetic:
    s <- mod(16807 * s, 2147483647),  x = (s - 1073741823.5) / 1073741824
(every step is exact in binary64). Starting from s = seed, the first K*M values fill A column by column, the
next K*N fill B. Sets:
  small    M, N = 1..13, K = 1..20 (every shape)
  rowvec   M = 1 (row vector times matrix), N in {1, 2, 3, 4, 5, 8, 13, 16}, K = 1..64, three seeds
  colvec   N = 1, M in {2, 3, 4, 5, 8, 13, 16}, K = 1..64
  longk    (M, N) in a few shapes, K = 255..2000 (blocking along K)
  edge     vectors with the other dimension w = 13..17, K = 1..20, three seeds (the w boundary)
Synthetic only. Needs no other input.
"""
import json, os, sys


def main(out):
    os.makedirs(out, exist_ok=True)
    tn = []
    seed = [1]
    def add(kind, M, N, K):
        seed[0] = (seed[0] * 48271) % 2147483647
        tn.append({"case": f"p2d-{kind}-M{M:02d}-N{N:02d}-K{K:04d}-{len(tn):05d}", "M": M, "N": N, "K": K, "seed": seed[0]})
    for M in range(1, 14):
        for N in range(1, 14):
            for K in range(1, 21):
                add("small", M, N, K)
    for N in (1, 2, 3, 4, 5, 8, 13, 16):
        for K in range(1, 65):
            for _ in range(3):
                add("rowvec", 1, N, K)
    for M in (2, 3, 4, 5, 8, 13, 16):
        for K in range(1, 65):
            add("colvec", M, 1, K)
    for (M, N) in ((12, 4), (13, 5), (24, 8), (40, 40), (5, 3), (1, 2), (2, 1)):
        for K in (255, 256, 257, 258, 300, 383, 384, 385, 511, 512, 513, 514, 600, 767, 768, 769, 1000, 1024, 1025, 2000):
            add("longk", M, N, K)
    for w in (13, 14, 15, 16, 17):
        for K in range(1, 21):
            for _ in range(3):
                add("edge", 1, w, K)
                add("edge", w, 1, K)
    json.dump({"tn": tn}, open(os.path.join(out, "tn.json"), "w"))
    print(f"p2d: {len(tn)} A.'*B records", file=sys.stderr)


def stream(seed, count):
    """the same values the MATLAB driver draws (for the analysis side)"""
    s = float(seed); v = []
    for _ in range(count):
        s = (16807.0 * s) % 2147483647.0
        v.append((s - 1073741823.5) / 1073741824.0)
    return v


if __name__ == "__main__":
    main(sys.argv[1])
