"""Generate the phase-2 inputs: probes of MATLAB's real matrix multiply (MKL dgemm).

Writes cases/p2/gemm.json. Pure Python, fixed seeds, no patient data.
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


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"tree": [], "fma": [], "zero": [], "rand": [], "mpow": []}
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
