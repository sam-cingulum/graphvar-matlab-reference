"""Inputs for eig probe 6c: drivers/probe_eig6.m run on <out>/eig6.json with only its larft6 set filled.

Probe 6c pins MKL dlarft('B','C')'s signed zeros in column 1 when K = rows - k = 0, so the sum in P2-04 §5.3
step 2b is empty for i = 1: tau(1) = +0 or -0 (and, for contrast, nonzero tau(1) with V(1, c) = +-0), for
k = 2..16 and a few wider k, with K = 0 and, for contrast, K = 1. V is dense N(0,1) or zero-heavy (both
signs of zero); the other tau are in [1, 2) or +-0. Records: {"case", "V", "tau"} as in probe 6.
Synthetic only (seeded). Needs no other input.
"""
import json, os, random, struct, sys

hx = lambda x: struct.pack(">d", x).hex()


def zword(rng, pz):
    if rng.random() < pz:
        return 0.0 if rng.random() < 0.5 else -0.0
    return rng.gauss(0, 1)


def main(out):
    os.makedirs(os.path.join(out, "eig6"), exist_ok=True)
    R = random.Random(66600)
    lf = []

    def add(rows, k, V, t, tag):
        name = f"eig6c-larft-{len(lf):03d}-r{rows:03d}-k{k:02d}-{tag}"
        lf.append({"case": name, "V": {"size": [rows, k], "hex": [hx(x) for x in V]},
                   "tau": {"size": [k, 1], "hex": [hx(x) for x in t]}})

    for k in list(range(2, 17)) + [17, 20, 24, 32]:
        for K in (0, 1):
            rows = k + K
            for t1 in (0.0, -0.0):
                for rep in range(8 if K == 0 else 3):
                    pz = (0.0, 0.0, 0.5, 0.9)[rep % 4]
                    V = [zword(R, pz) for _ in range(rows * k)]
                    t = [t1] + [((0.0 if R.random() < 0.5 else -0.0) if R.random() < (0.0 if rep < 4 else 0.3)
                                 else 1 + R.random()) for _ in range(k - 1)]
                    add(rows, k, V, t, f"K{K}-t1{'p' if str(t1)[0] != '-' else 'm'}0-z{int(pz * 10)}")
            # nonzero tau(1) with row K+1 of V all +-0 (the empty-sum start for tau != 0)
            for rep in range(3):
                V = [zword(R, 0.3) for _ in range(rows * k)]
                for c in range(1, k):
                    V[K + c * rows] = 0.0 if R.random() < 0.5 else -0.0
                t = [1 + R.random()] + [1 + R.random() for _ in range(k - 1)]
                add(rows, k, V, t, f"K{K}-tnz-vrow0")
    doc = {"full": [], "larft6": lf, "gemmtn0": [], "gemmnn6": [], "fprev6": [], "trmm6": []}
    json.dump(doc, open(os.path.join(out, "eig6.json"), "w"))
    print(f"eig6c: {len(lf)} larft records", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
