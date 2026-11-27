"""Inputs for the second null-model battery (drivers/ref_nm2.m). Writes <out>/nm2.json.

  craft    [{"case", "n", "k", "u"}]: randperm(n, k) on a crafted twister stream: the driver puts the
           untempered words of the doubles u (each u = A / 2^27, so its two words are A << 5 and 0) at the
           start of the state with index 0, so rand returns u(1), u(2), ... exactly
  words    [{"case", "u", "state"}]: the same crafting, checked: state = the 624 state words the driver
           must set; the driver records rand(1, numel(u)) after setting them
  seeds    [{"n", "k", "seed"}]: randperm(n, k) after rng(seed, 'twister'), for n/4 <= k < n
Synthetic only. Needs no other input.
"""
import json, os, sys


def untemper(y):
    y ^= y >> 18
    y ^= (y << 15) & 0xefc60000
    x = y
    for _ in range(5):
        x = y ^ ((x << 7) & 0x9d2c5680)
    y = x
    x = y
    for _ in range(3):
        x = y ^ (x >> 11)
    return x & 0xffffffff


def temper(y):
    y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >> 18
    return y & 0xffffffff


def state_for(us):
    words = []
    for u in us:
        A = round(u * 2 ** 27)
        assert 0 < A < 2 ** 27 and A / 2 ** 27 == u, u
        words += [A << 5, 0]
    st = [untemper(w) for w in words]
    assert [temper(s) for s in st] == words
    return st


def main(out):
    os.makedirs(out, exist_ok=True)
    grid = [1 / 64, 15 / 64, 16 / 64, 26 / 64, 38 / 64, 48 / 64, 63 / 64]
    craft = []
    shapes = [(2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (5, 2), (8, 2), (4, 3), (6, 3), (12, 3), (5, 4), (16, 4),
              (8, 7), (9, 8)]
    for n, k in shapes:
        m = 2 * k
        base = [0.5] * m
        craft.append({"case": f"craft-n{n}-k{k}-base", "n": n, "k": k, "u": base})
        for p in range(m):
            for v in grid:
                u = list(base); u[p] = v
                craft.append({"case": f"craft-n{n}-k{k}-p{p + 1}-{round(v * 64)}of64", "n": n, "k": k, "u": u})
        # strictly increasing and strictly decreasing streams
        inc = [(i + 1) / (m + 1) for i in range(m)]
        inc = [round(x * 64) / 64 or 1 / 64 for x in inc]
        craft.append({"case": f"craft-n{n}-k{k}-inc", "n": n, "k": k, "u": inc})
        craft.append({"case": f"craft-n{n}-k{k}-dec", "n": n, "k": k, "u": inc[::-1]})
    for c in craft:
        c["state"] = state_for(c["u"])
    words = [{"case": "words-check", "u": [0.5, 1 / 64, 63 / 64], "state": state_for([0.5, 1 / 64, 63 / 64])}]
    seeds = [{"n": n, "k": k, "seed": s} for (n, k) in ((8, 2), (8, 3), (12, 5), (16, 5), (10, 9), (6, 4), (4, 2))
             for s in range(1, 101)]
    json.dump({"craft": craft, "words": words, "seeds": seeds}, open(os.path.join(out, "nm2.json"), "w"))
    print(f"nm2: {len(craft)} crafted, {len(seeds)} seeded", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
