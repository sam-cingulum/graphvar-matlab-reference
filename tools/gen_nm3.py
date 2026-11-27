"""Inputs for null-model battery 3 (drivers/ref_nm2.m run on <out>/nm2.json). Writes <out>/nm2.json.

Battery 3 separates rounding variants of randperm(n, k)'s regime-B selection step (n/4 <= k < n), which battery 2
left open. Each record crafts the stream so that one selection draw u sits exactly on a value where the variants
take different branches:
  qorder  V = 1 - u equals the smaller of q*(top/N) and (q*top)/N at the first step where the two differ, so
          "while q > V" stops at that step under one update order and goes on under the other
  cmp     u = fl(1 - q) for a q < 1/2 whose 1 - q rounds up, so "q > 1 - u" and "1 - q < u" disagree
The other draws are 1/2. Doubles are any multiple of 2^-53 in (0, 1): u = (a*2^26 + b)/2^53 with a = w1 >> 5 and
b = w2 >> 6, so the two state words are a << 5 and b << 6, untempered. Records: {"case", "n", "k", "u", "state"} as in
battery 2, plus "family", "pos" (1-based draw position) and "note". words: a crafting check with nonzero low words.
Synthetic only. Needs no other input.
"""
import json, os, sys
from fractions import Fraction


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


def words_of(u):
    m = Fraction(u) * 2 ** 53
    assert m.denominator == 1 and 0 < m < 2 ** 53, u
    m = int(m); a, b = m >> 26, m & (2 ** 26 - 1)
    return [a << 5, b << 6]


def state_for(us):
    ws = []
    for u in us: ws += words_of(u)
    st = [untemper(w) for w in ws]
    assert [temper(s) for s in st] == ws
    return st


def on_grid(x):
    return (Fraction(x) * 2 ** 53).denominator == 1


def qseq(n, k, order):
    top, N = n - k, n
    q = top / N; out = [q]
    while top > 0 and N > 1:
        top -= 1; N -= 1
        q = (q * top) / N if order == 'a' else q * (top / N)
        out.append(q)
    return out


def main(out):
    os.makedirs(out, exist_ok=True)
    craft = []
    shapes = [(n, k) for n in range(5, 61) for k in range(max(2, -(-n // 4)), n) if 4 * k >= n]
    for n, k in shapes:
        qa, qb = qseq(n, k, 'a'), qseq(n, k, 'b')
        for s in range(1, min(len(qa), len(qb))):
            if qa[s] != qb[s]:
                V = min(qa[s], qb[s])
                if 0 < V < 1 and on_grid(V) and on_grid(1 - V) and 1.0 - (1.0 - V) == V:
                    u = [0.5] * (2 * k); u[0] = 1.0 - V
                    craft.append({"case": f"nm3-qorder-n{n}-k{k}-s{s}", "family": "qorder", "n": n, "k": k, "pos": 1,
                                  "note": f"step {s}: (q*top)/N = {qa[s]!r}, q*(top/N) = {qb[s]!r}", "u": u})
                break
    for n, k in shapes:
        q = (n - k) / n
        if q < 0.5:
            fl = 1.0 - q
            if Fraction(fl) > 1 - Fraction(q) and on_grid(fl) and fl < 1:
                u = [0.5] * (2 * k); u[0] = fl
                craft.append({"case": f"nm3-cmp-n{n}-k{k}", "family": "cmp", "n": n, "k": k, "pos": 1,
                              "note": f"q = {q!r}, fl(1 - q) = {fl!r} rounds up", "u": u})
    for c in craft:
        c["state"] = state_for(c["u"])
    wu = [0.5, 1 / 3, 0.1, 1 - 2 ** -53, 2 ** -53]
    wu = [float(Fraction(x).limit_denominator(2 ** 53)) for x in wu]
    wu = [x for x in wu if on_grid(x)]
    words = [{"case": "words-check-53bit", "u": wu, "state": state_for(wu)}]
    json.dump({"craft": craft, "words": words, "seeds": []}, open(os.path.join(out, "nm2.json"), "w"))
    print(f"nm3: {len(craft)} crafted ({sum(c['family'] == 'qorder' for c in craft)} qorder, "
          f"{sum(c['family'] == 'cmp' for c in craft)} cmp), words {len(wu)}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
