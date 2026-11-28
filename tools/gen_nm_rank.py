"""Inputs for the signed null-model rank-failure probe (drivers/ref_nm_rank.m).
Writes <out>/nm_rank.json.

  graphs  [{"name", "dir", "sign", "n", "W"}]: W column-major as 16-hex binary64 words.
  calls   [{"case", "graph", "seed", "bin_swaps", "wei_freq"}]: the driver resets
          rng(seed, 'twister'), calls BCT 2019-03-03 null_model_und_sign, then
          records the next four rand draws after either success or a caught error.

The eight-node asymmetric signed case is the suspected failure; the four-node
asymmetric control is expected to complete. Both use bin_swaps=1 at wei_freq=1
and 0.1. Synthetic only; no external inputs.
"""
import json
import os
import struct
import sys


hx = lambda x: struct.pack(">d", x).hex()


def main(out):
    os.makedirs(out, exist_ok=True)
    rank_rows = [
        [0, .5, 0, 0, -.5, 0, 1, 0],
        [-.5, 0, 1, 0, 1, -.5, 0, 1],
        [.5, 1, 0, 0, .5, 0, 0, 0],
        [-.5, 0, -.5, 0, 1, .5, 0, 1],
        [1, 0, 0, 0, 0, -.5, -.5, 0],
        [1, 0, .5, 1, 0, 0, 0, .5],
        [.5, .5, 0, 0, .5, 0, 0, 0],
        [0, 0, .5, .5, 1, 0, .5, 0],
    ]
    control_rows = [[0.0] * 4 for _ in range(4)]
    # MATLAB entries W(1,2)=W(1,3)=W(2,4)=1, hence zero-based rows below.
    control_rows[0][1] = control_rows[0][2] = control_rows[1][3] = 1.0

    graphs = []
    for name, rows in (("rank-failure-asym-n8", rank_rows), ("control-asym-n4", control_rows)):
        n = len(rows)
        assert all(len(row) == n for row in rows)
        assert all(rows[i][i] == 0 for i in range(n))
        assert any(rows[i][j] != rows[j][i] for i in range(n) for j in range(n))
        W = [rows[i][j] for j in range(n) for i in range(n)]
        graphs.append({"name": name, "dir": True, "sign": any(x < 0 for x in W), "n": n,
                       "W": [hx(float(x)) for x in W]})

    calls = []
    for graph, seed in (("rank-failure-asym-n8", 2), ("control-asym-n4", 1)):
        for wei_freq in (1.0, 0.1):
            calls.append({"case": f"nm-rank-{graph}-wf{wei_freq}", "graph": graph, "seed": seed,
                          "bin_swaps": 1, "wei_freq": wei_freq})
    assert len(calls) == 4
    path = os.path.join(out, "nm_rank.json")
    with open(path, "w") as f:
        json.dump({"graphs": graphs, "calls": calls}, f)
        f.write("\n")
    print("nm_rank: 2 asymmetric graphs, 4 calls", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
