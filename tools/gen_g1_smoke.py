"""Small kit in the firstblind G1 layout, to smoke-test the G1 path of the harness.

Writes <out>/{matrices,construct,measures,gemm}.json. Every graph key resolves
inside <out>. Needs cases/phase1 (committed) and cases/p2/gemm.json
(python3 -I tools/gen_p2.py cases/p2). Not a reference battery.
"""
import json, os, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
p1 = json.load(open("cases/phase1/matrices.json"))
mats = {"m_" + k: p1[k] for k in ("s5", "p_s5", "s44", "g_star_7", "g_ring_9", "w_onlypos_6", "w_signed_10", "ties_12")}
cons = [
    {"case": "g1s-construct-01", "matrix": "m_s5", "type": "proportional", "value": "0.5", "mode": "weighted", "neg_rule": "keep", "mask": None, "pvalues": None},
    {"case": "g1s-construct-02", "matrix": "m_s5", "type": "significance", "value": "0.05", "mode": "binary", "neg_rule": "keep", "mask": None, "pvalues": "m_p_s5"},
    {"case": "g1s-construct-03", "matrix": "m_s5", "type": "significance", "value": "0.05", "mode": "binary", "neg_rule": "keep", "mask": None, "pvalues": None},
    {"case": "g1s-construct-04", "matrix": "m_s44", "type": "absolute", "value": "0.3", "mode": "weighted", "neg_rule": "absolute", "mask": [True] * 40 + [False] * 4, "pvalues": None},
    # Optional "rank" key, present on some records only (absent = signed).
    # 05 and 06 repeat reference/p2 p2-construct-0052 and 0056; 07 must equal 01.
    {"case": "g1s-construct-05", "matrix": "m_s5", "type": "proportional", "value": "0.5", "mode": "weighted", "neg_rule": "keep", "mask": None, "pvalues": None, "rank": "magnitude"},
    {"case": "g1s-construct-06", "matrix": "m_ties_12", "type": "proportional", "value": "0.5", "mode": "weighted", "neg_rule": "keep", "mask": None, "pvalues": None, "rank": "magnitude"},
    {"case": "g1s-construct-07", "matrix": "m_s5", "type": "proportional", "value": "0.5", "mode": "weighted", "neg_rule": "keep", "mask": None, "pvalues": None, "rank": "signed"},
]
B = ["M01", "M02", "M05", "M11", "M12", "M13", "M26", "M33", "M62", "M64", "M75"]
W = ["M43", "M44", "M49", "M50", "M51", "M58", "M59", "M60", "M61", "M65", "M69", "M73", "M74"]
meas = [
    {"graph": {"construct": "g1s-construct-01"}, "measures": W},
    {"graph": {"construct": "g1s-construct-02"}, "measures": B},
    {"graph": {"construct": "g1s-construct-03"}, "measures": ["M01"]},
    {"graph": {"construct": "g1s-construct-04"}, "measures": W},
    {"graph": {"matrix": "m_g_star_7"}, "measures": B},
    {"graph": {"matrix": "m_w_onlypos_6"}, "measures": W},
    {"graph": {"matrix": "m_w_signed_10"}, "measures": W},
    {"graph": {"matrix": "m_missing"}, "measures": ["M01"]},
    {"graph": {"construct": "g1s-construct-05"}, "measures": W},
]
for i, m in enumerate(meas, 1):
    m["case"] = f"g1s-measure-{i:02d}"
g = json.load(open("cases/p2/gemm.json"))
gemm = {"rand": [min(g["rand"], key=lambda r: r["n"])], "mpow": [min(g["mpow"], key=lambda r: r["n"])]}
for name, obj in (("matrices", mats), ("construct", cons), ("measures", meas), ("gemm", gemm)):
    json.dump(obj, open(os.path.join(out, name + ".json"), "w"))
print(f"{out}: {len(mats)} matrices, {len(cons)} constructions, {len(meas)} measure cases", file=sys.stderr)
