# G1 firstblind: result

**PASS.** Every approved job matches the frozen forecast (cingulum-cpm 9dda105) word for word:
**459/459 cases, 12,327,370/12,327,370 words** (construct 359/359 cases, 5,610,015 words; measures 83/83
cases, 6,261,932 words; matrix multiply 17/17 cases, 455,423 words). No MATLAB warnings.

Scored once per job by the kit's `strict_audit.py` (frozen with the kit), exit status 0 for each, and checked
independently by the spec writer (canonical JSON per case and item, `notes` dropped): every item in the
forecast equals MATLAB's. MATLAB's `AtB` items (91, outside the claim, P2-00 MM7) are not in the forecast.

## Runs (ref-g1.yml, kit=firstblind/g1, harness 3b82a71, MATLAB R2026b, maxNumCompThreads(1))
| run | job | CPU |
|---|---|---|
| 45899862938 | ubuntu-22.04 draw 6 | AMD EPYC 7763 |
| 45899862938 | ubuntu-24.04 draw 4 | AMD EPYC 7763 |
| 45899925108 | ubuntu-22.04 draw 2 | AMD EPYC 9V45 |
| 45899925108 | ubuntu-22.04 draw 5 | AMD EPYC 9V45 |
| 45899925108 | ubuntu-22.04 draw 6 | AMD EPYC 7763 |
| 45899925108 | ubuntu-24.04 draw 3 | AMD EPYC 7763 |
| 45899925108 | ubuntu-24.04 draw 4 | AMD EPYC 9V45 |

The other 17 jobs drew unapproved CPUs and stopped at "Approved CPU only", as designed.
All 7 approved jobs' outputs are byte-identical:
    506ac31b25c076eb39d1749a6b236dc5bee1419588c274f0dc7660e5785ac587  construct.json
    ed63b5f5a98dfd44c304b6c29dc09a420670dda5854f441168e6af27faa4c311  measures.json
    4af61c1c3f5123094559e71102829c3e1207f4f1cbd0b7549592677a968897a0  gemm.jsonl

The file bytes differ from the forecast's files only in JSON layout (whitespace, key order); the audit compares
parsed values.
