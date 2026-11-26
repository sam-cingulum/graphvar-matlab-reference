# G1 firstblind: commitment

These four inputs are the G1 graph firstblind kit, copied unchanged (decompressed from the committed
`.gz` files) from cingulumtechnology/cingulum-cpm `tests/parity/g1_graph_firstblind/input/`.

## The freeze (verified by the spec writer before this commit)
- Freeze commit: `9dda105cfa401efb625fcac2546070471b3ccea7` (branch graph/g1-firstblind, PR #13),
  "graph: freeze the G1 firstblind forecast". Commit time 2026-11-27T08:37:13+11:00
  (2026-11-26T21:37:13Z). README status: FROZEN 2026-11-26T21:37:02Z.
- GitHub had the commit by 2026-11-26T21:37:21Z (its first check run, "forecast", started then).
- Forecast built from the kit at `8f7035d`; the inputs are unchanged between 8f7035d and 9dda105.
- Forecast sha256 (uncompressed), recomputed here from the committed `forecast/*.gz`, equal to
  `forecast/SHA256SUMS`:
  - construct.json `89900dda5a5ccfcd4f887143db8f85ab546ef3811690e43a025d2655f820aeaa`
  - measures.json  `4759ffcb13bbbe43ce50a6b434f4da592424ccb33717263acd49b3f7b37bd31a`
  - gemm.jsonl     `2debb31be5cb979809a94f60b3f3d6a1a710068b89146958dd61f74200a5fd93`

## The inputs (sha256 of the files in this folder = the kit MANIFEST.txt)
- matrices.json  `439a7e730f8a496947f4f27c6583ba0a48a75f7aec5934bb6dfce87652f59eed`
- construct.json `b53a20e474a12f0dad318f6183cb3050a9f7eb9fe2d00bc270b1eef0468bdaf5`
- measures.json  `ed939105f8ff060cf8917cef71b2af1028141cdba121dfc41db5ec4aca0d2799`
- gemm.json      `857230ef24f6ab32673ee09bd396878f5022c3623fefad16aae20b64bfd08d69`

## The run
`ref-g1.yml` with `kit=firstblind/g1`: genuine MATLAB R2026b, `maxNumCompThreads(1)`, approved CPUs only
(AMD EPYC 7763 and 9V45). The run reads these inputs and never the forecast. No MATLAB run on these
inputs happened before this commit. Each approved job is scored once against the frozen forecast,
and the result stays on record whether it passes or fails.
