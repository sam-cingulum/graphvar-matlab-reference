# GraphVar reference runs on genuine MATLAB

Synthetic-only reference outputs from GraphVar 2.03a running on genuine MathWorks MATLAB R2026b in GitHub Actions.

- GraphVar 2.03a (GPL-3, Charité Berlin / TU Dresden, https://www.nitrc.org/projects/graphvar/) is downloaded from NITRC at run time and checked against its SHA-256. Its source is not stored in this repository.
- Jobs run only on AMD EPYC 7763 or 9V45 runners, with `MKL_CBWR=COMPATIBLE` and one thread. They run serially, with fixed seeds.
- Inputs are synthetic matrices generated from fixed seeds. There is no patient data.
- Outputs are recorded as IEEE-754 hex words and uploaded as workflow artifacts.

Layout: `drivers/` holds the MATLAB driver scripts for this repository, `tools/` holds the fetch, CPU-gate and assembly scripts, and `cases/` holds the synthetic case definitions.
