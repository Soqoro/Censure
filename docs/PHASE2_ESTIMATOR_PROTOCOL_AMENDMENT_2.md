# Phase 2 estimator protocol amendment 2

Amendment ID: `censure-phase2-estimator-v1-amendment-2`  
Parent amendment: `censure-phase2-estimator-v1-amendment-1`  
Parent freeze commit: `77f1784aae21c9e5043ac49faf0006bfe0b39ef1`  
Decision date: 2026-09-02, Asia/Singapore  
Inferential status: outcome-blind execution-grid clarification

No frozen primary calibration cell or held-out agent suffix outcome has been inspected at this
amendment. Runtime benchmarks used nonprimary `bench` namespaces and are not analyzed as Phase 2
evidence.

## Validity grid

The Experiment 2 primary validity policy is `target_mass`. It is evaluated with 2,000 independent
cohort/audit-tape repetitions in every primary DGP cell. Full overlap has one zero-support value,
zero; deterministic cloneable non-overlap and mixed auditability use zero-support masses 0.10,
0.25, 0.50, and 0.75. All use cohort sizes 200, 500, and 1,000; target-harm-prevalence parameters
0.05, 0.20, and 0.50; and the six frozen budgets.

## Efficiency grid

Experiment 3 compares all six allocation policies using common cohorts and audit random tapes in
the following primary subset:

- support regimes: deterministic cloneable non-overlap and mixed auditability;
- cohort size: 500;
- target-harm prevalence: 0.05, 0.20, 0.50;
- zero-support mass: 0.25, 0.50, 0.75;
- repetitions: 2,000;
- budgets: all six frozen budgets.

`target_mass` rows shared with the validity grid are reused byte-for-byte rather than rerun.
Efficiency conclusions are primary only for this subset; other grid-wide policy results, if later
run, are labeled supplementary.

## Execution and longitudinal reporting

Each work item is a deterministic contiguous chunk of 25 repetition indices within one cell.
Every row retains its prospectively derived per-repetition cohort and audit seeds. Chunks are
atomically checksummed, shards are deterministic, resume skips only checksum-valid chunks, and
summaries require all 2,000 repetitions. The prospectively fixed base seed is 20260902. The
longitudinality analysis uses each unique validity-grid cohort once, independent of policy, and is
reported by support regime, harm prevalence, zero-support mass, and cohort size. Full-overlap
cells have no suffix audit and retain their exact supported risk at every nominal budget.

All remaining parent-protocol and amendment-1 provisions are unchanged.
