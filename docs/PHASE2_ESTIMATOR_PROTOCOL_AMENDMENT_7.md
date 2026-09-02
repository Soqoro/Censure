# Phase 2 estimator protocol amendment 7

Amendment ID: `censure-phase2-estimator-v1-amendment-7`  
Parent amendment: `censure-phase2-estimator-v1-amendment-6`  
Parent implementation commit: `99dc1a2265eaf6432b63d70b9aba148aa24c8d98`  
Decision date: 2026-09-02, Asia/Singapore  
Inferential status: outcome-blind paper-analysis freeze

No frozen calibration, robustness, shared-support, held-out behavior, selected-suffix, or held-out
full-target outcome was inspected before this amendment. This amendment fixes the final aggregation
and publication rules needed for the estimator paper; it does not alter a scenario, suffix, audit
policy, budget, outcome, or confidence certificate.

## Complete-artifact rule

Synthesis reads checksum-valid raw chunks for every frozen CPU cell and the checksummed held-out
agent study summary. It refuses a partial catalog, an unexpected manifest or actor set, duplicate
agent rows, a dirty Git checkout, or an unsealed/incomplete full-target evaluation. Every cell is
exported to CSV. Headline minima and contrasts are additional views, not filters.

The committed source hashes are:

- calibration: `9464daadf2c76e2257fbb4b26eb1f7b657bb9474f842cf6646dc3a038423e007`;
- robustness: `0b3ca69e6216fd070ea901df24c866890b1c6861f84b4a814b68f31572dabf23`;
- shared support: `b47beb22fbc660c7d0577db4340bf25bb1cee73471da5fe70d7496515d170abe`;
- held-out manifest: `e4a7b11680ed0d6181f24b3bf5c26420453503122b032fc302733fbb7bdfb96d`.

## Calibration and efficiency

Finite-cohort and population coverage remain separate. The paper reports the minimum observed
cell coverage, the number of frozen gate failures, false release, and every cell-by-budget row.
The pre-existing one-sided Clopper--Pearson gate is unchanged.

The primary efficiency statistic compares `censure_bound_targeted` with `uniform` at 20% budget.
For each of the 18 frozen DGP designs, it computes the CENSURE median upper slack minus the uniform
median upper slack over 2,000 paired repetitions, then averages those 18 design contrasts with
equal weight. A 10,000-sample percentile interval resamples paired repetitions within each design
with seed 20260907. The secondary 0--40% upper-slack AUC uses trapezoidal integration over all six
frozen budgets and the same design-stratified statistic with seed 20260908. These intervals quantify
Monte Carlo uncertainty on the fixed DGP grid; they are not deployment-population intervals.

An efficiency claim requires both an upper confidence endpoint below zero and no CENSURE coverage
gate failure at the primary budget. A narrow under-covering certificate is never counted as a win.

## Assumptions and held-out agents

Raw coverage is the primary robustness quantity for identified axes. Declared-radius-corrected
coverage is used for sandbox harm/transition shift. Positive hidden-guard-feature cells remain
explicitly unidentified and may not be pooled into a validity headline. Shared-support IPS, SNIPS,
direct, and sequential doubly robust rows remain secondary and are never extrapolated into
zero-support branches.

Held-out results are actor-specific. The primary applied row is the CENSURE policy at 20% of each
actor's first-block frontier, compared with the full-target risk upper identification endpoint
after the audit seal. Every actor/policy/budget row, logical policy cost, and all-policy physical
cache cost is retained. The three actors are not treated as a probability sample of model
families, and no pooled model-family prevalence claim is permitted.

The generated JSON evidence bundle, CSV tables, LaTeX macros, and PDF/PNG figures are checksummed.
Experiment 1 remains retrospective motivation for why the estimator is needed; it is not relabeled
as prospective Phase 2 evidence.

The canonical machine-readable amendment SHA-256 is
`566a316f9bcfa0dea085bb0901b76adbc04869dc1eb5e20dbab54967a8a8ecd5`.
