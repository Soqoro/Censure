# Phase 2 estimator protocol amendment 3

Amendment ID: `censure-phase2-estimator-v1-amendment-3`  
Parent amendment: `censure-phase2-estimator-v1-amendment-2`  
Parent freeze commit: `86588e2017835a29edda21a8208f91c19d2c5ca2`  
Decision date: 2026-09-02, Asia/Singapore  
Inferential status: outcome-blind population/robustness clarification

No frozen primary calibration, robustness, or held-out agent suffix outcome has been inspected at
this amendment. Deterministic test-namespace simulations are software verification only.

## Population certificate

Experiment 2 additionally evaluates the secondary IID population claim. The exact enumerable-DGP
population target risk equals the cell's target-harm-prevalence parameter. Allocation and suffix
disclosures are shared with the cohort analysis, but the population certificate recomputes the
audit confidence sequence at `alpha_A=0.025` and adds the one-sided Hoeffding task-sampling radius

\[
\sqrt{\log(1/\alpha_L)/(2n)},\qquad \alpha_L=0.025.
\]

Population and finite-cohort coverage are reported separately. The cohort-level primary
certificate remains at `alpha_A=0.05`.

## Robustness execution grid

Experiment 5 uses one-factor-at-a-time cells around a deterministic cloneable non-overlap baseline
with cohort size 500, target-harm prevalence 0.20, zero-support mass 0.50, audit budget 20%, 2,000
repetitions, base seed 20260903, and 25 repetitions per checksummed chunk.

The primary target-mass policy is used for:

- non-auditable candidate probability: 0, 0.10, 0.25, 0.50;
- hidden-frontier-candidate probability: 0, 0.10, 0.25;
- sandbox harm shift radius: 0, 0.05, 0.10, 0.20;
- sandbox transition shift radius: 0, 0.05, 0.10, 0.20;
- target-harm prevalence: 0.01, 0.05, 0.20.

The `censure_bound_targeted` policy is used for outcome-model conditions `correct`, `misspecified`,
and `constant`. Correct metadata uses the frozen DGP strata; misspecified metadata uses an
outcome-independent hash stratum and reverses the guard score; constant metadata uses one stratum
and a constant score.

Non-auditable candidates remain in the observed harmful envelope and therefore do not violate the
primary assumptions. Hidden candidates are removed from the observed frontier, receive no
cosmetic correction, and are labeled unidentified. Sandbox-shift cells deterministically make a
subset of actually harmful frontier suffixes appear safe in the audit sandbox, with total shifted
target mass no greater than the declared radius. Both the uncorrected certificate and the
sensitivity certificate `min(1,U+epsilon_sim)` are reported.

The maximum-supported-importance-ratio sweep remains a separate shared-support OPE experiment and
will use the parent protocol's ratios 1, 2, 5, and 10; it is not conflated with frontier auditing.

All remaining parent provisions are unchanged.
