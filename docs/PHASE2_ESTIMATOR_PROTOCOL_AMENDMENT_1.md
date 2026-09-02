# Phase 2 estimator protocol amendment 1

Amendment ID: `censure-phase2-estimator-v1-amendment-1`  
Parent protocol: `censure-phase2-estimator-v1`  
Parent freeze commit: `c1c6d0d3c401ed02bef632b2c793cb4596e2fa98`  
Decision date: 2026-09-02, Asia/Singapore  
Inferential status: outcome-blind implementation clarification

At this amendment, no frozen primary calibration cell and no held-out agent suffix outcome has
been run or inspected. Small deterministic unit tests and exhaustive tiny-frontier conformance
tests have been executed. They are software verification, not empirical Phase 2 results.

This amendment fixes implementation details that the parent protocol left implicit. It does not
change the estimand, certificate, confidence level, allocation-policy set, primary grid, or
reporting rules.

## Frozen clarifications

1. In the primary `mixed_auditability` calibration regime, every realized frontier candidate is
   auditable independently with probability 0.75. The robustness experiment replaces this value
   with its frozen non-auditable-mass sweep.
2. In the enumerable delayed-harm DGP, a harmful terminal suffix is missed by the one-step
   endpoint with probability 0.60. A nonharmful suffix cannot be one-step harmful.
3. The two equally likely outcome-free risk strata have target-harm probabilities `0.5 p` and
   `1.5 p`, clipped at one, where `p` is the cell's target-harm-prevalence parameter. The guard
   score exposed to allocation is the corresponding prespecified safe probability, not a realized
   outcome.
4. A positive audit budget `f` produces `ceil(f N_A)` randomized draws, where `N_A` is the number
   of auditable candidates in the realized cohort. Zero budget produces zero draws.
5. Policies compared on one cohort use the same prospectively derived sequence of uniform random
   variates. Policy identity changes propensities, but not the common random tape.
6. Sampling remains with replacement. A duplicate draw of a successfully completed fixed-random-
   tape suffix reuses its checksummed outcome and incurs zero additional simulated suffix tool or
   generation cost. Its draw, propensity, importance value, and duplicate count remain in the
   ledger and certificate. Failed disclosures are retried and remain worst-case until completed.
7. The reported interval lower endpoint is the known supported harm contribution plus terminal
   harm mass from uniquely selected, successfully completed candidates. Failed and unselected
   suffixes contribute zero to this lower endpoint. Reported interval width is the nonnegative
   difference between the primary target-risk upper certificate and this identified lower
   endpoint. Any upper-bound crossing is still recorded as a coverage failure.
8. Exact conformance enumerates every short audit path for every binary tiny-frontier outcome map,
   including adaptive policies. The preregistered Monte Carlo calibration remains the empirical
   coverage analysis.

All other parent-protocol provisions remain unchanged.
