# Phase 2 estimator protocol amendment 4

Amendment ID: `censure-phase2-estimator-v1-amendment-4`  
Parent amendment: `censure-phase2-estimator-v1-amendment-3`  
Parent freeze commit: `b20fae97a1551c31efab27592c61331571edf10d`  
Decision date: 2026-09-02, Asia/Singapore  
Inferential status: outcome-blind shared-support clarification

No frozen shared-support outcome has been inspected at this amendment. Test-namespace simulations
used only for software verification are not Phase 2 evidence.

## Shared-support DGP

The maximum-importance-ratio sweep uses a two-step, binary-action, full-overlap environment. The
initial binary context is equiprobable. Behavior and target policies both choose the first action
uniformly. At the second step, the target probability of action one is 0.75 or 0.25 according to
the deterministic intermediate state; behavior uses the target probability divided by the cell's
ratio cap `R`. The trajectory importance weight therefore never exceeds `R` for `R` in 1, 2, 5,
or 10. Terminal harm is Bernoulli with probability

\[
0.05+0.10x+0.05a_0+0.55a_1,
\]

and exact target risk is 0.40.

The frozen grid has cohort size 1,000; 2,000 repetitions; base seed 20260904; 25 repetitions per
checksummed chunk; all four ratio caps; and outcome-model conditions `correct`, `misspecified`, and
`constant`, for 12 cells total. Conditions share prospective random tapes where their DGP is the
same.

## Estimators and guarantee

Every cell reports behavior risk, ordinary trajectory IPS, self-normalized IPS, direct outcome
regression, and sequential doubly robust estimation. In this DGP the first-step behavior and target
policies coincide and the intermediate transition is deterministic, so the implemented
trajectory correction is algebraically the two-step sequential-DR recursion. Correct regression
uses the known simulation response surface; misspecified regression omits the second action;
constant regression is 0.25. These oracle/misspecified models are calibration baselines, not claims
about deployable model fitting.

The shared-support IPS score lies in `[0,R]`; its one-sided 95% upper certificate is

\[
\min\{1,\widehat V_{IPS}+R\sqrt{\log(1/\alpha)/(2n)}\},\quad\alpha=0.05.
\]

For a hybrid finite-cohort analysis containing both shared-support and frontier contributions,
CENSURE combines a supported-harm UCB and audited-safe-mass LCB as

\[
U=\min\{1,U_{sup}+M_\star-L_A\}.
\]

Such a hybrid claim splits error `alpha_sup=0.025` and `alpha_A=0.025`. Shared-support-only
diagnostics use `alpha=0.05`. Unsupported IPS/DR is still prohibited.

All remaining parent provisions are unchanged.
