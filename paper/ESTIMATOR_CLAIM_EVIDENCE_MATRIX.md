# Estimator-paper claim/evidence matrix

This file constrains `censure_estimator.tex`. Phase 2 is prospective and outcome-blind at
Amendment 7; software-verification tests are not empirical evidence.

| Claim | Required evidence | Status before Phase 2 | Permitted wording |
|---|---|---|---|
| Anytime finite-cohort coverage | Theorem assumptions, proof, exact-enumeration tests, all 81 validity cells | Theory and software complete; empirical calibration pending | “Guaranteed under the stated frontier, restoration, propensity, and outcome assumptions” |
| Population coverage | IID task sampling plus separate audit/task error budgets and all population cells | Prospective, pending | Never merge with the finite-cohort claim |
| CENSURE improves audit efficiency | 20% paired contrast CI upper endpoint below zero and zero CENSURE primary-budget coverage-gate failures | Prospective, pending | Claim only if both conditions pass; otherwise report “not supported” |
| AUC efficiency | Frozen 0–40% trapezoidal AUC paired analysis | Prospective secondary, pending | Secondary fixed-grid result |
| One-step replay misses harm | Synthetic and held-out one-step/full-suffix diagnostics | Prospective, pending | Terminal harm remains primary; no causal-mediation wording |
| Robust to non-auditable mass/rare harm/model misspecification | Every identified robustness cell | Prospective, pending | Actor/allocation model affects efficiency, not certificate validity |
| Robust to sandbox shift | Declared-radius-corrected cells | Prospective sensitivity analysis, pending | State the radius; do not call uncorrected failures valid |
| Hidden guard features | Positive hidden-feature cells | Prospectively designated unidentified | Never pool these cells into a validity headline |
| Shared-support IPS/DR performance | All 12 full-overlap cells | Prospective secondary, pending | Never extrapolate IPS/DR across zero support |
| Held-out agent certificate | Sealed ledgers, selected suffixes, complete full targets, checksummed summary | Prospective, pending | Actor-specific finite-cohort result; no pooled model-family claim |
| Guard-induced masking exists | Completed Experiment 1 Qwen/Gemma/Ministral synthesis | Retrospective prior evidence | Motivation only; do not relabel as Phase 2 |

Generated empirical wording is allowed only when `paper/generated/phase2_results.tex` comes from a
complete bundle produced by `censure.estimation.cli synthesize-paper`. All source CSVs and every
failed cell remain part of the publication artifact.
