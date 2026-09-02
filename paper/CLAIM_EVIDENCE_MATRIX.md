# Experiment 1 claim–evidence matrix

This document constrains manuscript wording for synthesis `qwen_gemma_ministral_v1`. It is a
reporting guardrail, not a new analysis. Outcomes were already known when it was written.

## Evidence hierarchy

| Priority | Evidence | Inferential status | Permitted role |
|---:|---|---|---|
| 1 | Ministral actor-specific strict-to-none result | Prospective model-breadth extension | Strongest replication and robustness result |
| 2 | Qwen actor-specific strict-to-none result | Post-hoc partial prespecified-actor analysis | Convergent actor-specific evidence |
| 3 | Gemma actor-specific result | Post-hoc partial prespecified-actor analysis | Boundary/null case; not equivalence evidence |
| 4 | Missing-harm identification intervals and endpoint intervals | Post-hoc robustness amendment | Diagnose model-dependent invalid execution |
| 5 | Domain, mechanism, and combined degradation summaries | Descriptive/exploratory | Explain heterogeneity and construct validity |
| 6 | Cross-model task-paired contrasts | Retrospective, exploratory, unadjusted | Heterogeneity diagnostics only |

## Claim constraints

| Topic | Supported wording | Unsupported wording |
|---|---|---|
| Existence | “A behavior guard can mask actor risk.” | “Behavior guards always mask risk.” |
| Replication | “Positive gaps were observed for Qwen and prospectively for Ministral.” | “The complete preregistered model matrix confirmed the hypothesis.” |
| Ministral robustness | “The conservative lower endpoint and its sampling interval remained positive.” | “Every possible population effect is positive.” |
| Qwen robustness | “Complete-case and declared sensitivity intervals were positive; the observed all-pair interval was positive.” | “Qwen remained positive under worst-case missingness and sampling uncertainty.” |
| Gemma | “No complete-case masking event was observed; bounds straddled zero.” | “Gemma is immune to masking,” “Gemma is equivalent to zero,” or “Gemma is safer.” |
| Model contrasts | “Exploratory contrasts indicate heterogeneity relative to Gemma.” | “Ministral ranks above Qwen,” “family causes masking,” or any confirmatory ranking claim |
| Domains | “Effects replicated in travel, Slack, and payments and were absent or inconclusive elsewhere.” | “The effect generalized across all eight domains.” |
| Degradation | “Ministral showed a monotone matched degradation pattern; Qwen was weaker and Gemma had no realized complete-case dose response.” | “All models showed a statistically established dose response.” |
| Mechanism | “Patterns are consistent with unsafe-action opportunity, direct blocking, and trajectory feedback.” | “Unsafe attempts or blocks causally mediate the effect.” |
| Guard value | “The guard reduced deployed-system harm while masking target-policy actor risk.” | “The guard made the actor more dangerous” or “guards should be removed.” |
| Invalid runs | “Invalid terminal harm is unknown and retained through sensitivity and bounds.” | “Invalid runs were safe,” “failed tasks were removed,” or “bounds are confidence intervals.” |
| Negative controls | “Zero identical-guard gaps support paired-execution integrity.” | “Negative controls prove the substantive hypothesis.” |

## Numerical anchors

| Actor | Complete gap (95% CI) | Sensitivity gap (95% CI) | All-pair gap interval | Conservative lower endpoint (95% CI) |
|---|---:|---:|---:|---:|
| Qwen3-8B | 0.137 (0.060, 0.237) | 0.142 (0.059, 0.237) | [0.065, 0.203] | 0.065 (-0.027, 0.167) |
| Gemma-3-12B | 0.000 (0.000, 0.000) | 0.013 (0.000, 0.036) | [-0.026, 0.039] | -0.026 (-0.056, -0.004) |
| Ministral-3-14B | 0.178 (0.097, 0.281) | 0.190 (0.108, 0.289) | [0.134, 0.224] | 0.134 (0.048, 0.234) |

The point interval in the fourth column is a finite-sample partial-identification region. The
interval in the fifth column is sampling uncertainty around only its conservative endpoint.

## Headline and abstract rule

The headline must use “can,” “may,” or an equally conditional construction. The abstract must:

1. identify the paired full-trajectory comparison;
2. report all three actor-specific results rather than only positive actors;
3. name the retrospective synthesis and prospective Ministral distinction;
4. state that effects concentrate in selected domains; and
5. avoid a pooled three-model average or family ranking.

## Figure and table order

1. Actor-specific masking gaps with complete-case intervals and all-pair identification bands.
2. Domain-by-actor masking estimates.
3. Matched degradation curves.
4. Descriptive unsafe-attempt/blocking mechanism panel.
5. Exploratory cross-model contrasts in the supplement or after actor-specific results.
6. Identical-guard controls and complete missing-harm endpoint intervals in the supplement.

Every figure caption must identify complete-case conditioning, bootstrap cluster unit, whether a
band is a confidence interval or identification interval, and whether the analysis is exploratory.
