# Retrospective Qwen–Gemma–Ministral synthesis

Status: frozen on 2026-09-02 (Asia/Singapore), after actor-specific outcomes and the all-pair
robustness results had been inspected. This is an explicitly **retrospective cross-experiment
synthesis**, not a preregistration and not completion of the original three-actor matrix.

The machine-readable specification is
`configs/analysis/exp1_three_model_synthesis_v1.yaml`. Model collection is closed for this
synthesis. GPT-OSS and Granite remain outcome-blind technical exclusions; Llama remains the
status-only feasibility deferral recorded by `qwen_gemma_v1`.

## Inputs and alignment

The synthesis combines the validated Qwen3-8B and Gemma-3-12B rows from the frozen
`qwen_gemma_v1` scope with the validated Ministral-3-14B breadth extension. The two experiments
use the same manifest seed, 320 scenarios, environment/task splits, guard-pair matrix, degradation
subset, attack construction, controlled strata, and state serialization contract.

Rows must align one-to-one for every actor on `(scenario_id, guard_pair_id)`. Scenario identity,
task, attack, policy, initial state, environment seed, and both guard IDs must agree across actors.
The source manifest, validation, scope/extension declaration, actor set, row count, guard-pair
count, and scenario-set hash are verified before analysis. Any mismatch fails closed.

Task-clustered uncertainty uses the composite
`(environment_layer, domain, user_task_id)` so identically named tasks in different suites cannot
be merged into one bootstrap cluster.

## Frozen reporting hierarchy

Actor-specific confirmatory `strict_none` effects are primary. No pooled three-model effect is a
primary estimand. Each actor must be reported with:

- complete-case behavior risk, target risk, signed masking gap, and clustered interval;
- the original invalid-run sensitivity estimate;
- sharp all-pair binary-harm identification bounds and endpoint intervals;
- invalid-pair and harm-or-invalid rates; and
- utility, unsafe-attempt, blocking, call-count, and guard-dependence diagnostics.

The three new task-paired actor-by-guard contrasts are Qwen minus Gemma, Ministral minus Gemma,
and Ministral minus Qwen. Complete-case contrasts use only tasks valid for both actors. Sensitivity
contrasts use every shared task. For actors A and B, the all-pair contrast bounds are

\[
[L_A-U_B,\ U_A-L_B],
\]

computed pairwise before averaging. Their 95% task-cluster bootstrap intervals are exploratory
and unadjusted for three comparisons.

## Mechanism and falsification analyses

Degradation curves use the same confirmatory scenarios present at every degraded target-guard
level and at `strict_none`. Complete-case, sensitivity, and all-pair endpoints remain separate.
The `same_guard_strict` rows are reported as negative controls, not inserted into the dose curve.

Unsafe-attempt, block, proposed-call, zero-call, utility, and guard-dependence summaries are
descriptive. They can support an actor-conditional mechanism interpretation but cannot establish
causal mediation. Domain cells are exploratory.

## Interpretation constraints

- Retain signed gaps and reverse events; never truncate a negative value to zero.
- Do not call the cross-model contrasts confirmatory or multiplicity-adjusted.
- Do not interpret Gemma's near-zero estimate as equivalence without a prespecified equivalence
  margin.
- Do not use an equal-weighted pooled actor average as the headline result.
- Preserve each source study's original inferential label in all machine- and human-readable
  artifacts.
- Report GPT-OSS, Granite, and Llama dispositions separately as technical/feasibility evidence,
  not as observed zero-harm actors.

## CPU execution

No model weights, AgentDojo runtime, or GPU are required. In a fresh CPU Colab, install
`requirements/colab-analysis.txt` and CENSURE itself, then run:

```bash
bash experiments/exp1/analyze_three_model_synthesis.sh \
  --spec configs/analysis/exp1_three_model_synthesis_v1.yaml \
  --source-root core=/content/drive/MyDrive/CENSURE/outputs/exp1 \
  --source-root extensions=/content/drive/MyDrive/CENSURE/outputs/exp1_extensions \
  --out-dir /content/drive/MyDrive/CENSURE/outputs/synthesis/qwen_gemma_ministral_v1
```

The command verifies both sources before writing `metrics.json`, `combined_pairs.parquet`, actor,
contrast, domain, degradation, and negative-control CSVs, six LaTeX tables, a synthesis report,
three PNG/PDF figures, source/run provenance, and a checksummed artifact manifest. The manuscript
tables cover actor effects, task-paired contrasts, domain effects, mechanism diagnostics,
degradation, and identical-guard controls.
