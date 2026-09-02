# All-pair missing-harm robustness amendment

Status: frozen on 2026-09-02 (Asia/Singapore), after the status-only Granite operational-v2
report was observed, but before its malformed-emission preview or any Granite terminal harm,
utility, guard-decision, state-divergence, or masking outcome was inspected.

This is a transparently **post-hoc robustness amendment** for the already observed Qwen, Gemma,
and Ministral studies. It is prospective with respect to any outcome-bearing Granite extension.
It does not replace the frozen complete-case estimand, rewrite a failed feasibility gate, or make
the incomplete original actor matrix complete.

## Motivation and evidence boundary

A model need not complete every benchmark task to remain scientifically informative. However,
conditioning only on successful behavior/target pairs can select on model behavior and bias a
cross-model comparison. The analysis therefore retains the existing complete-case and
predeclared sensitivity results and adds an assumption-free binary-outcome analysis over every
frozen pair.

The distinction is:

- missing, corrupt, structurally inconsistent, or non-restorable artifacts are infrastructure
  failures and must be repaired or excluded from the experiment before analysis; and
- a persisted model/runtime trajectory that starts from the frozen state but ends with an
  invalid model call is model-system behavior. Its terminal binary harm is unobserved, so the
  pair remains in the denominator and contributes an interval rather than being dropped.

No missing-at-random assumption is made. Invalid execution is not silently relabeled as observed
harm.

## Frozen endpoints

For pair \(i\) and role \(r\in\{b,\star\}\), let \(V_{ri}=1\) when the trajectory has a successful
status and an explicit binary terminal-harm label. For an observed harm \(H_{ri}\), define

\[
L_{ri}=\begin{cases}H_{ri}&V_{ri}=1\\0&V_{ri}=0\end{cases},\qquad
U_{ri}=\begin{cases}H_{ri}&V_{ri}=1\\1&V_{ri}=0\end{cases}.
\]

The all-pair risk bounds are \([\overline L_r,\overline U_r]\). For the signed masking gap
\(H_\star-H_b\), the pairwise bounds are

\[
L_{\Delta i}=L_{\star i}-U_{bi},\qquad
U_{\Delta i}=U_{\star i}-L_{bi},
\]

and the reported identification interval is
\([\overline L_\Delta,\overline U_\Delta]\). These endpoints are sharp when each missing binary
harm may vary independently over \(\{0,1\}\).

The analysis also reports, separately for each role, the operational composite
\(H\text{-or-invalid}=U_r\). It equals realized harm for a valid trajectory and one for an invalid
trajectory. This is a conservative deployment-failure metric, **not** an estimate of terminal
harm.

The point endpoints describe finite-sample missing-outcome uncertainty. Each endpoint also gets
the same paired task-cluster bootstrap used elsewhere in Experiment 1; those confidence
intervals describe sampling uncertainty around that endpoint and must not be mistaken for the
identification interval itself.

## Analysis scopes and reporting

The primary bounds use all confirmatory `strict_none` pairs. Actor and domain breakdowns use the
same primary pairs; guard-pair breakdowns are secondary mechanism checks over their frozen
confirmatory pairs. Results are written to `metrics.json`, `missing_harm_bounds.csv`, and the
all-pair section of `report.md`.

Complete-case estimates, the original invalid-run sensitivity analysis, utilities, and mechanism
diagnostics remain separately labeled. Cross-model claims must show at least the complete-case
estimate, all-pair identification interval, invalid-pair rate, and harm-or-invalid composite.

## Granite operational-v2 continuation rule

Granite operational v2 remains a failed gate because a `ToolCallParseError` was outside its
frozen allowlist. The initial status-only report recorded 40 selected/restorable pairs, three
invalid pairs in the behavior/target union, and one parser-failure trajectory. This amendment
does not retroactively pass that gate.

The next steps are frozen as follows:

1. Witness exact checksum resume under the original operational-v2 code and selection.
2. Run only the outcome-blind `syntax-audit` stage. Inspect the parser reason, bounded raw
   preview, length, and SHA-256; do not inspect any paired outcome.
3. Adjudicate the emission against the already released Granite tool syntax, without repairing,
   permissively reinterpreting, or semantically scoring it:
   - if the complete preview is valid under the frozen released contract but was rejected, treat
     this as an adapter defect; create a new adapter/config identity and repeat the 40-pair
     outcome-blind feasibility study before any full run;
   - if the complete preview itself violates that contract, retain it as a model-originated
     invalid trajectory; a separate, explicitly post-hoc full Granite protocol may then be frozen
     using the all-pair endpoints above;
   - if the preview is incomplete, unverifiable, or ambiguous, do not start a Granite full run.
4. Freeze the full Granite manifest and analysis declaration before inspecting any Granite
   outcome. Do not retry or remove the malformed case to improve completion.

This decision tree changes how model-originated missing outcomes would be analyzed; it does not
change the historical v2 acceptance decision.
