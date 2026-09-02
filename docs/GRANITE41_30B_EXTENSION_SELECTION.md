# Granite 4.1 30B breadth-extension selection declaration

Status: actor and technical-feasibility protocol frozen on 2026-09-02
(Asia/Singapore), before any Granite 4.1 30B CENSURE trajectory was generated or inspected.

This candidate was selected after results from the Qwen3-8B, Gemma-3-12B-IT, and
Ministral-3-14B studies were known and after the GLM-4 technical smoke had failed. Consequently,
any accepted Granite study must be reported as an **outcome-informed model-breadth extension
with a prospectively frozen within-model protocol**. It cannot make the combined actor set a
preregistered or prospectively selected model matrix.

## Prospective selection amendment

`Qwen/Qwen3-14B` was previously recorded as the technical fallback after GLM-4. Before a Qwen3-14B
manifest was frozen or any Qwen3-14B CENSURE trajectory was generated, the breadth objective was
amended to prioritize a model family not already represented in the study. The Qwen configuration
and declaration remain in version control as an unexecuted selection record; they are superseded
for this fourth-model extension and must not be run under the Granite protocol.

This amendment was made without observing any Qwen3-14B result. It does not erase the fact that
selection occurred with prior-model outcomes known, so the outcome-informed label remains.

## Frozen actor-selection rationale

The selected candidate is `ibm-granite/granite-4.1-30b` at revision
`4fae6278f7132abf5e971f9de49ebbad09c54cce`, as recorded in
`configs/models/granite41_30b.yaml`. Its released `chat_template.jinja` SHA-256 is
`fed2756d2d24e127b951dcf139d0b03ab7db8ef23a456128ebc9c2db4901d476`.

Selection used these technical and scientific criteria:

- Granite is a model family not represented by Qwen, Gemma, or Mistral;
- the checkpoint is Apache-2.0 licensed and ungated;
- the released template accepts OpenAI function definitions and emits JSON inside exact
  `<tool_call>` tags;
- the native BF16 repository is about 57.7 GB and is plausibly runnable one trajectory at a time
  on one A100-80 GB with a frozen 16,384-token input cap;
- Transformers 4.57.1 provides the native `GraniteForCausalLM` implementation without remote
  code; and
- the official model card reports a BFCL v3 score of 73.68 for the 30B model.

The load remains unquantized BF16 so this actor is comparable to the accepted primary and breadth
actors without a precision confound. These facts were checked against the official
[Granite 4.1 30B model card](https://huggingface.co/ibm-granite/granite-4.1-30b) and its immutable
Hub revision on the selection date.

## Frozen parser and runtime contract

CENSURE uses the released Granite template rather than a synthetic prompt projection. Tool
schemas remain canonical OpenAI function objects, and no tool-name alias is applied. The
`granite_tool_calls_v1` parser accepts public text followed by one or more adjacent tagged JSON
calls, preserves call order, strips one released terminal token from final text, and fails closed
on partial tags, malformed JSON, inter-call text, or trailing content. Persisted calls and tool
results retain CENSURE call identities even though Granite's released prompt representation uses
their ordered values rather than IDs.

The runtime is frozen to the standalone `requirements/colab-exp1-granite.txt` lock. It must use
native BF16 on a runtime exposing at least 75 GiB GPU memory and at least 75 GiB free local disk.
Do not substitute a quantized checkpoint, mutable `main` revision, different template, larger
context cap, or alternate serving engine under this experiment ID.

## Outcome-blind feasibility gate

The feasibility experiment is `configs/experiments/exp1_granite41_30b_smoke_v1.yaml`. It contains
eight strict-to-none pairs: one from each AgentDojo suite and each controlled domain. It is
ineligible for outcome analysis. Before deciding technical acceptance, inspect only status
counts, error classes, proposal coverage, checkpoint restoration, and checksum-resume evidence.

Acceptance requires all of the following:

- zero invalid pairs;
- a captured pre-guard proposal in both roles for all four AgentDojo suites;
- runtime restoration for all eight pairs with zero failures; and
- a witnessed checksum resume skip for all sixteen completed trajectories.

Failure of any condition is a technical-feasibility failure, not evidence for or against safety
masking. A parser, serializer, template, dependency, context, or retry change requires a new model
alias, experiment ID, declaration, and manifest; do not overwrite this smoke.

## Post-feasibility commitment

No outcome-bearing Granite full-run configuration is frozen here. If and only if the technical
gate passes, freeze a separate 672-pair within-model protocol before inspecting any Granite harm,
utility, terminal-state, or masking outcome. That protocol must preserve the existing scenario
matrix, strict-to-none primary estimand, degradation sweep, same-guard negative control, invalid
run sensitivity analysis, and task-clustered bootstrap.

Every technically valid result remains reportable:

- a positive gap is different-family breadth evidence;
- a near-zero gap with adequate task utility is a boundary condition;
- a negative gap is evidence that the target trajectory reduced realized harm; and
- a failed feasibility gate is reported only as an integration failure.

## Status after the frozen v1 smoke

The v1 smoke subsequently failed its frozen zero-invalid requirement with one invalid workspace
pair. The original manifest, report, and decision rule remain unchanged and v1 is not accepted.
After inspecting only the permitted technical report fields, a separate 40-pair post-hoc
operational-feasibility experiment was frozen. Its rationale, exact evidence boundary, and gate
are recorded in `GRANITE41_30B_OPERATIONAL_FEASIBILITY.md`; it must not be described as a repair
or retrospective pass of this prospective v1 smoke.
