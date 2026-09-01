# Qwen3-14B fallback breadth-extension selection declaration

Status: fallback actor and technical-feasibility protocol frozen on 2026-09-02
(Asia/Singapore), before any Qwen3-14B CENSURE trajectory was generated or inspected.

Current status: superseded before manifest freeze or model execution by the prospective
different-family amendment in `GRANITE41_30B_EXTENSION_SELECTION.md`.

This candidate was precommitted as the technical fallback in
`GLM4_EXTENSION_SELECTION.md`. It was activated only after the outcome-blind GLM-4 feasibility
gate failed. Results from the earlier Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-14B studies were
already known, so any accepted Qwen3-14B study must be reported as an **outcome-informed
model-breadth extension with a prospectively frozen within-model protocol**. It cannot turn the
combined actor portfolio into a preregistered or prospectively selected model matrix.

## Status-only GLM-4 disposition

No GLM-4 harm, utility, terminal-state, or masking outcome was inspected for this decision. The
allowed technical feasibility fields showed:

- two invalid pairs among eight selected pairs, versus the frozen maximum of zero;
- one repeated `InvalidToolCallError` and one repeated `ToolCallParseError` in each role;
- no captured proposal in either role for the travel suite;
- successful restoration for all eight checkpointed pairs; and
- no resume witness before the other acceptance conditions had already failed.

The travel response exposed a GLM parser-compatibility gap. Independently, the workspace pair
made a genuine environment-rejected tool call on both deterministic attempts and in both roles.
Consequently, a parser-only revision could not make the frozen zero-invalid gate pass. The GLM-4
smoke remains an integration failure and its manifest and artifacts must not be overwritten.

## Frozen fallback actor

The selected fallback is `Qwen/Qwen3-14B` at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, as recorded in
`configs/models/qwen3_14b.yaml`. Its tokenizer chat-template SHA-256 is
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

The fallback preserves BF16 precision on one A100-80 GB, uses the model's released native JSON
tool template with thinking disabled, and reuses the already tested Qwen tool-call path. The
official model source is the [Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B).

## Outcome-blind feasibility gate

The feasibility experiment is `configs/experiments/exp1_qwen3_14b_smoke_v1.yaml`. It contains
eight strict-to-none pairs: one from each AgentDojo suite and each controlled domain. It is
ineligible for outcome analysis. Before deciding technical acceptance, inspect only status
counts, error classes, proposal coverage, checkpoint restoration, and checksum-resume evidence.

Acceptance requires all of the following:

- zero invalid pairs;
- a captured pre-guard proposal in both roles for all four AgentDojo suites;
- runtime restoration for all eight pairs with zero failures; and
- a witnessed checksum resume skip for all sixteen completed trajectories.

Failure of any condition is a technical-feasibility failure, not evidence for or against safety
masking. Parser or serializer corrections require a new model alias, experiment ID, and manifest;
do not overwrite this smoke manifest.

## Post-feasibility commitment

No outcome-bearing Qwen3-14B full-run configuration is frozen here. If and only if the technical
gate passes, freeze a separate 672-pair within-model protocol before inspecting any Qwen3-14B
harm, utility, terminal-state, or masking outcome. That protocol must preserve the existing
scenario matrix, strict-to-none primary estimand, degradation sweep, same-guard negative control,
invalid-run sensitivity analysis, and clustered bootstrap.

## Append-only prospective amendment (2026-09-02)

Before this fallback's manifest was frozen and before any Qwen3-14B CENSURE trajectory was
generated, the study owner revised the fourth-model breadth objective to require a model family
not already represented by Qwen3-8B. No Qwen3-14B outcome informed that decision. The active
candidate is now Granite 4.1 30B under `GRANITE41_30B_EXTENSION_SELECTION.md`.

This Qwen smoke configuration remains an unexecuted audit record. Do not run it as part of the
Granite extension and do not describe Granite as the originally precommitted GLM fallback.
