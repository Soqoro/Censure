# GLM-4 breadth-extension selection declaration

Status: actor and technical-feasibility protocol frozen on 2026-09-01 (Asia/Singapore), before
any GLM-4 CENSURE trajectory was generated or inspected.

This candidate was selected after results from the Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-14B
runs were known. Consequently, any accepted GLM-4 study must be reported as an **outcome-informed
model-breadth extension with a prospectively frozen within-model protocol**. It cannot make the
combined actor set a preregistered or prospectively selected four-model matrix.

## Frozen actor-selection rationale

The selected candidate is `zai-org/GLM-4-32B-0414` at the immutable revision recorded in
`configs/models/glm4_32b_0414.yaml`. Selection used these criteria:

- a model family not already represented by Qwen, Gemma, or Mistral;
- native, documented external JSON function calling;
- unquantized BF16 inference plausibly runnable one trajectory at a time on one A100-80 GB;
- a permissive MIT model license and a directly loadable Transformers implementation; and
- enough reported function-calling capability to justify the technical integration cost.

Llama-3.3-70B and tool-capable Kimi-K2 were excluded because their BF16 checkpoints do not fit a
single A100-80 GB. Llama-4 Scout would require single-GPU Int4 and therefore introduce a precision
confound. Kimi-Linear-48B-A3B also exceeds the available GPU memory in BF16 and does not provide
the same clearly documented native tool protocol. `Qwen/Qwen3-14B` remains the technical fallback
only if GLM-4 fails the frozen feasibility gate; it must not replace GLM-4 because a valid GLM-4
outcome is null or otherwise scientifically inconvenient.

Model-selection sources were the official
[GLM-4 model card](https://huggingface.co/zai-org/GLM-4-32B-0414),
[Llama-3.3 model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
[Meta Llama inference requirements](https://github.com/meta-llama/llama-models/blob/main/README.md#running-the-models),
[Kimi-K2 model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct), and
[Kimi-Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct),
accessed on the selection date.

## Outcome-blind feasibility gate

The feasibility experiment is `configs/experiments/exp1_glm4_32b_smoke_v1.yaml`. It has eight
strict-to-none pairs: one from every AgentDojo suite and every controlled domain. It is ineligible
for outcome analysis. Before deciding technical acceptance, inspect only its status counts, error
classes, proposal coverage, checkpoint restoration, and checksum-resume evidence.

Acceptance requires all of the following:

- zero invalid pairs;
- a captured pre-guard proposal in both roles for all four AgentDojo suites;
- runtime restoration for all eight pairs with zero failures; and
- a witnessed checksum resume skip for all sixteen completed trajectories.

Failure of any gate is a technical-feasibility failure, not evidence for or against safety
masking. Parser or serializer corrections require a new model alias, experiment ID, and manifest.
Do not overwrite this smoke manifest.

## Post-feasibility commitment

No outcome-bearing GLM-4 full-run configuration is frozen in this declaration. If and only if the
technical gate passes, freeze a separate 672-pair configuration before inspecting any GLM-4 harm,
utility, terminal-state, or masking outcome. That protocol must preserve the existing scenario
matrix, strict-to-none primary estimand, degradation sweep, same-guard negative control, invalid
run sensitivity analysis, and clustered bootstrap.

Every technically valid result remains reportable:

- a positive gap is fourth-family breadth evidence;
- a near-zero gap with adequate task utility is a boundary condition;
- a negative gap is evidence that the target trajectory reduced realized harm; and
- an invalid feasibility run is reported only as an integration failure.

## Append-only technical disposition (2026-09-02)

The frozen smoke failed its outcome-blind feasibility gate: two of eight pairs were invalid and
the travel suite had no captured proposal in either role. Checkpoint restoration passed for all
eight pairs. One failure was a parser-compatibility gap, while a separate workspace pair made the
same environment-rejected call on both deterministic attempts and in both roles. A parser-only
revision therefore could not satisfy the frozen zero-invalid requirement.

No GLM-4 harm, utility, terminal-state, or masking outcome was inspected. The immutable smoke
artifacts remain an integration failure, and the precommitted Qwen3-14B fallback is activated in
`QWEN3_14B_EXTENSION_SELECTION.md`.

## Append-only prospective family-breadth amendment (2026-09-02)

The Qwen3-14B fallback was subsequently superseded before its manifest was frozen or any model
trajectory was generated. The study owner elected to prioritize a genuinely different model
family, without observing a Qwen3-14B outcome. Granite 4.1 30B is now the active technical
candidate under `GRANITE41_30B_EXTENSION_SELECTION.md`; the outcome-informed extension label is
unchanged.
