# Prospective Ministral 3 breadth extension

Status: frozen before outcome-bearing extension execution on 2026-09-01 (Asia/Singapore).

This is a prospective model-breadth extension of Experiment 1, not completion of the original
three-actor preregistered matrix. It preserves the original endpoints and sampling design while
adding one independently selected actor, `mistralai/Ministral-3-14B-Instruct-2512-BF16`, at the
immutable revision and adapter configuration recorded in
`configs/models/ministral3_14b_tool_alias_v1.yaml`.

## Selection and feasibility history

- Model selection used technical status only. No harm, utility, or terminal-outcome values from
  the Ministral smoke were inspected before freezing this protocol.
- `exp1_ministral3_14b_smoke_v1` failed before generation because Mistral Common rejects dotted
  control-tool names.
- `exp1_ministral3_14b_smoke_v2` passed the frozen technical gate: zero invalid pairs, proposals
  in both trajectories for every AgentDojo suite, 8/8 runtime-restorable pairs, and a witnessed
  checksum resume skip for all 16 trajectories.
- GPT-OSS 20B remains a failed technical-feasibility candidate. The status-only replacement
  decision is to evaluate `Qwen/Qwen3-14B` separately; that decision does not depend on Ministral
  extension outcomes.

## Frozen matrix

The configuration is `configs/experiments/exp1_ministral3_14b_full_v1.yaml`. It contains 320
frozen scenarios and 672 paired sessions (1,344 trajectories): 320 primary strict-to-none pairs,
320 degradation-sweep pairs on the same balanced 80-scenario subset, and 32 identical-strict
negative controls. Splits, seeds, AgentDojo tasks, controlled strata, guard pairs, terminal
validators, retry count, and bootstrap procedure match `exp1_full_v2`; the wall-clock ceiling is
900 seconds for the larger actor.

## Interpretation

Report this run as a prospective extension/replication. Actor-specific behavior risk, target risk,
masking gap, invalid-run rate, utility, and mechanism metrics use the existing paired analysis.
Complete-case estimates must be accompanied by the preregistered conservative invalid-run
sensitivity analysis. Comparisons with Qwen3-8B and Gemma-3-12B-IT are cross-experiment breadth
comparisons and must not be described as completing the original preregistered actor matrix.
