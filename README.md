# CENSURE

CENSURE is an experimental framework for first-divergence auditing of action guards around
tool-using language agents. The paired-oracle foundation and Experiment 1 (guardrail-induced
safety masking) are complete. Phase 2 now has a frozen prospective protocol for the CENSURE
estimator, randomized suffix auditing, confidence sequences, and Experiments 2–5. The estimator,
CPU studies, and selected-only held-out agent workflow are implemented; prospective execution is
pending.

Each frozen scenario is run twice from independently restored copies of one canonical initial
state: first with a behavior guard and then as a complete trajectory under the target guard. Raw
oracle artifacts live under `oracle_private/` and require an explicit evaluation capability.
Individual outcomes are stored as realized harm labels; risk is reported only as an aggregate over
the frozen scenario/seed distribution.

## Layout

- `src/censure/adapters/agentdojo_v0135.py`: the only version-sensitive AgentDojo boundary.
- `src/censure/environments/`: the 160-instance declarative controlled layer and state runtimes.
- `src/censure/guards.py`: strict, weak, none, same-guard, and degraded-strict middleware.
- `src/censure/actors/`: scripted and direct Transformers actors with normalized tool calls.
- `src/censure/execution.py`: instrumented full-trajectory and paired execution.
- `src/censure/manifest.py`: outcome-free scenario freezing and session-key expansion.
- `src/censure/analysis/`: Experiment 1 metrics, clustered bootstrap, tables, and figures.
- `src/censure/analysis_scope.py`: validated declarations for explicitly partial actor analyses.
- `configs/`: frozen experiment/model configurations.
- `experiments/`: Colab-safe shell entrypoints.
- `docs/COLAB_EXP1.md`: copy-paste notebook workflow.
- `docs/PHASE2_ESTIMATOR_PROTOCOL.md`: prospectively frozen estimator and Experiments 2–5 design.
- `configs/experiments/phase2_estimator_v1.yaml`: machine-readable Phase 2 protocol.
- `docs/PHASE2_ESTIMATOR_PROTOCOL_AMENDMENT_1.md`: outcome-blind enumerable-DGP and audit-cost clarifications.
- `docs/PHASE2_ESTIMATOR_PROTOCOL_AMENDMENT_2.md`: outcome-blind validity/efficiency execution grids.
- `docs/PHASE2_ESTIMATOR_PROTOCOL_AMENDMENT_3.md`: outcome-blind population and robustness grids.
- `docs/PHASE2_ESTIMATOR_PROTOCOL_AMENDMENT_4.md`: outcome-blind shared-support OPE grid.
- `docs/PHASE2_CPU_WORKFLOW.md`: resumable CPU-only Experiment 2/3 commands.
- `docs/PHASE2_AGENT_WORKFLOW.md`: sealed, restart-safe held-out behavior/suffix/target workflow.
- `docs/PHASE2_ESTIMATOR_PROTOCOL_AMENDMENT_6.md`: selected-only suffix execution and outcome-release seal.
- `docs/PHASE2_ESTIMATOR_PROTOCOL_AMENDMENT_7.md`: outcome-blind paper aggregation and publication freeze.
- `paper/MANUSCRIPT_DRAFT.md`: results-grounded Experiment 1 manuscript draft.
- `paper/CLAIM_EVIDENCE_MATRIX.md`: permitted claims and inferential-status guardrails.
- `paper/references.bib`: source-verified bibliography for the manuscript draft.
- `paper/censure_iclr2027.tex`: compile-ready ICLR 2027 LaTeX manuscript for Experiment 1.
- `paper/censure_estimator.tex`: result-gated full estimator manuscript with theorem and proof.
- `paper/ESTIMATOR_CLAIM_EVIDENCE_MATRIX.md`: allowed estimator-paper claims and evidence gates.
- `paper/README.md`: paper build and reporting handoff.

The active Colab workflow uses the `_v2` experiment configs and CENSURE-Control scenario v2.
Legacy configs without that suffix remain reconstructible for the archived first pilot; v1 and v2
outputs have distinct experiment and session IDs and must not be combined.

## Local CPU development

No model weights are needed for the local suite.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[analysis,agentdojo,dev]'
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/pyright
```

AgentDojo is pinned to `0.1.35` and benchmark `v1.2.2`. Durable state and trace files use canonical
versioned JSON, never pickle. See [docs/COLAB_EXP1.md](docs/COLAB_EXP1.md) for GPU execution.

Pair alignment (`diverged`, `no_divergence`, or `invalid`) is stored separately from each raw
trajectory status; a successfully executed trajectory remains `completed` because divergence is a
property that exists only after evaluation-gated pairing. Quantized T4 smoke has a separate model
config, experiment ID, and no confirmatory rows, so it cannot be mixed into the BF16 primary
analysis.

The primary estimand is `strict → none` over all frozen scenarios. The secondary degradation
sweep holds behavior at strict and moves the target toward none with
`degraded_strict(rho)`; `rho=1` therefore reproduces target-none on the balanced secondary subset.
Those repeated subset rows and same-guard controls are reported by guard pair but never reweight
the primary overall/domain/actor summaries.

AgentDojo's released validators rebuild a few current collections from `initial_*` fields. The
version-pinned adapter explicitly rehydrates and verifies those mappings so mutated checkpoints
round-trip; arbitrary aliases introduced by external Python code are not preserved by canonical
JSON. The official Tool Filter is not reported in this phase because its direct local-model
integration has not been verified, and no Task Shield/Tool Filter approximation is labeled as the
published defense.
