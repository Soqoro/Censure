# Paper builds

`censure_iclr2027.tex` is the ICLR-formatted Experiment 1 manuscript. It replaces the prospective
placeholder draft with only results supported by the frozen three-model synthesis. Numerical and
wording constraints are recorded in `CLAIM_EVIDENCE_MATRIX.md`.

`censure_estimator.tex` is the full estimator-paper manuscript. It is intentionally result-gated:
before Phase 2 finishes it loads `phase2_results_placeholder.tex` and states that empirical results
are pending. After a complete `synthesize-paper` run, copy `phase2_results.tex` and the `figures/`
directory from the checksummed bundle into `paper/generated/`. The manuscript then includes only
artifact-derived numerical claims. Its allowed claims are recorded in
`ESTIMATOR_CLAIM_EVIDENCE_MATRIX.md`.

To build the PDF, place the official `iclr2027_conference.sty` and
`iclr2027_conference.bst` files in this directory, then run:

```bash
latexmk -pdf censure_iclr2027.tex
latexmk -pdf censure_estimator.tex
```

Run the three-model CPU synthesis before copying generated figures or table fragments into a
submission bundle. Its command is documented in `docs/THREE_MODEL_SYNTHESIS_PROTOCOL.md`. Do not
replace the actor-specific headline estimates with a pooled three-model average, and do not
describe the Gemma estimate as equivalence to zero.

For the estimator paper, do not hand-edit generated macros or figures. Preserve the synthesis
bundle's `artifacts.json` and `artifacts.sha256`, and keep Experiment 1 labeled retrospective.
