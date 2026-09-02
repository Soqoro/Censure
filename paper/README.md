# Paper build

`censure_iclr2027.tex` is the ICLR-formatted Experiment 1 manuscript. It replaces the prospective
placeholder draft with only results supported by the frozen three-model synthesis. Numerical and
wording constraints are recorded in `CLAIM_EVIDENCE_MATRIX.md`.

To build the PDF, place the official `iclr2027_conference.sty` and
`iclr2027_conference.bst` files in this directory, then run:

```bash
latexmk -pdf censure_iclr2027.tex
```

Run the three-model CPU synthesis before copying generated figures or table fragments into a
submission bundle. Its command is documented in `docs/THREE_MODEL_SYNTHESIS_PROTOCOL.md`. Do not
replace the actor-specific headline estimates with a pooled three-model average, and do not
describe the Gemma estimate as equivalence to zero.
