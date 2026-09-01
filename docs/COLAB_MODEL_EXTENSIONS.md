# Outcome-blind model-extension smoke runs

These are technical feasibility runs for model-breadth candidates. They are separate from the
frozen Experiment 1 matrix, use unique experiment IDs, and are explicitly ineligible for outcome
analysis. Inspect only the outcome-blind feasibility report until the applicable extension
protocol is frozen.

The frozen Hugging Face revisions and chat templates were verified on their selection dates:

| Track | Frozen model revision | Frozen chat-template SHA-256 | Load path |
| --- | --- | --- | --- |
| `ministral3_14b_tool_alias_v1` | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | `2f545122222db8bb43ca0ea0c49e9185320a8670f7d35575b0da0eb48b1e8970` | Native BF16; reversible Mistral tool-name projection |
| `gpt_oss_20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146` | Frozen MXFP4 weights dequantized to BF16 |
| `glm4_32b_0414` | `077b5c2f5c43bd3239fd605a0600229e8facbd4a` | `db700f25fa300e53634c6fc78dee86b7fbd6d27e624edb855b18a4078c83a822` | Native BF16; GLM assistant-metadata/observation projection |
| `qwen3_14b` (archived; unexecuted) | `40c069824f4251a91eefaf281ebe4c544efd3e18` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` | Native BF16; native Qwen JSON tools; thinking disabled |
| `granite41_30b` (active feasibility candidate) | `4fae6278f7132abf5e971f9de49ebbad09c54cce` | `fed2756d2d24e127b951dcf139d0b03ab7db8ef23a456128ebc9c2db4901d476` | Native BF16; released Granite XML-wrapped JSON tools |

Do not replace any revision with `main`. If a Hub head changes, review the template/protocol
change and create a new model config and experiment ID.

The frozen `exp1_ministral3_14b_smoke_v1` feasibility run is retained as a failed integration
attempt. Mistral Common rejected dotted CENSURE control-tool names before generation. The v2
track uses the explicit `mistral_tool_name_alias_v1` prompt projection, maps generated aliases
back to canonical environment names before guard evaluation, and receives new session identities.
Do not force or overwrite the v1 manifest.

The frozen `exp1_glm4_32b_smoke_v1` run is also retained as a failed integration attempt. It
failed the zero-invalid and AgentDojo proposal-coverage gates. Its precommitted fallback is the
separate `exp1_qwen3_14b_smoke_v1` experiment. That fallback was superseded before execution by
the prospective different-family Granite amendment. Do not force, repair, or overwrite either
the GLM smoke or the archived Qwen selection record.

## Runtime and installation

Use a fresh A100-80 GB Colab runtime and run only one track in that runtime. All configs require
at least 75 GiB reported GPU memory. Qwen3-14B and Ministral require at least 60 GiB free local
disk, Granite requires at least 75 GiB, GPT-OSS requires at least 40 GiB, and GLM-4 requires at
least 90 GiB. Model weights live in ephemeral `/content`; persisted CENSURE artifacts belong on
Drive.

Mount Drive and set the durable output root:

```python
from google.colab import drive
import os

drive.mount("/content/drive")
os.environ["HF_HOME"] = "/content/hf-cache"
os.environ["CENSURE_OUT_ROOT"] = (
    "/content/drive/MyDrive/CENSURE/outputs/exp1_extensions"
)
```

Clone the repository and check out the exact CENSURE commit recorded for the run. Replace the
placeholder with the reviewed extension commit; use the same commit after every restart.

```bash
%cd /content
!git clone https://github.com/Soqoro/Censure.git censure
%cd /content/censure
!git checkout <reviewed-extension-commit-sha>
```

Select exactly one track. For Ministral:

```python
import os

os.environ["CENSURE_MODEL"] = "ministral3_14b_tool_alias_v1"
os.environ["CENSURE_CONFIG"] = (
    "configs/experiments/exp1_ministral3_14b_smoke_v2.yaml"
)
os.environ["CENSURE_REQUIREMENTS"] = (
    "requirements/colab-exp1-ministral3.txt"
)
```

For GPT-OSS:

```python
import os

os.environ["CENSURE_MODEL"] = "gpt_oss_20b"
os.environ["CENSURE_CONFIG"] = (
    "configs/experiments/exp1_gpt_oss_20b_smoke_v1.yaml"
)
os.environ["CENSURE_REQUIREMENTS"] = (
    "requirements/colab-exp1-gpt-oss.txt"
)
```

For the outcome-informed GLM-4 feasibility candidate:

```python
import os

os.environ["CENSURE_MODEL"] = "glm4_32b_0414"
os.environ["CENSURE_CONFIG"] = (
    "configs/experiments/exp1_glm4_32b_smoke_v1.yaml"
)
os.environ["CENSURE_REQUIREMENTS"] = "requirements/colab-exp1-glm4.txt"
```

The GLM-4 selection declaration is `GLM4_EXTENSION_SELECTION.md`. Its smoke is outcome-blind,
but actor selection occurred after prior-model outcomes were known. If accepted, its later
within-model protocol can be frozen prospectively; the combined actor portfolio remains an
outcome-informed breadth extension.

The precommitted Qwen3-14B fallback after the GLM-4 technical failure is archived and must not be
executed under the current protocol. Its identifiers are retained in
`configs/models/qwen3_14b.yaml` and
`configs/experiments/exp1_qwen3_14b_smoke_v1.yaml` for audit only. Its frozen status-only decision
record is `QWEN3_14B_EXTENSION_SELECTION.md`.

For the active Granite 4.1 30B different-family feasibility candidate:

```python
import os

os.environ["CENSURE_MODEL"] = "granite41_30b"
os.environ["CENSURE_CONFIG"] = (
    "configs/experiments/exp1_granite41_30b_smoke_v1.yaml"
)
os.environ["CENSURE_REQUIREMENTS"] = (
    "requirements/colab-exp1-granite.txt"
)
```

Its frozen decision record is `GRANITE41_30B_EXTENSION_SELECTION.md`. It documents the
prospective amendment, immutable model and template identifiers, parser contract, resource cap,
and outcome-blind acceptance gate.

Run the parameterized setup. `CENSURE_REQUIREMENTS` selects the track's frozen requirements while
the script preserves Colab's CUDA PyTorch and checks the configured resource gates. Ministral and
GPT-OSS require their standalone Transformers 5.x locks; GLM-4, Qwen3-14B, and Granite use their
specified Transformers 4.x locks. Do not install the `.[models]` extra on top of a selected track.

```bash
!bash experiments/colab/setup_colab.sh
```

## First run

The doctor verifies the exact package versions, model revision, template hash, A100-80 memory,
free disk, benchmark installation, and output path. The dry run does not fetch model weights.

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage doctor \
  --dry-run \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage doctor \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
```

Preview and freeze the eight-pair manifest before model execution:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --dry-run \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT"
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT"
```

Run both trajectories and automatically write the outcome-blind feasibility report:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
```

On this first pass, the report may show the resume gate as pending; that alone does not fail the
run. Run it once more with `--resume`. Completed checksummed trajectories should be skipped, and
the feasibility report should record a positive resume witness:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL" \
  --resume
```

The acceptance evidence is
`<output-root>/<experiment-id>/feasibility/report.json`: technical completeness, error classes,
checkpoint restoration, proposal coverage, and resume behavior only. Acceptance requires zero
invalid pairs, a captured proposal in both roles for every AgentDojo suite, restorable checkpoints,
and the witnessed resume skip. Do not run `validate` or `analyze` for these outcome-ineligible
smoke IDs.

## Accepted Ministral full extension

After `exp1_ministral3_14b_smoke_v2` passes every feasibility gate, use the separately frozen
prospective protocol in `MINISTRAL_EXTENSION_PROTOCOL.md`. Keep the accepted model runtime and
cache, but switch to the outcome-bearing configuration before freezing its manifest:

```python
os.environ["CENSURE_CONFIG"] = (
    "configs/experiments/exp1_ministral3_14b_full_v1.yaml"
)
```

Preview and freeze the 320-scenario, 672-pair matrix before executing any trajectory:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --dry-run \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
```

Run one shard and one role at a time. Shard 0 of 4 is:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage behavior \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL" \
  --num-shards 4 \
  --shard-index 0 \
  --resume
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage oracle \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL" \
  --num-shards 4 \
  --shard-index 0 \
  --resume
```

Repeat both commands for shard indices `0`, `1`, `2`, and `3`. During execution, inspect only
status counts, error classes, domains, and checkpoint integrity. Do not inspect harm, utility, or
terminal outcomes until the complete 672-pair extension has been persisted and validated.

## Resume after a GPU reset

Remount Drive, clone or reopen the repository at the same recorded CENSURE commit, reselect the
same track, and reinstall its matching lock. The model cache must be downloaded again if work is
pending; completed Drive artifacts do not need to be regenerated.

Verify that the persisted manifest still matches, then resume the same smoke command:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --resume
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL" \
  --resume
```

If all trajectories are already complete, feasibility can be regenerated on a CPU runtime; it
does not load model weights or use the GPU. On that CPU runtime, install the selected requirements
file and CENSURE directly instead of running the GPU-gated setup script:

```bash
!python -m pip install -r "$CENSURE_REQUIREMENTS"
!python -m pip install --no-deps -e .
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage feasibility \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
```

## Validate and analyze the accepted full extension

After all 672 behavior trajectories and all 672 target trajectories for
`exp1_ministral3_14b_full_v1` are persisted, validation and analysis can run on a CPU runtime.
Install the selected requirements file and CENSURE directly as shown above; do not run the
GPU-gated setup script or download model weights.

Validate every frozen pair and restore every saved checkpoint before inspecting outcomes:

```bash
!bash experiments/exp1/validate_exp1.sh \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT"
```

Only after validation reports `ok: true`, no missing/structural issues, and complete runtime
restoration, run the frozen paired analysis:

```bash
!bash experiments/exp1/analyze_exp1.sh \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT"
```

The analysis writes `extension_analysis.json`, embeds the same extension context in
`metrics.json`, and prepends the declaration to `report.md` and `table_masking.tex`. The stage
result must report `inferential_status: prospective_model_breadth_extension` and
`complete_preregistered_actor_matrix: false`; it must not write a pilot go/no-go report.
