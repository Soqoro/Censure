# Experiment 1 on Google Colab

Use one GPU worker. An 80 GB H100 is preferred. A100 can run BF16 subject to memory; T4 is limited
to explicitly labeled quantized smoke tests. Never combine quantized smoke and BF16 primary rows.
Every completed trajectory is persisted to Drive; `/content` holds only the clone, model cache, and
active computation.

The active protocol uses CENSURE-Control scenario v2. The original configs without a `_v2`
suffix remain available only to reconstruct the archived pilot-v1 manifest; do not overwrite or
mix their outputs with v2. Version 2 makes every authorized task parameter explicit and projects
the separately frozen external/untrusted context into the actor-visible prompt.

## 1. Mount Drive and load secrets

Create a Colab secret named `HF_TOKEN`; never print it. Accept the Llama and Gemma licenses on
Hugging Face before selecting those gated models.

```python
from google.colab import drive, userdata
import os

drive.mount("/content/drive")

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_HOME"] = "/content/hf-cache"
os.environ["CENSURE_OUT_ROOT"] = (
    "/content/drive/MyDrive/CENSURE/outputs/exp1"
)
```

`!export` does not persist between notebook cells. Use `os.environ`, `%env`, or command flags.

## 2. Clone or update

First session:

```bash
%cd /content
!git clone https://github.com/Soqoro/Censure.git censure
%cd /content/censure
```

Later sessions:

```bash
%cd /content/censure
!git pull --ff-only
```

## 3. Install

```python
import os
os.environ["CENSURE_MODEL"] = "qwen3_8b"
os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_smoke_v2.yaml"
```

```bash
!bash experiments/colab/setup_colab.sh
```

Setup preserves Colab's CUDA PyTorch; checks disk, GPU, CUDA, BF16, and memory; verifies AgentDojo
`0.1.35`/benchmark `v1.2.2`; checks the selected frozen model revision and Drive writes; and runs a
synthetic guarded tool round trip. It refuses CPU fallback.

For a T4, use only the separately keyed 4-bit smoke configuration before running setup:

```python
import os
os.environ["CENSURE_MODEL"] = "qwen3_8b_4bit_smoke"
os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_smoke_quantized_v2.yaml"
os.environ["CENSURE_ALLOW_QUANTIZED_SMOKE"] = "1"
```

Then run setup and the smoke command with `exp1_smoke_quantized_v2.yaml`. This uses pinned
bitsandbytes NF4/FP16, writes under the separate `exp1_smoke_quantized_v2` experiment ID, contains no
confirmatory split, and is rejected as primary-analysis input. Restore `CENSURE_MODEL=qwen3_8b`
and use A100/H100 for every BF16 pilot/full command.

## 4. Doctor

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage doctor \
  --config configs/experiments/exp1_smoke_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b
```

The hardware messages distinguish T4, A100, and H100 and fail if the selected primary mode cannot
run as configured.

## 5. Manifest dry-run and freeze

Dry-run checks deterministic balance and reports shortages without writing:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config configs/experiments/exp1_pilot_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --dry-run
```

Freeze before inspecting model outcomes:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config configs/experiments/exp1_pilot_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

## 6. GPU smoke

The smoke manifest includes at least one AgentDojo scenario from every suite under strict and none,
plus controlled cases.

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config configs/experiments/exp1_smoke_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --resume
```

Run it again: all checksummed completed sessions must be skipped.

T4-only alternative (diagnostic smoke, never a primary result):

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config configs/experiments/exp1_smoke_quantized_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b_4bit_smoke \
  --resume
```

## 7. Pilot behavior and oracle

The frozen pilot contains 32 base scenarios expanded to 40 paired sessions / 80 full
trajectories: 32 primary strict→none pairs plus eight identical-strict controls.

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage behavior \
  --config configs/experiments/exp1_pilot_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --resume
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage oracle \
  --config configs/experiments/exp1_pilot_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --resume
```

## 8. Validate and analyze

```bash
!bash experiments/exp1/validate_exp1.sh \
  --config configs/experiments/exp1_pilot_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

```bash
!bash experiments/exp1/analyze_exp1.sh \
  --config configs/experiments/exp1_pilot_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

The pilot report answers all six go/no-go questions only from real completed runs. Missing evidence
is `N/A` with a reason; no result is fabricated.

## 9. Full sharded runs

Before starting Gemma's full shards, run its separately keyed BF16 adapter smoke. These 16
trajectories are diagnostic only and are never combined with `exp1_full_v2`:

```python
import os
os.environ["CENSURE_MODEL"] = "gemma3_12b"
os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_gemma_smoke_v3.yaml"
```

```bash
!bash experiments/colab/setup_colab.sh
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage doctor \
  --config configs/experiments/exp1_gemma_smoke_v3.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model gemma3_12b
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config configs/experiments/exp1_gemma_smoke_v3.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model gemma3_12b \
  --resume
```

The smoke must validate all eight pairs, capture a pre-guard proposal in both trajectories for all
four AgentDojo suites, and have no parser or template errors before proceeding. Keep its output
under `exp1_gemma_smoke_v3`; it is explicitly ineligible for primary analysis. The preserved
`exp1_gemma_smoke_v2` run diagnosed Gemma's Markdown-fenced tool-call syntax and must not be retried
or treated as conformance evidence.

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config configs/experiments/exp1_full_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

Run one model and one shard at a time (shown for Qwen, shard 0 of 4):

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage behavior \
  --config configs/experiments/exp1_full_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --num-shards 4 \
  --shard-index 0 \
  --resume
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage oracle \
  --config configs/experiments/exp1_full_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --num-shards 4 \
  --shard-index 0 \
  --resume
```

Repeat both stages for shard indices `0..3` and models `qwen3_8b`, `llama31_8b`, and
`gemma3_12b`. Before setup for full runs, set
`os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_full_v2.yaml"`; change
`CENSURE_MODEL`, rerun setup/doctor, and verify access before each gated model.
The full frozen matrix has 2,016 paired sessions / 4,032 trajectories: 960 primary strict→none, 960
degradation sweep pairs on a balanced 25% subset, and 96 identical-strict negative controls.

## 10. Resume after disconnect

Remount Drive, restore the environment variables, update the clone with `git pull --ff-only`, rerun
setup, then repeat the exact command with `--resume`. Partial/corrupt artifacts are not completed.
Use `--retry-failed` only to retry preserved failures; routine recovery does not need `--force`.
When an adapter correction applies to one known failure class, add a repeatable exact filter such as
`--retry-error-type ToolCallParseError` so unrelated environment failures remain untouched.

## 11. Explicit Qwen + Gemma partial analysis

The frozen `qwen_gemma_v1` scope records the status-only decision to defer Llama after shard 0. It
does not change `exp1_full_v2`, silently substitute an actor, or claim completion of the original
three-actor matrix. The scope also records that applying the pilot's 10% threshold per actor is a
post-hoc protocol deviation. Run validation and analysis without `--model` or extra filters:

```bash
!bash experiments/exp1/validate_exp1.sh \
  --config configs/experiments/exp1_full_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --analysis-scope configs/analysis/exp1_qwen_gemma_v1.yaml
```

```bash
!bash experiments/exp1/analyze_exp1.sh \
  --config configs/experiments/exp1_full_v2.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --analysis-scope configs/analysis/exp1_qwen_gemma_v1.yaml
```

The scoped validation must select 1,344 pairs and report no missing trajectories or structural
issues. Outputs are isolated under
`exp1_full_v2/results/exp1_scopes/qwen_gemma_v1/`; `analysis_scope.json`, `metrics.json`, and the
top of `report.md` all carry the partial-analysis limitation. Running unscoped full analysis while
the Llama arm is incomplete fails closed instead of consuming stale single-model validation rows.
Model inference is finished, so these commands do not require a GPU.

## 12. Locate and download tables/figures

```python
from pathlib import Path
import os

root = Path(os.environ["CENSURE_OUT_ROOT"])
for path in sorted(root.rglob("results/exp1_scopes/qwen_gemma_v1/*")):
    print(path)
```

```python
from google.colab import files
import shutil, os

results_dir = os.path.join(
    os.environ["CENSURE_OUT_ROOT"],
    "exp1_full_v2",
    "results",
    "exp1_scopes",
    "qwen_gemma_v1",
)
archive = shutil.make_archive("/content/censure-exp1-qwen-gemma-results", "zip", results_dir)
files.download(archive)
```

Expected artifacts are `metrics.json`, `paired_runs.parquet`, four CSV summaries (including
`missing_harm_bounds.csv`),
`table_masking.tex`, `report.md`, and PNG/PDF figures for behavior-vs-target risk, masking gaps, and
ranking reversals.
