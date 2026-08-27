# Experiment 1 on Google Colab

Use one GPU worker. An 80 GB H100 is preferred. A100 can run BF16 subject to memory; T4 is limited
to explicitly labeled quantized smoke tests. Never combine quantized smoke and BF16 primary rows.
Every completed trajectory is persisted to Drive; `/content` holds only the clone, model cache, and
active computation.

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
os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_smoke.yaml"
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
os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_smoke_quantized.yaml"
os.environ["CENSURE_ALLOW_QUANTIZED_SMOKE"] = "1"
```

Then run setup and the smoke command with `exp1_smoke_quantized.yaml`. This uses pinned
bitsandbytes NF4/FP16, writes under the separate `exp1_smoke_quantized` experiment ID, contains no
confirmatory split, and is rejected as primary-analysis input. Restore `CENSURE_MODEL=qwen3_8b`
and use A100/H100 for every BF16 pilot/full command.

## 4. Doctor

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage doctor \
  --config configs/experiments/exp1_smoke.yaml \
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
  --config configs/experiments/exp1_pilot.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --dry-run
```

Freeze before inspecting model outcomes:

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config configs/experiments/exp1_pilot.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

## 6. GPU smoke

The smoke manifest includes at least one AgentDojo scenario from every suite under strict and none,
plus controlled cases.

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config configs/experiments/exp1_smoke.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --resume
```

Run it again: all checksummed completed sessions must be skipped.

T4-only alternative (diagnostic smoke, never a primary result):

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage smoke \
  --config configs/experiments/exp1_smoke_quantized.yaml \
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
  --config configs/experiments/exp1_pilot.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --resume
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage oracle \
  --config configs/experiments/exp1_pilot.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --resume
```

## 8. Validate and analyze

```bash
!bash experiments/exp1/validate_exp1.sh \
  --config configs/experiments/exp1_pilot.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

```bash
!bash experiments/exp1/analyze_exp1.sh \
  --config configs/experiments/exp1_pilot.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

The pilot report answers all six go/no-go questions only from real completed runs. Missing evidence
is `N/A` with a reason; no result is fabricated.

## 9. Full sharded runs

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage manifest \
  --config configs/experiments/exp1_full.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

Run one model and one shard at a time (shown for Qwen, shard 0 of 4):

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage behavior \
  --config configs/experiments/exp1_full.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --num-shards 4 \
  --shard-index 0 \
  --resume
```

```bash
!bash experiments/exp1/run_exp1.sh \
  --stage oracle \
  --config configs/experiments/exp1_full.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model qwen3_8b \
  --num-shards 4 \
  --shard-index 0 \
  --resume
```

Repeat both stages for shard indices `0..3` and models `qwen3_8b`, `llama31_8b`, and
`gemma3_12b`. Before setup for full runs, set
`os.environ["CENSURE_CONFIG"] = "configs/experiments/exp1_full.yaml"`; change
`CENSURE_MODEL`, rerun setup/doctor, and verify access before each gated model.
The full frozen matrix has 2,016 paired sessions / 4,032 trajectories: 960 primary strict→none, 960
degradation sweep pairs on a balanced 25% subset, and 96 identical-strict negative controls.

## 10. Resume after disconnect

Remount Drive, restore the environment variables, update the clone with `git pull --ff-only`, rerun
setup, then repeat the exact command with `--resume`. Partial/corrupt artifacts are not completed.
Use `--retry-failed` only to retry preserved failures; routine recovery does not need `--force`.

## 11. Locate and download tables/figures

```python
from pathlib import Path
import os

root = Path(os.environ["CENSURE_OUT_ROOT"])
for path in sorted(root.rglob("results/exp1/*")):
    print(path)
```

```python
from google.colab import files
import shutil, os

results_dir = os.path.join(os.environ["CENSURE_OUT_ROOT"], "exp1_full", "results", "exp1")
archive = shutil.make_archive("/content/censure-exp1-results", "zip", results_dir)
files.download(archive)
```

Expected artifacts are `metrics.json`, `paired_runs.parquet`, three CSV summaries,
`table_masking.tex`, `report.md`, and PNG/PDF figures for behavior-vs-target risk, masking gaps, and
ranking reversals.
