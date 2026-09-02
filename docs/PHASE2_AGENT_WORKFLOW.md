# Phase 2 held-out agent workflow

This is the prospective GPU workflow for `phase2_held_out_agents_v1`. Model execution is pinned to
commit `b31ac4549692a828151e0002400e3604ae8437d0`. Drive contains all durable artifacts, so a Colab
runtime or GPU may be released after any command exits. Never copy the output tree to a different
experiment ID and never run the full target stage before the audit seal exists.

The enforced order is:

`behavior -> cohort freeze -> selected suffix audits -> audit seal -> full targets -> analysis`.

## Constants

Use this output root for every runtime:

```python
import os
from google.colab import drive

drive.mount("/content/drive")
os.environ["CENSURE_OUT_ROOT"] = "/content/drive/MyDrive/CENSURE/outputs/phase2_agents"
os.environ["CENSURE_CONFIG"] = "configs/experiments/phase2_held_out_agents_v1.yaml"
# Set this through Colab Secrets for Gemma and any rate-limited Hub download.
# os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

The durable experiment directory will be
`$CENSURE_OUT_ROOT/phase2_held_out_agents_v1`. Do not delete it when resetting the runtime.

## Fresh-runtime bootstrap

Run this after every Colab reset. It is intentionally safe only when `/content/censure` does not
already contain work you need.

```bash
%%bash
set -euo pipefail
git clone https://github.com/Soqoro/Censure.git /content/censure
cd /content/censure
git checkout --detach b31ac4549692a828151e0002400e3604ae8437d0
git status --short
git rev-parse HEAD
```

The final command must print exactly
`b31ac4549692a828151e0002400e3604ae8437d0`, and `git status --short` must be empty.

## 1. Freeze the manifest once

Use a Qwen-compatible A100 runtime for the initial doctor and manifest. The same bootstrap is also
the restart sequence for Qwen and Gemma; only `CENSURE_MODEL` changes.

```python
import os
os.environ["CENSURE_MODEL"] = "qwen3_8b"
os.environ["CENSURE_REQUIREMENTS"] = "requirements/colab-exp1.txt"
```

```bash
%%bash
set -euo pipefail
cd /content/censure
bash experiments/colab/setup_colab.sh
python -m censure.cli \
  --stage doctor \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$CENSURE_MODEL"
python -m censure.cli \
  --stage manifest \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT"
```

The manifest must report 160 scenarios, 480 paired sessions, and SHA-256
`e4a7b11680ed0d6181f24b3bf5c26420453503122b032fc302733fbb7bdfb96d`. On every later fresh
runtime, verify and reuse it with the same command plus `--resume`.

## 2. Run all behavior trajectories

Run one actor at a time. Every command is restart-safe with `--resume`.

### Qwen3-8B

```python
import os
os.environ["CENSURE_MODEL"] = "qwen3_8b"
os.environ["CENSURE_REQUIREMENTS"] = "requirements/colab-exp1.txt"
```

```bash
%%bash
set -euo pipefail
cd /content/censure
bash experiments/colab/setup_colab.sh
python -m censure.cli --stage manifest --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" --resume
python -m censure.cli --stage behavior --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" --model "$CENSURE_MODEL" --resume
```

### Gemma-3-12B

Start a fresh runtime, repeat the bootstrap, and run:

```python
import os
os.environ["CENSURE_MODEL"] = "gemma3_12b"
os.environ["CENSURE_REQUIREMENTS"] = "requirements/colab-exp1.txt"
```

```bash
%%bash
set -euo pipefail
cd /content/censure
bash experiments/colab/setup_colab.sh
python -m censure.cli --stage manifest --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" --resume
python -m censure.cli --stage behavior --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" --model "$CENSURE_MODEL" --resume
```

### Ministral-3-14B

Ministral requires its isolated Transformers 5 lock and an A100-80GB/H100 runtime:

```python
import os
os.environ["CENSURE_MODEL"] = "ministral3_14b_tool_alias_v1"
os.environ["CENSURE_REQUIREMENTS"] = "requirements/colab-exp1-ministral3.txt"
```

```bash
%%bash
set -euo pipefail
cd /content/censure
bash experiments/colab/setup_colab.sh
python -m censure.cli --stage manifest --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" --resume
python -m censure.cli --stage behavior --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" --model "$CENSURE_MODEL" --resume
```

Each actor must end with `selected: 160`. Failed behavior artifacts are retained and treated as
harmful; do not selectively rerun or remove them based on harm outcomes.

## 3. Freeze behavior-derived cohorts on CPU

Release the GPU. In a fresh CPU runtime, mount Drive, bootstrap the pinned commit, and install the
environment runtime:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m pip install -e '.[agentdojo]'
python -m censure.estimation.cli freeze-agent-cohort \
  --config "$CENSURE_CONFIG" \
  --freeze configs/experiments/phase2_held_out_agents_v1.freeze.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

This command refuses to freeze if any behavior artifact is missing or any full target artifact
already exists. Save the printed collection hash with the experiment log.

## 4. Run selected suffix audits

Return to one appropriate GPU runtime per actor and use the same model-specific setup from step 2.
Replace `MODEL_ALIAS` below with exactly one of `qwen3_8b`, `gemma3_12b`, or
`ministral3_14b_tool_alias_v1`:

```bash
%%bash
set -euo pipefail
cd /content/censure
MODEL_ALIAS=qwen3_8b
python -m censure.estimation.cli run-agent-audits \
  --config "$CENSURE_CONFIG" \
  --freeze configs/experiments/phase2_held_out_agents_v1.freeze.yaml \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$MODEL_ALIAS" \
  --policy all \
  --resume
```

The command runs all six allocation policies to the frozen 40% maximum budget. It generates only
newly selected suffixes, reuses checksummed selected outcomes, and will not load the model at all
when every requested policy is already complete.

After all three actors, inspect the ledgers on CPU:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli agent-audit-status \
  --config "$CENSURE_CONFIG" \
  --freeze configs/experiments/phase2_held_out_agents_v1.freeze.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

Proceed only when `all_maximum_budget_ledgers_complete` is `true`, every row has `complete: true`,
and `full_target_trajectory_count` is zero.

## 5. Seal the ledgers before target release

This CPU-only command is the irreversible outcome-release boundary:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli seal-agent-audits \
  --config "$CENSURE_CONFIG" \
  --freeze configs/experiments/phase2_held_out_agents_v1.freeze.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

It verifies all 18 actor-policy ledgers and their certificate paths, refuses preexisting full
targets, and writes a checksummed audit seal. The ordinary target command remains locked until this
seal validates.

## 6. Generate the post-seal full target matrix

For each actor, start its matching fresh GPU runtime, run setup, verify the manifest with
`--resume`, then run:

```bash
%%bash
set -euo pipefail
cd /content/censure
MODEL_ALIAS=qwen3_8b
python -m censure.cli \
  --stage oracle \
  --config "$CENSURE_CONFIG" \
  --out-root "$CENSURE_OUT_ROOT" \
  --model "$MODEL_ALIAS" \
  --resume
```

Repeat for Gemma and Ministral with their respective aliases and dependency locks. Each actor must
end with `selected: 160`. Invalid targets remain part of the frozen matrix.

## 7. Summarize on CPU

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m pip install -e '.[analysis,agentdojo]'
python -m censure.estimation.cli summarize-agent-audits \
  --config "$CENSURE_CONFIG" \
  --freeze configs/experiments/phase2_held_out_agents_v1.freeze.yaml \
  --out-root "$CENSURE_OUT_ROOT"
```

The summarizer refuses a missing target artifact, incomplete ledger, corrupt checksum, or seal
mismatch. The primary output is
`phase2_held_out_agents_v1/phase2/agent_audits/study_summary.json`, accompanied by
`study_summary.sha256`. The CPU workflow contains the final cross-study synthesis command.

## Reset and recovery rules

- A GPU reset never requires deleting or re-freezing Drive artifacts.
- Clone and detach at the pinned commit after every reset, restore the same environment variables,
  run the model-specific setup, verify the manifest with `--resume`, and rerun the interrupted
  stage with `--resume`.
- Never use `--force` in Phase 2.
- Never retry only favorable or unfavorable failures. The frozen retry count is already applied
  inside each trajectory/suffix execution.
- Do not edit cohort, ledger, selected-suffix, seal, or target files. Checksum failure is a hard
  stop, not a reason to skip the unit.
