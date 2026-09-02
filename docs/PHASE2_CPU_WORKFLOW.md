# Phase 2 CPU calibration workflow

Experiments 2 and 3 are model-free and should run on CPU. Do not reserve a GPU for this stage.
The frozen catalog contains 171 unique cells, 342,000 repetitions, and 13,680 atomic chunks of 25
repetitions. Results are checksummed per chunk and are safe to resume after a Colab runtime reset.

## Fresh Colab CPU runtime

```bash
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
set -euo pipefail
git clone https://github.com/Soqoro/Censure.git /content/censure
cd /content/censure
python -m pip install -e '.[analysis]'
python -m censure.estimation.cli catalog
```

The catalog must report SHA-256
`9464daadf2c76e2257fbb4b26eb1f7b657bb9474f842cf6646dc3a038423e007`, 171
unique cells, and 13,680 work items. If it does not, stop rather than mixing protocol versions.

## Run or resume one shard

Set `NUM_SHARDS` to the number of CPU workers/runtimes you intend to use and give each worker a
different zero-based `SHARD_INDEX`. Keep `NUM_SHARDS` fixed until all shards finish.

```bash
%%bash
set -euo pipefail
cd /content/censure

OUT_ROOT=/content/drive/MyDrive/CENSURE/outputs/phase2
NUM_SHARDS=16
SHARD_INDEX=0

python -m censure.estimation.cli run-calibration \
  --out-root "$OUT_ROOT" \
  --experiment-id phase2_estimator_v1 \
  --purpose all \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --resume \
  --progress-every 10
```

Re-running the identical command with `--resume` skips checksum-valid chunks. `--max-work-items N`
may be added to bound one session; it changes only when the process stops, not any scientific row.

## Check progress

```bash
%%bash
set -euo pipefail
cd /content/censure

OUT_ROOT=/content/drive/MyDrive/CENSURE/outputs/phase2
NUM_SHARDS=16

for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  python -m censure.estimation.cli calibration-status \
    --out-root "$OUT_ROOT" \
    --experiment-id phase2_estimator_v1 \
    --purpose all \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
done
```

Every shard must show `remaining: 0` before analysis.

## Summarize

This command refuses partial cells and therefore should run only after all shards finish.

```bash
%%bash
set -euo pipefail
cd /content/censure

python -m censure.estimation.cli summarize-calibration \
  --out-root /content/drive/MyDrive/CENSURE/outputs/phase2 \
  --experiment-id phase2_estimator_v1 \
  --purpose all
```

The combined result is written under
`phase2_estimator_v1/phase2/calibration/results/all_summary.json`. Keep the raw chunk directory;
the summary is reproducible from it and is not a substitute for the audit ledgers.

## Run Experiment 5 robustness sweeps

The robustness catalog is separate: 21 one-factor cells and 1,680 chunks. Verify it first:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli robustness-catalog
```

It must report catalog SHA-256
`0b3ca69e6216fd070ea901df24c866890b1c6861f84b4a814b68f31572dabf23`.
Run each shard with the same output root and shard convention used above:

```bash
%%bash
set -euo pipefail
cd /content/censure

python -m censure.estimation.cli run-robustness \
  --out-root /content/drive/MyDrive/CENSURE/outputs/phase2 \
  --experiment-id phase2_estimator_v1 \
  --num-shards 16 \
  --shard-index 0 \
  --resume \
  --progress-every 10
```

Use `robustness-status` with the same shard arguments to check completion. After every shard has
zero remaining chunks, summarize:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli summarize-robustness \
  --out-root /content/drive/MyDrive/CENSURE/outputs/phase2 \
  --experiment-id phase2_estimator_v1
```

## Run the shared-support OPE sweep

Verify the 12-cell catalog and its SHA-256
`b47beb22fbc660c7d0577db4340bf25bb1cee73471da5fe70d7496515d170abe`:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli shared-support-catalog
```

Then run the 960 chunks (again CPU-only):

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli run-shared-support \
  --out-root /content/drive/MyDrive/CENSURE/outputs/phase2 \
  --experiment-id phase2_estimator_v1 \
  --num-shards 16 \
  --shard-index 0 \
  --resume \
  --progress-every 10
```

Use `shared-support-status` to verify each shard, then run:

```bash
%%bash
set -euo pipefail
cd /content/censure
python -m censure.estimation.cli summarize-shared-support \
  --out-root /content/drive/MyDrive/CENSURE/outputs/phase2 \
  --experiment-id phase2_estimator_v1
```
