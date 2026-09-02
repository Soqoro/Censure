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
`2df2c87aa604979572b06816aa21227dc06fc78544390d05fd59f40d163102fd`, 171
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
