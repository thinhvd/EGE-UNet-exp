#!/usr/bin/env bash
# EXP-4: cross-stage fusion run matrix (6 runs, ISIC17, seed 42).
#
#   bash scripts/exp04_train.sh              # all six, in priority order
#   bash scripts/exp04_train.sh 1 3          # only runs 1 and 3 (see the table below)
#   EPOCHS=300 bash scripts/exp04_train.sh   # override a training setting
#
# Each run writes to results/<name>/ and auto-resumes from checkpoints/latest.pth if interrupted,
# so re-running the same command after a disconnect continues where it stopped.
#
#  #  name                              params   what it isolates
#  1  fuse-sum-deep3                     57445   extra decoder path, minimal capacity (control)
#  2  fuse-concat-deep3                  64086   full linear cross-scale mixing (control)
#  3  fuse-concat-all5                   66030   the same, on every decoder stage
#  4  fuse-sum-all5                      57863   control, every decoder stage
#  5  fuse-csaa-deep3                    65718   concat + cross-stage attention (+1632 params)
#  6  fuse-csaa-all5                     67662   the same, on every decoder stage
#
# Baselines already trained (do not re-run): egeunet_isic17_learnable_s42 (53374 params, 80.35 mIoU),
# egeunet_isic17_none_s42 (44988 params, 78.17 mIoU).
#
# CHECK AT THE START OF EVERY RUN: the log line must read the params count from the table above.
# "params: 53374" means the fusion flags did not reach the model and the run is just the baseline.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASET=${DATASET:-isic17}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}

# index|mode|stages|expected params
RUNS=(
    "1|sum|deep3|57445"
    "2|concat|deep3|64086"
    "3|concat|all5|66030"
    "4|sum|all5|57863"
    "5|csaa|deep3|65718"
    "6|csaa|all5|67662"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

for entry in "${RUNS[@]}"; do
    IFS='|' read -r idx mode stages params <<<"$entry"
    want "$idx" || continue
    name="egeunet_${DATASET}_learnable_fuse-${mode}-${stages}_s${SEED}"
    echo
    echo "#================ run $idx: $name (expect $params params) ================#"
    python train.py \
        --work-dir "results/$name" \
        --dataset "$DATASET" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --fusion "$mode" \
        --fusion-stages "$stages"
done

cat <<'EOF'

#---------- after training ----------#
# GPU cost table (run on the server, where the GPU is):
python analysis/benchmark_speed.py --device cuda --out results/benchmark_gpu.json \
    --checkpoint results/egeunet_isic17_learnable_s42            --label learnable \
    --checkpoint results/egeunet_isic17_none_s42                 --label none \
    --checkpoint results/egeunet_isic17_learnable_fuse-sum-deep3_s42    --label sum-deep3 \
    --checkpoint results/egeunet_isic17_learnable_fuse-sum-all5_s42     --label sum-all5 \
    --checkpoint results/egeunet_isic17_learnable_fuse-concat-deep3_s42 --label concat-deep3 \
    --checkpoint results/egeunet_isic17_learnable_fuse-concat-all5_s42  --label concat-all5 \
    --checkpoint results/egeunet_isic17_learnable_fuse-csaa-deep3_s42   --label csaa-deep3 \
    --checkpoint results/egeunet_isic17_learnable_fuse-csaa-all5_s42    --label csaa-all5

# then pull everything to the local machine and analyse there:
#   SERVER=user@host bash scripts/sync_results.sh
EOF
