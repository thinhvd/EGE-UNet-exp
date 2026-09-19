#!/usr/bin/env bash
# EXP-4: cross-stage fusion run matrix (ISIC17, seed 42).
#
#   bash scripts/exp04_train.sh                 # all runs, sequentially
#   bash scripts/exp04_train.sh 1 3             # only runs 1 and 3 (see the table)
#   PARALLEL=1 bash scripts/exp04_train.sh      # launch every run at once, detached
#   PARALLEL=1 bash scripts/exp04_train.sh 0 1  # launch a subset, detached
#
# Each run writes to results/<name>/ and auto-resumes from checkpoints/latest.pth if interrupted,
# so re-running the same command after a disconnect continues where it stopped.
#
#  #  name                        params   what it isolates
#  0  learnable (baseline)         53374   the original model, retrained HERE (see note)
#  0b none                         44988   no GHPA gate at all, retrained HERE
#  1  fuse-sum-deep3               57445   extra decoder path, minimal capacity (control)
#  2  fuse-concat-deep3            64086   full linear cross-scale mixing (control)
#  3  fuse-concat-all5             66030   the same, on every decoder stage
#  4  fuse-sum-all5                57863   control, every decoder stage
#  5  fuse-csaa-deep3              65718   concat + cross-stage attention (+1632 params)
#  6  fuse-csaa-all5               67662   the same, on every decoder stage
#
# WHY THE BASELINES ARE RETRAINED HERE. The EXP-1..3 numbers (learnable 80.35 mIoU, none 78.17)
# come from Colab T4 with an older PyTorch. This machine has a different GPU and a different torch
# build, and the two disagree in floating-point detail: the same seed-0 model produces a forward sum
# of 77392.56770953648 here versus 77392.56237548799 there (same weights, same code - a ~7e-8
# relative difference from different kernels). Over 300 epochs that grows chaotically, so comparing
# a fusion run trained here against a baseline trained there would mix the fusion effect with an
# environment effect. Retraining both baselines in this environment costs two runs and makes every
# comparison within-environment.
#
# CHECK AT THE START OF EVERY RUN: the log line must read the params count from the table above.
# "params: 53374" on a fusion run means the flags did not reach the model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASET=${DATASET:-isic17}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}   # EXP-1..3 all ran with 0; changing it changes the augmentation RNG stream
DEVICE=${DEVICE:-cuda}
PARALLEL=${PARALLEL:-}

# One process per run, each defaulting to a thread per core, means ~8x oversubscription when the
# matrix runs in parallel: the runs then spend their time fighting over cores (observed: ~1570
# threads on 128 cores, 220% CPU per process, GPU idle, ~20 s per iteration). The model trains on
# the GPU and the loader is single-process, so a couple of CPU threads per run is plenty. Affects
# speed only - the CPU-side work is deterministic image transforms, not thread-order reductions.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

# index|hpa_mode|fusion mode|stages|expected params
RUNS=(
    "0|learnable|none|-|53374"
    "0b|none|none|-|44988"
    "1|learnable|sum|deep3|57445"
    "2|learnable|concat|deep3|64086"
    "3|learnable|concat|all5|66030"
    "4|learnable|sum|all5|57863"
    "5|learnable|csaa|deep3|65718"
    "6|learnable|csaa|all5|67662"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

run_name() {   # hpa_mode, fusion mode, stages
    if [ "$2" = "none" ]; then echo "egeunet_${DATASET}_$1_s${SEED}"
    else echo "egeunet_${DATASET}_$1_fuse-$2-$3_s${SEED}"; fi
}

for entry in "${RUNS[@]}"; do
    IFS='|' read -r idx hpa mode stages params <<<"$entry"
    want "$idx" || continue
    name=$(run_name "$hpa" "$mode" "$stages")
    args=(--work-dir "results/$name" --dataset "$DATASET" --epochs "$EPOCHS"
          --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE"
          --seed "$SEED" --hpa-mode "$hpa")
    [ "$mode" != "none" ] && args+=(--fusion "$mode" --fusion-stages "$stages")

    if [ -n "$PARALLEL" ]; then
        mkdir -p "results/$name"
        echo "launching run $idx: $name (expect $params params) -> results/$name/stdout.log"
        setsid nohup python train.py "${args[@]}" > "results/$name/stdout.log" 2>&1 < /dev/null &
        sleep 3      # stagger so the runs do not all hit the dataset at the same instant
    else
        echo
        echo "#================ run $idx: $name (expect $params params) ================#"
        python train.py "${args[@]}"
    fi
done

if [ -n "$PARALLEL" ]; then
    cat <<'EOF'

Launched detached. Watch them with:
  tail -f results/*/stdout.log | grep -E "params:|epoch|loss"
  grep -h "params:" results/*/stdout.log        # verify every run got its flags
  for d in results/*/; do echo -n "$d "; tail -1 "$d/metrics.csv" 2>/dev/null; done
EOF
else
    cat <<'EOF'

#---------- after training ----------#
# GPU cost table (run on the server, where the GPU is):
python analysis/benchmark_speed.py --device cuda --out results/benchmark_gpu.json \
    --checkpoint results/egeunet_isic17_learnable_s42 --label learnable \
    --checkpoint results/egeunet_isic17_none_s42      --label none \
    --checkpoint results/egeunet_isic17_learnable_fuse-sum-deep3_s42    --label sum-deep3 \
    --checkpoint results/egeunet_isic17_learnable_fuse-sum-all5_s42     --label sum-all5 \
    --checkpoint results/egeunet_isic17_learnable_fuse-concat-deep3_s42 --label concat-deep3 \
    --checkpoint results/egeunet_isic17_learnable_fuse-concat-all5_s42  --label concat-all5 \
    --checkpoint results/egeunet_isic17_learnable_fuse-csaa-deep3_s42   --label csaa-deep3 \
    --checkpoint results/egeunet_isic17_learnable_fuse-csaa-all5_s42    --label csaa-all5

# then pull everything to the local machine and analyse there:
#   SERVER=user@host bash scripts/sync_results.sh
EOF
fi
