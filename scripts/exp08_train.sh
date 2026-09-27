#!/usr/bin/env bash
# EXP-8: EXP-7's X2 loss (boundary loss at half strength, --extra-term bl --extra-weight 0.0475) applied to
# the two GHPA-free fusion variants of EXP-4/5, next to the author's original model and loss.
# ISIC17 + ISIC18, seed 42, seed-determined rotation. One box per dataset, 5 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp08_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp08_train.sh   # pre-flight
#
#  #  name suffix                                   params  model                                   loss
#  0  learnable                                     53374   original EGE-UNet (GHPA on)             original (BCE + Dice)
#  1  none_fuse-sum-deep3                           49059   GHPA off, sum fusion into dec1..dec3     original
#  2  none_fuse-sum-deep3_loss-blhalf               49059   same                                    original + 0.0475 * bl (X2)
#  3  none_fuse-sum_attn-all5                       51109   GHPA off, sum + attention into dec1..5   original
#  4  none_fuse-sum_attn-all5_loss-blhalf           51109   same                                    original + 0.0475 * bl (X2)
#
# Runs 1 and 3 (architecture with the original loss) separate the effect of the loss from that of the
# architecture: X2's effect on an architecture = run 2 - run 1 (resp. 4 - 3).
# Every run keeps epoch200..300 every 10 epochs (--save-every 10) for checkpoint-choice noise.
# results/ MUST be empty before the real launch.
# CHECK AT THE START: "params:" matches the table; one rotation angle 230.19364744483815; the "loss:" line.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
DEVICE=${DEVICE:-cuda}
PARALLEL=${PARALLEL:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
SAVE_EVERY=${SAVE_EVERY:-10}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

# index|name suffix (after egeunet_<ds>_)|expected params|flags
RUNS=(
    "0|learnable|53374|--hpa-mode learnable"
    "1|none_fuse-sum-deep3|49059|--hpa-mode none --fusion sum --fusion-stages deep3"
    "2|none_fuse-sum-deep3_loss-blhalf|49059|--hpa-mode none --fusion sum --fusion-stages deep3 --extra-term bl --extra-weight 0.0475"
    "3|none_fuse-sum_attn-all5|51109|--hpa-mode none --fusion sum_attn --fusion-stages all5"
    "4|none_fuse-sum_attn-all5_loss-blhalf|51109|--hpa-mode none --fusion sum_attn --fusion-stages all5 --extra-term bl --extra-weight 0.0475"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    [ -d "$DATA_PATH/train/images" ] || { echo "missing $DATA_PATH/train/images" >&2; exit 1; }

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r idx suffix params flags <<<"$entry"
        want "$idx" || continue
        name="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        args=(--work-dir "results/$name" --dataset "$DATASET" --data-path "$DATA_PATH"
              --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
              --device "$DEVICE" --seed "$SEED" --save-every "$SAVE_EVERY")
        # shellcheck disable=SC2206
        args+=($flags)

        if [ -n "$PARALLEL" ]; then
            mkdir -p "results/$name"
            echo "launching run $idx on $DATASET: $name (expect $params params) -> results/$name/stdout.log"
            setsid nohup python train.py "${args[@]}" > "results/$name/stdout.log" 2>&1 < /dev/null &
            sleep 3
        else
            echo
            echo "#================ run $idx on $DATASET: $name (expect $params params) ================#"
            python train.py "${args[@]}"
        fi
    done
done
