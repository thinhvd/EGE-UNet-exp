#!/usr/bin/env bash
# EXP-7: refine E3a (boundary loss) on the ORIGINAL EGE-UNet, ISIC17 + ISIC18, seed 42. Only the loss
# changes. One box per dataset (DATASETS=isic17 on one, DATASETS=isic18 on the other), 6 runs each.
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp07_train.sh
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp07_train.sh 2 3        # only runs 2 and 3
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp07_train.sh   # pre-flight
#
#  #  tag            flags                                                  what it is (exp_docs/07)
#  0  base           (none)                                                 Baseline, retrained in the batch
#  1  bl             --extra-term bl --extra-weight 0.095                   E3a, rerun (same as EXP-6)
#  2  bl+fndp        --extra-term bl,fn_dp --extra-weight 0.095,0.51        X1: E3a + misses weighted by distance to the prediction
#  3  bl+fndp _rep1  (same as 2)                                            X1 repeat (power)
#  4  fndp           --extra-term fn_dp --extra-weight 0.51                 X1-FN ablation: the new term alone
#  5  blhalf         --extra-term bl --extra-weight 0.0475                  X2: E3a at half strength
#
# fn_dp weight 0.51: pre-registered total-push balance on the EXP-6 E3a checkpoints' train images
# (analysis/calibrate_loss_weights.py --mass-balance): 0.4396 (ISIC17), 0.5874 (ISIC18), shared
# geometric mean, 2 digits. Tables in results/EGE-UNet-results-exp7/loss_weights/.
#
# Every run also keeps epoch200..300 every 10 epochs (--save-every 10) to measure checkpoint-choice noise.
# results/ MUST be empty before the real launch (the baseline and E3a run names equal EXP-6's).
# CHECK AT THE START: params 53374 on every run; one rotation angle 230.19364744483815; the "loss:" line.
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

# index|tag|name suffix|loss flags
RUNS=(
    "0|base||"
    "1|bl||--extra-term bl --extra-weight 0.095"
    "2|bl+fndp||--extra-term bl,fn_dp --extra-weight 0.095,0.51"
    "3|bl+fndp|_rep1|--extra-term bl,fn_dp --extra-weight 0.095,0.51"
    "4|fndp||--extra-term fn_dp --extra-weight 0.51"
    "5|blhalf||--extra-term bl --extra-weight 0.0475"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

run_name() {   # dataset, tag, suffix
    if [ "$2" = "base" ]; then echo "egeunet_$1_learnable_s${SEED}$3"
    else echo "egeunet_$1_learnable_loss-$2_s${SEED}$3"; fi
}

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    [ -d "$DATA_PATH/train/images" ] || { echo "missing $DATA_PATH/train/images" >&2; exit 1; }

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r idx tag suffix lossflags <<<"$entry"
        want "$idx" || continue
        name=$(run_name "$DATASET" "$tag" "$suffix")${RUN_SUFFIX}
        args=(--work-dir "results/$name" --dataset "$DATASET" --data-path "$DATA_PATH"
              --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
              --device "$DEVICE" --seed "$SEED" --save-every "$SAVE_EVERY")
        # shellcheck disable=SC2206
        [ -n "$lossflags" ] && args+=($lossflags)

        if [ -n "$PARALLEL" ]; then
            mkdir -p "results/$name"
            echo "launching run $idx ($tag$suffix) on $DATASET: $name -> results/$name/stdout.log"
            setsid nohup python train.py "${args[@]}" > "results/$name/stdout.log" 2>&1 < /dev/null &
            sleep 3
        else
            echo
            echo "#================ run $idx ($tag$suffix) on $DATASET: $name ================#"
            python train.py "${args[@]}"
        fi
    done
done
