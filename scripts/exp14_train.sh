#!/usr/bin/env bash
# EXP-14: interior hard-positive loss on the cross-stage fusion, under the new 70:10:20 protocol
# (train 70 % / val 10 % for checkpoint selection / test 20 % held out; scripts/make_split_701020.py).
# ISIC17 + ISIC18, seed 42, seed-determined rotation. One box per dataset, 5 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp14_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp14_train.sh   # pre-flight
#
#  #  arm  name suffix                                   params  model        loss
#  0  A0   learnable                                     53374   original     original (BCE + Dice)
#  1  A1   learnable_fuse-sum-shallow3-d8                54965   + CSF        original
#  2  A2   learnable_fuse-sum-shallow3-d8_loss-ihp       54965   + CSF        + lambda * interior hard-positive (focal BCE on lesion
#                                                                                pixels with normalized depth >= tau and u < 0.5)
#  3  A3   learnable_fuse-sum-shallow3-d8_loss-focal     54965   + CSF        + lambda * focal loss (alpha 0.75 on class 1, gamma 2)
#  4  A4   learnable_fuse-sum-shallow3-d8_loss-tversky   54965   + CSF        + lambda * Tversky loss (alpha 0.3 FP, beta 0.7 FN)
#
# A3 and A4 weight misses up everywhere; A2 only in the interior where the model currently misses.
# tau and the three lambdas were fixed on train / val before any run (analysis/exp14_step0.py,
# analysis/exp14_lambda_sweep.py; doc 14 section 2). Checkpoint selection: original loss on the val split.
# The final test (test_results.json) runs on the test split (--test-split test).
# Every run keeps epoch200..300 every 10 epochs (--save-every 10). results/ MUST be empty before the real launch.
# CHECK AT THE START: "params:" matches the table; one rotation angle 230.19364744483815; the "loss:" line;
# "test split: test (434 | 538 images); checkpoint selection on val (216 | 270 images)" at the end.
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
RESULTS_ROOT=${RESULTS_ROOT:-results}            # local smoke tests point this elsewhere
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset with train / val / test

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

# fixed on train / val by step 0
TAU=0.15
LAM_IHP=1.3
LAM_FOCAL=0.7
LAM_TVERSKY=0.37

CSF="--fusion sum --fusion-stages shallow3 --fusion-dim 8"
# index|name suffix (after egeunet_<ds>_)|expected params|expected loss (the "loss:" log line)|flags
RUNS=(
    "0|learnable|53374|bcedice|--hpa-mode learnable"
    "1|learnable_fuse-sum-shallow3-d8|54965|bcedice|--hpa-mode learnable $CSF"
    "2|learnable_fuse-sum-shallow3-d8_loss-ihp|54965|bcedice + $LAM_IHP * ihp (ihp depth>=$TAU, p<0.5, gamma 2)|--hpa-mode learnable $CSF --extra-term ihp --extra-weight $LAM_IHP --ihp-tau $TAU --ihp-gamma 2 --ihp-plow 0.5"
    "3|learnable_fuse-sum-shallow3-d8_loss-focal|54965|bcedice + $LAM_FOCAL * focal (focal alpha 0.75, gamma 2)|--hpa-mode learnable $CSF --extra-term focal --extra-weight $LAM_FOCAL --focal-alpha 0.75 --focal-gamma 2"
    "4|learnable_fuse-sum-shallow3-d8_loss-tversky|54965|bcedice + $LAM_TVERSKY * tversky (tversky alpha 0.3, beta 0.7)|--hpa-mode learnable $CSF --extra-term tversky --extra-weight $LAM_TVERSKY --tversky-alpha 0.3 --tversky-beta 0.7"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718_701020/isic2017 ;;
        isic18) DATA_PATH=./data/data_isic1718_701020/isic2018 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    [ -n "$DATA_PATH_OVERRIDE" ] && DATA_PATH=$DATA_PATH_OVERRIDE
    for s in train val test; do
        [ -d "$DATA_PATH/$s/images" ] || { echo "missing $DATA_PATH/$s/images (run scripts/make_split_701020.py)" >&2; exit 1; }
    done

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r idx suffix params loss flags <<<"$entry"
        want "$idx" || continue
        name="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        dir="$RESULTS_ROOT/$name"
        # resume guard: the loss recorded by an earlier attempt must be this run's loss
        if [ -f "$dir/log/train.info.log" ]; then
            logged=$(grep -m1 ' - loss: ' "$dir/log/train.info.log" | sed -e 's/.* - loss: //' -e 's/; checkpoint selection.*//' || true)
            if [ -n "$logged" ] && [ "$logged" != "$loss" ]; then
                echo "REFUSING to resume $name: its log says loss '$logged', this run needs '$loss'" >&2
                continue
            fi
        fi
        args=(--work-dir "$dir" --dataset "$DATASET" --data-path "$DATA_PATH" --test-split test
              --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
              --device "$DEVICE" --seed "$SEED" --save-every "$SAVE_EVERY")
        # shellcheck disable=SC2206
        args+=($flags)

        if [ -n "$PARALLEL" ]; then
            mkdir -p "$dir"
            echo "launching run $idx on $DATASET: $name (expect $params params, loss: $loss) -> $dir/stdout.log"
            setsid nohup python train.py "${args[@]}" > "$dir/stdout.log" 2>&1 < /dev/null &
            sleep 3
        else
            echo
            echo "#================ run $idx on $DATASET: $name (expect $params params, loss: $loss) ================#"
            python train.py "${args[@]}"
        fi
    done
done
