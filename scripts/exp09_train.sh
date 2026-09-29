#!/usr/bin/env bash
# EXP-9: boundary-guided cross-stage fusion (BG-CSF) with E3a, next to the author's original model and loss.
# ISIC17 + ISIC18, seed 42, seed-determined rotation. One box per dataset, 6 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp09_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp09_train.sh   # pre-flight
#
#  #  arm  name suffix                                         params  model                              loss
#  0  A0   learnable                                           53374   original EGE-UNet                  original (BCE + Dice)
#  1  A0e  learnable_loss-bl                                   53374   original                           original + 0.095 * bl (E3a)
#  2  A1e  learnable_fuse-sum-shallow3-d8_loss-bl              54965   + sum fusion into dec3..5, width 8 original + E3a
#  3  A3   learnable_fuse-bg_stage-shallow3-d8_loss-bnd        55031   + boundary-guided fusion (bg_stage) original + contour-band loss
#  4  A3e  learnable_fuse-bg_stage-shallow3-d8_loss-bl+bnd     55031   same                               original + E3a + contour-band loss
#  5  A4e  learnable_fuse-bg_stage-shallow3-d8_loss-bl         55031   same (free gate)                   original + E3a (band loss logged only)
#
# The decision compares every arm with run 0. Run 1 isolates E3a; run 3 isolates the architecture with the
# original loss; run 5 is the same model as run 4 whose boundary heads learn only through the gate.
# Every run keeps epoch200..300 every 10 epochs (--save-every 10) for checkpoint-choice checks.
# results/ MUST be empty before the real launch.
# CHECK AT THE START: "params:" matches the table; one rotation angle 230.19364744483815; the "loss:" line.
# A work dir whose log records a different loss than this table is never resumed (the model weights alone
# cannot tell run 4 from run 5).
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
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

E3A="--extra-term bl --extra-weight 0.095"
BG="--fusion bg_stage --fusion-stages shallow3 --fusion-dim 8"
BND="1 * bnd (0.1*dec3 + 0.2*dec4 + 0.3*dec5, BceDice(0.5,1) on the contour band per head grid)"
# index|name suffix (after egeunet_<ds>_)|expected params|expected loss (the "loss:" log line)|flags
RUNS=(
    "0|learnable|53374|bcedice|--hpa-mode learnable"
    "1|learnable_loss-bl|53374|bcedice + 0.095 * bl|--hpa-mode learnable $E3A"
    "2|learnable_fuse-sum-shallow3-d8_loss-bl|54965|bcedice + 0.095 * bl|--hpa-mode learnable --fusion sum --fusion-stages shallow3 --fusion-dim 8 $E3A"
    "3|learnable_fuse-bg_stage-shallow3-d8_loss-bnd|55031|bcedice + $BND|--hpa-mode learnable $BG --boundary-weight 1"
    "4|learnable_fuse-bg_stage-shallow3-d8_loss-bl+bnd|55031|bcedice + 0.095 * bl + $BND|--hpa-mode learnable $BG $E3A --boundary-weight 1"
    "5|learnable_fuse-bg_stage-shallow3-d8_loss-bl|55031|bcedice + 0.095 * bl (+ bnd contour-band loss logged only, weight 0)|--hpa-mode learnable $BG $E3A --boundary-weight 0"
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
    [ -n "$DATA_PATH_OVERRIDE" ] && DATA_PATH=$DATA_PATH_OVERRIDE
    [ -d "$DATA_PATH/train/images" ] || { echo "missing $DATA_PATH/train/images" >&2; exit 1; }

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
        args=(--work-dir "$dir" --dataset "$DATASET" --data-path "$DATA_PATH"
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
