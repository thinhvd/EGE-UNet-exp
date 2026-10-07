#!/usr/bin/env bash
# EXP-17: LSA-CSF on the 80/20 protocol (train / test, NO validation split; scripts/make_split_8020.py), with the
# original loss and with E3a (EXP-6: + 0.095 * Kervadec boundary loss on the final output). Seed 123.
# ISIC17 + ISIC18, one box per dataset, 3 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp17_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp17_train.sh   # pre-flight
#
#  #  arm  name suffix                                              params  model                 loss
#  0  A0   learnable                                                53374   original              original (BCE + Dice, deep sup.)
#  1  A1   learnable_fuse-sum-shallow3-d8_lsa-route-probe           56510   LSA-CSF (EXP-15 E3)   original
#  2  A2   learnable_fuse-sum-shallow3-d8_lsa-route-probe_loss-bl   56510   LSA-CSF (EXP-15 E3)   original + 0.095 * bl (E3a)
#
# No validation split: --no-val skips the val loop and the best-checkpoint selection; the last-epoch weights are saved
# as checkpoints/final-epoch300.pth and tested on the test split (test_results.json). Every run keeps epoch200..300
# every 10 epochs for the SWA average (the headline model, scripts/exp17_analyze.sh). results/ MUST be empty before
# the real launch.
# CHECK AT THE START: "params:" matches the table (and "lsa: route aux=probe" for A1 / A2); one rotation angle
# 18.85089558633996 (seed 123); the "loss:" line; "test split: test (429 | 539 images); no validation split: testing the
# last-epoch weights" at the end.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-123}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
DEVICE=${DEVICE:-cuda}
PARALLEL=${PARALLEL:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
SAVE_EVERY=${SAVE_EVERY:-10}
SAVE_FROM=${SAVE_FROM:-200}
RESULTS_ROOT=${RESULTS_ROOT:-results}            # local smoke tests point this elsewhere
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset with train / test

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

CSF="--fusion sum --fusion-stages shallow3 --fusion-dim 8"
LSA="$CSF --lsa route --lsa-aux probe"
AUX_TXT='(0.5*[BD(D seg)+BD(.5,1)(D ring +-3px)] + 0.3*[BD(C seg)+focal interior depth>=0.15 /|H|])'
# index@name suffix (after egeunet_<ds>_)@expected params@expected loss (the "loss:" log line)@flags ('@': the loss text contains '|')
RUNS=(
    "0@learnable@53374@bcedice@--hpa-mode learnable"
    "1@learnable_fuse-sum-shallow3-d8_lsa-route-probe@56510@bcedice + expert aux probe $AUX_TXT@--hpa-mode learnable $LSA"
    "2@learnable_fuse-sum-shallow3-d8_lsa-route-probe_loss-bl@56510@bcedice + 0.095 * bl + expert aux probe $AUX_TXT@--hpa-mode learnable $LSA --extra-term bl --extra-weight 0.095"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718_8020/isic2017 ;;
        isic18) DATA_PATH=./data/data_isic1718_8020/isic2018 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    [ -n "$DATA_PATH_OVERRIDE" ] && DATA_PATH=$DATA_PATH_OVERRIDE
    for s in train test; do
        [ -d "$DATA_PATH/$s/images" ] || { echo "missing $DATA_PATH/$s/images (run scripts/make_split_8020.py)" >&2; exit 1; }
    done

    for entry in "${RUNS[@]}"; do
        IFS='@' read -r idx suffix params loss flags <<<"$entry"
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
        args=(--work-dir "$dir" --dataset "$DATASET" --data-path "$DATA_PATH" --test-split test --no-val
              --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
              --device "$DEVICE" --seed "$SEED" --save-every "$SAVE_EVERY" --save-from "$SAVE_FROM")
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
