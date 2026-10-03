#!/usr/bin/env bash
# EXP-15: lesion-state-aware selective cross-stage fusion (LSA-CSF, models/lsa_fusion.py), 70:10:20 protocol
# (train 70 % / val 10 % for checkpoint selection / test 20 % held out; scripts/make_split_701020.py).
# ISIC17 + ISIC18, seed 42, seed-determined rotation. One box per dataset, 6 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp15_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp15_train.sh   # pre-flight
#
#  #  arm  name suffix                                          params  fusion            gates (D, C, null)     expert aux heads
#  0  E0   learnable                                            53374   none              -                      -
#  1  E1   learnable_fuse-sum-shallow3-d8                       54965   plain CSF         -                      -
#  2  E2   learnable_fuse-sum-shallow3-d8_lsa-fixed-probe       56510   two experts       fixed (0.5, 0.5, 0)    probe (detached)
#  3  E3   learnable_fuse-sum-shallow3-d8_lsa-route-probe       56510   two experts       learned router         probe (detached)
#  4  E4   learnable_fuse-sum-shallow3-d8_lsa-route-supervise   56510   two experts       learned router         supervise
#  5  E5p  learnable_fuse-sum-shallow3-d8_lsa-fixed-supervise   56510   two experts       fixed (0.5, 0.5, 0)    supervise
#
# E5p separates the effect of the router (E4 vs E5p) from that of the expert supervision (E4 vs E3). The aux weights are
# fixed in advance (0.5 detail, 0.3 completion, tau 0.15; doc 15); no sweep. Checkpoint selection: the original loss on
# the val split. The final test (test_results.json) runs on the test split (--test-split test).
# Every run keeps epoch200..300 every 10 epochs (--save-every 10 --save-from 200) for the SWA average and the
# epoch-swing readout. results/ MUST be empty before the real launch.
# CHECK AT THE START: "params:" matches the table and the "lsa:" part of that line (fixed / route, probe / supervise);
# one rotation angle 230.19364744483815; the "loss:" line; "test split: test (434 | 538 images); checkpoint selection on
# val (216 | 270 images)" at the end.
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
SAVE_FROM=${SAVE_FROM:-200}
RESULTS_ROOT=${RESULTS_ROOT:-results}            # local smoke tests point this elsewhere
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset with train / val / test

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

CSF="--fusion sum --fusion-stages shallow3 --fusion-dim 8"
AUX_TXT='(0.5*[BD(D seg)+BD(.5,1)(D ring +-3px)] + 0.3*[BD(C seg)+focal interior depth>=0.15 /|H|])'
# index@name suffix (after egeunet_<ds>_)@expected params@expected loss (the "loss:" log line)@flags ('@': the loss text contains '|')
RUNS=(
    "0@learnable@53374@bcedice@--hpa-mode learnable"
    "1@learnable_fuse-sum-shallow3-d8@54965@bcedice@--hpa-mode learnable $CSF"
    "2@learnable_fuse-sum-shallow3-d8_lsa-fixed-probe@56510@bcedice + expert aux probe $AUX_TXT@--hpa-mode learnable $CSF --lsa fixed --lsa-aux probe"
    "3@learnable_fuse-sum-shallow3-d8_lsa-route-probe@56510@bcedice + expert aux probe $AUX_TXT@--hpa-mode learnable $CSF --lsa route --lsa-aux probe"
    "4@learnable_fuse-sum-shallow3-d8_lsa-route-supervise@56510@bcedice + expert aux supervise $AUX_TXT@--hpa-mode learnable $CSF --lsa route --lsa-aux supervise"
    "5@learnable_fuse-sum-shallow3-d8_lsa-fixed-supervise@56510@bcedice + expert aux supervise $AUX_TXT@--hpa-mode learnable $CSF --lsa fixed --lsa-aux supervise"
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
        IFS='@' read -r idx suffix params loss flags <<<"$entry"
        want "$idx" || continue
        name="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        dir="$RESULTS_ROOT/$name"
        # resume guard: the loss recorded by an earlier attempt must be this run's loss (the router / aux mode of an
        # LSA checkpoint is guarded again when train.py resumes: models/lsa_fusion.py refuses a mismatching checkpoint)
        if [ -f "$dir/log/train.info.log" ]; then
            logged=$(grep -m1 ' - loss: ' "$dir/log/train.info.log" | sed -e 's/.* - loss: //' -e 's/; checkpoint selection.*//' || true)
            if [ -n "$logged" ] && [ "$logged" != "$loss" ]; then
                echo "REFUSING to resume $name: its log says loss '$logged', this run needs '$loss'" >&2
                continue
            fi
        fi
        args=(--work-dir "$dir" --dataset "$DATASET" --data-path "$DATA_PATH" --test-split test
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
