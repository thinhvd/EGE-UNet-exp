#!/usr/bin/env bash
# EXP-11: boundary-constrained region completion (BCRC) at dec5 on top of the plain cross-stage fusion.
# ISIC17 + ISIC18, seed 42, seed-determined rotation. One box per dataset, 5 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp11_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp11_train.sh   # pre-flight
#
#  #  arm  name suffix                                                 params  model                                            loss
#  0  D0   learnable                                                   53374   original EGE-UNet                                original (BCE + Dice)
#  1  D1   learnable_fuse-sum-shallow3-d8                              54965   + sum fusion into dec3..5, width 8               original
#  2  D2   learnable_fuse-sum-shallow3-d8_brc_loss-ring3               55656   + region completion: core grows over A x (1 - B) original + 0.3 * ring loss on the barrier head
#  3  D3   learnable_fuse-sum-shallow3-d8_brc-nobarrier_loss-ring3     55656   + the same growth over A only (B trained, unused) same as D2
#  4  D4   learnable_fuse-sum-shallow3-d8_brc_loss-ring3+hp            55656   = D2                                             D2 + 1.2 * hard-positive (deep lesion px)
#
# The decision compares every arm with run 0. Run 1 is the fusion-only control (and a repeat of EXP-10 B1);
# run 3 isolates the barrier, run 4 the hard-positive term. Growth: T = 32 steps at the 128 grid, seed =
# base probability >= 0.9 (step 0: T = 32 puts 98.9 % / 94.6 % of the train interior miss within reach).
# hp weight 1.2: pre-registered push balance k = 1 on the EXP-10 B1 checkpoints' train images (1.211 / 1.219,
# geometric mean rounded to 2 significant digits).
# Every run keeps epoch200..300 every 10 epochs (--save-every 10) for checkpoint-choice checks.
# results/ MUST be empty before the real launch.
# CHECK AT THE START: "params:" matches the table; one rotation angle 230.19364744483815; the "loss:" line;
# "complete: full T=32 seed=0.9" (D2, D4) / "complete: nobarrier T=32 seed=0.9" (D3).
# A work dir whose log records a different loss than this table is never resumed; D2 and D3 share the loss,
# their checkpoints refuse to load into the other mode (provenance buffers in models/completion.py).
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

CSF="--fusion sum --fusion-stages shallow3 --fusion-dim 8"
BRC="--complete-steps 32 --complete-seed 0.9 --ring-radius 3 --boundary-weight 0.3"
RING="bcedice + 0.3 * ring (BceDice(0.5,1) on the +-3px contour ring, barrier head)"
HP="bcedice + 1.2 * hp (hp r=10px, gamma 2) + 0.3 * ring (BceDice(0.5,1) on the +-3px contour ring, barrier head)"
# index|name suffix (after egeunet_<ds>_)|expected params|expected loss (the "loss:" log line)|flags
RUNS=(
    "0|learnable|53374|bcedice|--hpa-mode learnable"
    "1|learnable_fuse-sum-shallow3-d8|54965|bcedice|--hpa-mode learnable $CSF"
    "2|learnable_fuse-sum-shallow3-d8_brc_loss-ring3|55656|$RING|--hpa-mode learnable $CSF --complete full $BRC"
    "3|learnable_fuse-sum-shallow3-d8_brc-nobarrier_loss-ring3|55656|$RING|--hpa-mode learnable $CSF --complete nobarrier $BRC"
    "4|learnable_fuse-sum-shallow3-d8_brc_loss-ring3+hp|55656|$HP|--hpa-mode learnable $CSF --complete full $BRC --extra-term hp --extra-weight 1.2 --hp-radius 10 --hp-gamma 2"
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
