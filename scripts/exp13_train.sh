#!/usr/bin/env bash
# EXP-13: boundary-constrained core-to-region propagation (CRP) at dec5 on top of the plain cross-stage fusion.
# ISIC17 + ISIC18, seed 42, seed-determined rotation. One box per dataset, 5 runs each:
#
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp13_train.sh
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 DATASETS=isic17 bash scripts/exp13_train.sh   # pre-flight
#
#  #  arm  name suffix                                                   params  model                                         loss
#  0  D0   learnable                                                     53374   original EGE-UNet                             original (BCE + Dice)
#  1  D1   learnable_fuse-sum-shallow3-d8                                54965   + sum fusion into dec3..5, width 8            original
#  2  E1   learnable_fuse-sum-shallow3-d8_crp_loss-aff                   55634   + propagation along edges S, relative cap,    original + edge-affinity loss
#                                                                                edge head trained by the affinity loss only
#  3  E2   learnable_fuse-sum-shallow3-d8_crp-coupled_loss-aff+comp      55634   E1 + segmentation / completion gradients      + completion loss
#                                                                                into the edge head through the propagation
#  4  E3   learnable_fuse-sum-shallow3-d8_crp-nocap_loss-aff             55634   E1 without the relative cap (W = S^beta)       as E1
#
# The decision compares every arm with run 0. Run 1 is the fusion-only control (4th CSF draw); E2 vs E1 isolates
# gradients through the propagation (EXP-11's training), E3 vs E1 the relative cap.
# Cap / edge parameters tau, kappa, beta and the completion weight: fixed by analysis/exp13_step0.py on TRAIN images
# (results/exp13_step0/, doc 13 section 2).
# Every run keeps epoch200..300 every 10 epochs (--save-every 10) for checkpoint-choice checks.
# results/ MUST be empty before the real launch.
# CHECK AT THE START: "params:" matches the table; one rotation angle 230.19364744483815; the "loss:" line;
# "propagate: pair | pair_coupled | pair_nocap T=32 seed=0.9 tau=... kappa=... beta=...".
# A work dir whose log records a different loss is never resumed; E1 and E3 share the loss, their checkpoints
# refuse to load into the other mode (provenance buffers in models/propagation.py).
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

# fixed by step 0 (train images only)
TAU=0.3
KAPPA=5
BETA=0.5
LCOMP=1

CSF="--fusion sum --fusion-stages shallow3 --fusion-dim 8"
PROP="--propagate-steps 32 --propagate-seed 0.9 --propagate-tau $TAU --propagate-kappa $KAPPA --propagate-beta $BETA --aff-weight 1 --hard-frac 0.2"
AFF="bcedice + 1 * aff (edge BCE same-label: 0.5 cut / 0.25 LL+BB / 0.25 hard 20% contrast)"
COMP="$AFF + $LCOMP * comp (BCE of propagated R vs GT, GT dilated 10 cells)"
# index|name suffix (after egeunet_<ds>_)|expected params|expected loss (the "loss:" log line)|flags
RUNS=(
    "0|learnable|53374|bcedice|--hpa-mode learnable"
    "1|learnable_fuse-sum-shallow3-d8|54965|bcedice|--hpa-mode learnable $CSF"
    "2|learnable_fuse-sum-shallow3-d8_crp_loss-aff|55634|$AFF|--hpa-mode learnable $CSF --propagate pair $PROP"
    "3|learnable_fuse-sum-shallow3-d8_crp-coupled_loss-aff+comp|55634|$COMP|--hpa-mode learnable $CSF --propagate pair_coupled $PROP --comp-weight $LCOMP"
    "4|learnable_fuse-sum-shallow3-d8_crp-nocap_loss-aff|55634|$AFF|--hpa-mode learnable $CSF --propagate pair_nocap $PROP"
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
