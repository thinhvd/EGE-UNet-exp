#!/usr/bin/env bash
# EXP-6: loss variants on the ORIGINAL EGE-UNet (GHPA learnable, no fusion), ISIC17 + ISIC18, seed 42,
# seed-determined rotation (as EXP-5). Only the training loss changes between runs.
#
#   PARALLEL=1 BATCH=A bash scripts/exp06_train.sh             # batch A: runs 0-4 on both datasets (10 runs)
#   PARALLEL=1 BATCH=B bash scripts/exp06_train.sh             # batch B: runs 5-6 (4 runs, distance maps)
#   PARALLEL=1 bash scripts/exp06_train.sh                     # everything at once (14 runs)
#   PARALLEL=1 bash scripts/exp06_train.sh 0 1                 # only runs 0 and 1 (on every dataset)
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 bash scripts/exp06_train.sh   # pre-flight smoke test
#
# Each run writes to results/<name>/ and auto-resumes from checkpoints/latest.pth if interrupted.
# results/ MUST be empty before the real launch: the baseline run name is identical to EXP-5's.
#
#  #  tag      flags                                         what it is (exp_docs/06, Phần 1)
#  0  base     (none)                                        Baseline: the current loss, BCE + Dice
#  1  area     --extra-term area     --extra-weight 0.46     E4  + penalty on the relative area error
#  2  tvmatch  --extra-term tv_match --extra-weight 0.32     E1b + contour as long as the true contour
#  3  tv       --extra-term tv       --extra-weight 45       E1a + short contour (the original ACL length term)
#  4  region   --loss bce_region                             E2  Dice replaced by the ACL region term
#  5  bl       --extra-term bl       --extra-weight 0.095    E3a + spill penalty growing with distance (pixels)
#  6  snbl     --extra-term snbl     --extra-weight 0.18     E3b + the same, distance in lesion radii
#
# WEIGHTS were fixed before any run by analysis/calibrate_loss_weights.py on the EXP-5 baselines' TRAIN
# images (median gradient norm of the term = 0.25 x that of BceDice on the final output). ISIC17 and ISIC18
# gave weights within 1.4x of each other; one shared weight (their geometric mean, 2 digits) is used on both
# datasets so the "same loss" is really the same. Tables: results/EGE-UNet-results-exp6/loss_weights/.
#
# CHECK AT THE START OF EVERY RUN (stdout.log and log/train.info.log):
#   params: 53374 total                  every run, the model never changes
#   rotation angle: 230.19364744483815 (seed 42)   one distinct line across all runs
#   loss: <matches the table>; checkpoint selection: original GT_BceDiceLoss   (baseline: "same loss")
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}   # changing it changes the augmentation RNG stream
DEVICE=${DEVICE:-cuda}
PARALLEL=${PARALLEL:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
BATCH=${BATCH:-}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

# index|tag|batch|loss flags
RUNS=(
    "0|base|A|"
    "1|area|A|--extra-term area --extra-weight 0.46"
    "2|tvmatch|A|--extra-term tv_match --extra-weight 0.32"
    "3|tv|A|--extra-term tv --extra-weight 45"
    "4|region|A|--loss bce_region"
    "5|bl|B|--extra-term bl --extra-weight 0.095"
    "6|snbl|B|--extra-term snbl --extra-weight 0.18 --snbl-tau 3"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

run_name() {   # dataset, tag
    if [ "$2" = "base" ]; then echo "egeunet_$1_learnable_s${SEED}"
    else echo "egeunet_$1_learnable_loss-$2_s${SEED}"; fi
}

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    [ -d "$DATA_PATH/train/images" ] || { echo "missing $DATA_PATH/train/images" >&2; exit 1; }

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r idx tag batch lossflags <<<"$entry"
        want "$idx" || continue
        [ -n "$BATCH" ] && [ "$BATCH" != "$batch" ] && continue
        name=$(run_name "$DATASET" "$tag")${RUN_SUFFIX}
        args=(--work-dir "results/$name" --dataset "$DATASET" --data-path "$DATA_PATH"
              --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
              --device "$DEVICE" --seed "$SEED")
        # shellcheck disable=SC2206
        [ -n "$lossflags" ] && args+=($lossflags)

        if [ -n "$PARALLEL" ]; then
            mkdir -p "results/$name"
            echo "launching run $idx ($tag, batch $batch) on $DATASET: $name -> results/$name/stdout.log"
            setsid nohup python train.py "${args[@]}" > "results/$name/stdout.log" 2>&1 < /dev/null &
            sleep 3
        else
            echo
            echo "#================ run $idx ($tag) on $DATASET: $name ================#"
            python train.py "${args[@]}"
        fi
    done
done

if [ -n "$PARALLEL" ]; then
    cat <<'EOF'

Launched detached. Verify every run, then watch:
  grep -h "params:" results/*/stdout.log | sort | uniq -c           # 53374 on every run
  grep -h "rotation angle" results/*/stdout.log | sort | uniq -c    # ONE distinct line
  grep -h "^loss:" results/*/stdout.log | sort | uniq -c            # one line per loss variant
  for d in results/*/; do echo -n "$d "; tail -1 "$d/metrics.csv" 2>/dev/null; done

To run the analysis automatically when the last run finishes:
  setsid nohup bash -c 'while pgrep -f "[t]rain.py" >/dev/null; do sleep 120; done; bash scripts/exp06_analyze.sh' \
      > results/exp06_analyze.log 2>&1 < /dev/null &
EOF
fi
