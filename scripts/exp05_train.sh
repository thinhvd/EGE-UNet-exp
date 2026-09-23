#!/usr/bin/env bash
# EXP-5: three configurations x two datasets (ISIC17 + ISIC18), seed 42, seed-determined rotation.
#
#   PARALLEL=1 bash scripts/exp05_train.sh                    # all 6 runs at once, detached
#   PARALLEL=1 DATASETS=isic17 bash scripts/exp05_train.sh    # only the ISIC17 half
#   PARALLEL=1 bash scripts/exp05_train.sh 0 2                # only runs 0 and 2 (on every dataset)
#   EPOCHS=1 RUN_SUFFIX=_smoke PARALLEL=1 bash scripts/exp05_train.sh   # pre-flight smoke test
#   bash scripts/exp05_train.sh                               # sequential, foreground
#
# Each run writes to results/<name>/ and auto-resumes from checkpoints/latest.pth if interrupted.
# results/ MUST be empty before the real launch: the ISIC17 run names are identical to EXP-4's, and a
# stale latest.pth would be resumed silently.
#
#  #  name                  params   what it is
#  0  learnable              53374   the original EGE-UNet (GHPA on)
#  1  none+sum-deep3         49059   GHPA off, weighted-sum cross-stage fusion into dec1..dec3
#  2  none+sum_attn-all5     51109   GHPA off, sum + cross-stage attention into dec1..dec5
#
# WHAT CHANGED SINCE EXP-4 (this branch only, not main). The rotation augmentation's single angle is
# now a function of --seed (utils.myRandomRotation(seed=...)): every run with seed 42 rotates by
# 230.19364744483815 degrees. Model init and the per-sample flip/rotate coins are untouched. What is
# left of the run-to-run noise is CUDA backward nondeterminism, which cannot be removed for this
# architecture (bilinear interpolate has no deterministic backward) - so single-run gaps are still
# single-run gaps, just with one nuisance source less.
#
# CHECK AT THE START OF EVERY RUN (both lines are in stdout.log and log/train.info.log):
#   params: <count from the table>       "53374" on run 1/2 means the flags did not reach the model
#   rotation angle: 230.19364744483815 (seed 42)   must be identical across all six runs
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

# Six single-process runs on one box: cap the CPU threads per run or they fight over cores and the
# GPU sits idle (EXP-4 measured ~100x slowdown without this).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}

# index|hpa_mode|fusion mode|stages|expected params
RUNS=(
    "0|learnable|none|-|53374"
    "1|none|sum|deep3|49059"
    "2|none|sum_attn|all5|51109"
)

WANTED=("$@")
want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

run_name() {   # dataset, hpa_mode, fusion mode, stages
    if [ "$3" = "none" ]; then echo "egeunet_$1_$2_s${SEED}"
    else echo "egeunet_$1_$2_fuse-$3-$4_s${SEED}"; fi
}

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    [ -d "$DATA_PATH/train/images" ] || { echo "missing $DATA_PATH/train/images" >&2; exit 1; }

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r idx hpa mode stages params <<<"$entry"
        want "$idx" || continue
        name=$(run_name "$DATASET" "$hpa" "$mode" "$stages")${RUN_SUFFIX}
        args=(--work-dir "results/$name" --dataset "$DATASET" --data-path "$DATA_PATH"
              --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
              --device "$DEVICE" --seed "$SEED" --hpa-mode "$hpa")
        [ "$mode" != "none" ] && args+=(--fusion "$mode" --fusion-stages "$stages")

        if [ -n "$PARALLEL" ]; then
            mkdir -p "results/$name"
            echo "launching run $idx on $DATASET: $name (expect $params params) -> results/$name/stdout.log"
            setsid nohup python train.py "${args[@]}" > "results/$name/stdout.log" 2>&1 < /dev/null &
            sleep 3      # stagger so the runs do not all hit the dataset at the same instant
        else
            echo
            echo "#================ run $idx on $DATASET: $name (expect $params params) ================#"
            python train.py "${args[@]}"
        fi
    done
done

if [ -n "$PARALLEL" ]; then
    cat <<'EOF'

Launched detached. Verify the flags reached every run, then watch:
  grep -h "params:" results/*/stdout.log                       # 53374 / 49059 / 51109, each once per dataset
  grep -h "rotation angle" results/*/stdout.log | sort | uniq -c   # ONE distinct line, count = number of runs
  tail -n 2 results/*/stdout.log
  for d in results/*/; do echo -n "$d "; tail -1 "$d/metrics.csv" 2>/dev/null; done

To run the analysis automatically when the last run finishes:
  setsid nohup bash -c 'while pgrep -f "[t]rain.py" >/dev/null; do sleep 120; done; bash scripts/exp05_analyze.sh' \
      > results/exp05_analyze.log 2>&1 < /dev/null &
EOF
else
    cat <<'EOF'

#---------- after training ----------#
bash scripts/exp05_analyze.sh          # per-image metrics, paired tests, summary tables (on the server)
# then pull to the local machine into its own batch folder (best checkpoints only):
#   SERVER=root@host PORT=... LEAN=1 LOCAL_DIR=$(pwd)/results/EGE-UNet-results-exp5 bash scripts/sync_results.sh
EOF
fi
