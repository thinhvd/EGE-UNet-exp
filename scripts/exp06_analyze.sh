#!/usr/bin/env bash
# EXP-6 post-training evaluation. Run ON THE SERVER after scripts/exp06_train.sh finishes (also works
# locally on a synced batch folder: RESULTS_DIR=results/EGE-UNet-results-exp6 DEVICE=cpu).
#
#   bash scripts/exp06_analyze.sh
#   DATASETS=isic18 bash scripts/exp06_analyze.sh
#
# Per dataset: per-image metrics for every finished run (analysis/eval_per_image.py), paired per-image
# DSC tests of each loss variant against the baseline of the same batch (reference only), then one
# summary across datasets (analysis/exp06_summary.py) that decides on pooled DSC. Runs that are not
# finished yet are skipped, so the script can be re-run after batch B.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
RESULTS_DIR=${RESULTS_DIR:-results}
THRESHOLDS=${THRESHOLDS:-0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}
TAGS=(base area tvmatch tv region bl snbl)

run_dir() {   # dataset, tag
    if [ "$2" = "base" ]; then echo "egeunet_$1_learnable_s${SEED}"
    else echo "egeunet_$1_learnable_loss-$2_s${SEED}"; fi
}

analysed=()
for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017; N_VAL=650 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018; N_VAL=808 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    n_found=$( (ls "$DATA_PATH/val/images" 2>/dev/null || true) | wc -l)
    if [ "$n_found" -ne "$N_VAL" ]; then
        echo "SKIP $DATASET: expected $N_VAL images in $DATA_PATH/val/images, found $n_found"; continue
    fi

    echo "#================ $DATASET ================#"
    present=()
    for tag in "${TAGS[@]}"; do
        dir=$(run_dir "$DATASET" "$tag")
        if [ ! -f "$RESULTS_DIR/$dir/test_results.json" ]; then
            echo "  not finished: $dir"; continue
        fi
        echo "  per-image metrics: $dir"
        python analysis/eval_per_image.py \
            --checkpoint "$RESULTS_DIR/$dir" --data-path "$DATA_PATH" --dataset "$DATASET" \
            --device "$DEVICE" --thresholds "$THRESHOLDS" \
            --out-csv "$RESULTS_DIR/$dir/analysis/per_image_metrics_full.csv" | tail -2
        present+=("$tag=$RESULTS_DIR/$dir")
    done
    if [ ${#present[@]} -lt 2 ] || [[ "${present[0]}" != base=* ]]; then
        echo "SKIP tests on $DATASET: needs the baseline and at least one variant"; continue
    fi
    args=()
    for p in "${present[@]}"; do args+=(--run "$p"); done
    python analysis/significance_tests.py "${args[@]}" --csv-name per_image_metrics_full.csv \
        --metric dsc --out-dir "$RESULTS_DIR/exp06_${DATASET}/significance" | tail -3
    analysed+=("$DATASET")
done

python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASETS" --seed "$SEED" \
    --out "$RESULTS_DIR/exp06_summary.md"

cat <<'EON'

#---------- pull to the local machine ----------#
#   SERVER=root@<host> PORT=<port> LEAN=1 \
#     LOCAL_DIR=$(pwd)/results/EGE-UNet-results-exp6 bash scripts/sync_results.sh
EON
