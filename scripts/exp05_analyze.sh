#!/usr/bin/env bash
# EXP-5 post-training evaluation. Run this ON THE SERVER after scripts/exp05_train.sh finishes
# (it also works locally on a synced batch folder: RESULTS_DIR=results/EGE-UNet-results-exp5 DEVICE=cpu).
#
#   bash scripts/exp05_analyze.sh
#   DATASETS=isic18 bash scripts/exp05_analyze.sh
#
# Per dataset: per-image metrics for the 3 runs (650 / 808 forward passes each, on the GPU), then
# paired per-image tests variant-vs-baseline (analysis/significance_tests.py: paired t-test as in
# notebooks/significant_test.ipynb, plus Wilcoxon, Cohen d_z and Holm over the 2 variants) on the
# whole set and on the small / large lesion tertiles, for DSC and IoU; then the stratified figure
# (analysis/compare_runs.py). Finally one summary (analysis/exp05_summary.py) across both datasets.
# The two datasets are never pooled: pairing is per image, so every test is within one dataset.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
RESULTS_DIR=${RESULTS_DIR:-results}
THRESHOLDS=${THRESHOLDS:-0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}

if [ "$(ps -eo args | grep -c '[t]rain.py')" -gt 0 ]; then
    echo "note: training processes are still running; evaluation is deterministic, only slower."
fi

analysed=()
for DATASET in $DATASETS; do
    # --dataset picks the normalization statistics and --data-path the images; a mismatch between
    # them is silent and produces plausible-looking numbers, so derive the path and check the count.
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017; N_VAL=650 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018; N_VAL=808 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    n_found=$( (ls "$DATA_PATH/val/images" 2>/dev/null || true) | wc -l)   # never abort under pipefail
    if [ "$n_found" -ne "$N_VAL" ]; then
        echo "SKIP $DATASET: expected $N_VAL images in $DATA_PATH/val/images, found $n_found"
        continue
    fi

    RUNS=(
        "learnable|egeunet_${DATASET}_learnable_s${SEED}"
        "none+sum-deep3|egeunet_${DATASET}_none_fuse-sum-deep3_s${SEED}"
        "none+sum_attn-all5|egeunet_${DATASET}_none_fuse-sum_attn-all5_s${SEED}"
    )
    OUT="$RESULTS_DIR/exp05_${DATASET}"

    echo "#================ $DATASET ================#"
    echo "#---------- per-image metrics ----------#"
    present=()
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label dir <<<"$entry"
        if [ ! -f "$RESULTS_DIR/$dir/test_results.json" ]; then
            echo "  MISSING $label - $RESULTS_DIR/$dir has no test_results.json (run unfinished?)"
            continue
        fi
        present+=("$label=$RESULTS_DIR/$dir")
        echo "  $label"
        python analysis/eval_per_image.py \
            --checkpoint "$RESULTS_DIR/$dir" --data-path "$DATA_PATH" --dataset "$DATASET" \
            --device "$DEVICE" --thresholds "$THRESHOLDS" \
            --out-csv "$RESULTS_DIR/$dir/analysis/per_image_metrics_full.csv" \
            | tail -4
    done

    if [ ${#present[@]} -ne ${#RUNS[@]} ]; then
        echo "SKIP $DATASET: needs all ${#RUNS[@]} runs finished (found ${#present[@]})"
        continue
    fi
    args=()
    for p in "${present[@]}"; do args+=(--run "$p"); done

    for metric in dsc iou; do
        echo "#---------- paired tests vs learnable ($metric) ----------#"
        python analysis/significance_tests.py "${args[@]}" \
            --csv-name per_image_metrics_full.csv --metric "$metric" \
            --out-dir "$OUT/significance"
    done

    echo "#---------- stratified comparison figure (dsc) ----------#"
    python analysis/compare_runs.py "${args[@]}" \
        --csv-name per_image_metrics_full.csv --metric dsc \
        --out-dir "$OUT/compare_dsc" | tail -12

    analysed+=("$DATASET")
done

if [ ${#analysed[@]} -eq 0 ]; then
    echo "nothing analysed"; exit 1
fi

echo
echo "#---------- summary across datasets ----------#"
python analysis/exp05_summary.py --results-dir "$RESULTS_DIR" --datasets "${analysed[*]}" \
    --seed "$SEED" --out "$RESULTS_DIR/exp05_summary.md"

cat <<'EOF'

#---------- pull to the local machine ----------#
# From LOCAL, into the batch's own folder (the ISIC17 run names collide with EXP-4's):
#   SERVER=root@<host> PORT=<port> LEAN=1 \
#     LOCAL_DIR=$(pwd)/results/EGE-UNet-results-exp5 bash scripts/sync_results.sh
EOF
