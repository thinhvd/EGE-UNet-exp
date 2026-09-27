#!/usr/bin/env bash
# EXP-7 post-training evaluation. Run ON THE BOX after scripts/exp07_train.sh finishes (one dataset per
# box), or locally on the synced batch: RESULTS_DIR=results/EGE-UNet-results-exp7 DEVICE=cpu.
#
#   DATASETS=isic17 bash scripts/exp07_analyze.sh
#
# Per dataset: per-image metrics for every finished run, paired per-image DSC tests against the baseline
# of the batch (reference only), and the pooled-DSC summary (repeats of X1 averaged). The mechanism probe
# (analysis/e3a_mechanism.py probe, fixed pixel sets from EXP-6) runs locally after the pull.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
RESULTS_DIR=${RESULTS_DIR:-results}
THRESHOLDS=${THRESHOLDS:-0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}
# label|run name pattern ({ds})
RUNS=(
    "base|egeunet_{ds}_learnable_s${SEED}"
    "e3a|egeunet_{ds}_learnable_loss-bl_s${SEED}"
    "x1|egeunet_{ds}_learnable_loss-bl+fndp_s${SEED}"
    "x1rep|egeunet_{ds}_learnable_loss-bl+fndp_s${SEED}_rep1"
    "x1fn|egeunet_{ds}_learnable_loss-fndp_s${SEED}"
    "x2|egeunet_{ds}_learnable_loss-blhalf_s${SEED}"
)

for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017; N_VAL=650 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018; N_VAL=808 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    n_found=$( (ls "$DATA_PATH/val/images" 2>/dev/null || true) | wc -l)
    [ "$n_found" -eq "$N_VAL" ] || { echo "SKIP $DATASET: expected $N_VAL val images, found $n_found"; continue; }

    echo "#================ $DATASET ================#"
    present=()
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label pat <<<"$entry"
        dir=${pat//\{ds\}/$DATASET}
        if [ ! -f "$RESULTS_DIR/$dir/test_results.json" ]; then echo "  not finished: $dir"; continue; fi
        echo "  per-image metrics: $dir"
        python analysis/eval_per_image.py --checkpoint "$RESULTS_DIR/$dir" --data-path "$DATA_PATH" \
            --dataset "$DATASET" --device "$DEVICE" --thresholds "$THRESHOLDS" \
            --out-csv "$RESULTS_DIR/$dir/analysis/per_image_metrics_full.csv" | tail -2
        present+=("$label=$RESULTS_DIR/$dir")
    done
    if [ ${#present[@]} -ge 2 ] && [[ "${present[0]}" == base=* ]]; then
        args=(); for p in "${present[@]}"; do args+=(--run "$p"); done
        python analysis/significance_tests.py "${args[@]}" --csv-name per_image_metrics_full.csv \
            --metric dsc --out-dir "$RESULTS_DIR/exp07_${DATASET}/significance" | tail -3
    fi
    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-7 — kết quả theo DSC pooled ($DATASET)" \
        --variant 'Baseline=egeunet_{ds}_learnable_s{seed}=luật hiện tại (BCE + Dice)' \
        --variant 'E3a=egeunet_{ds}_learnable_loss-bl_s{seed}=+ phạt vệt lem theo khoảng cách (chạy lại)' \
        --variant 'X1=egeunet_{ds}_learnable_loss-bl+fndp_s{seed}|egeunet_{ds}_learnable_loss-bl+fndp_s{seed}_rep1=E3a + phạt bỏ sót theo khoảng cách tới vùng đã tô' \
        --variant 'X1-FN=egeunet_{ds}_learnable_loss-fndp_s{seed}=chỉ phạt bỏ sót (không E3a)' \
        --variant 'X2=egeunet_{ds}_learnable_loss-blhalf_s{seed}=E3a nửa cường độ' \
        --out "$RESULTS_DIR/exp07_summary_${DATASET}.md" > /dev/null
    echo "  summary: $RESULTS_DIR/exp07_summary_${DATASET}.md"
done
