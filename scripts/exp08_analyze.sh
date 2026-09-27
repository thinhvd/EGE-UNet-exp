#!/usr/bin/env bash
# EXP-8 evaluation. ON THE BOX after scripts/exp08_train.sh (one dataset per box), or locally on the synced
# batch: RESULTS_DIR=results/EGE-UNet-results-exp8 DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp08_analyze.sh
# Per dataset: per-image metrics of every finished run, paired per-image DSC tests against the original
# model with the original loss (reference only), and the pooled-DSC summary.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
RESULTS_DIR=${RESULTS_DIR:-results}
THRESHOLDS=${THRESHOLDS:-0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}
RUNS=(
    "base|learnable"
    "sumdeep3|none_fuse-sum-deep3"
    "sumdeep3_x2|none_fuse-sum-deep3_loss-blhalf"
    "sumattn|none_fuse-sum_attn-all5"
    "sumattn_x2|none_fuse-sum_attn-all5_loss-blhalf"
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
        IFS='|' read -r label suffix <<<"$entry"
        dir="egeunet_${DATASET}_${suffix}_s${SEED}"
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
            --metric dsc --out-dir "$RESULTS_DIR/exp08_${DATASET}/significance" | tail -3
    fi
    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-8 — kết quả theo DSC pooled ($DATASET)" \
        --variant 'Gốc=egeunet_{ds}_learnable_s{seed}=model gốc + loss gốc (như paper)' \
        --variant 'sum-deep3=egeunet_{ds}_none_fuse-sum-deep3_s{seed}=none+sum-deep3 + loss gốc' \
        --variant 'sum-deep3+X2=egeunet_{ds}_none_fuse-sum-deep3_loss-blhalf_s{seed}=none+sum-deep3 + loss gốc + 0,0475 × bl' \
        --variant 'sum_attn-all5=egeunet_{ds}_none_fuse-sum_attn-all5_s{seed}=none+sum_attn-all5 + loss gốc' \
        --variant 'sum_attn-all5+X2=egeunet_{ds}_none_fuse-sum_attn-all5_loss-blhalf_s{seed}=none+sum_attn-all5 + loss gốc + 0,0475 × bl' \
        --out "$RESULTS_DIR/exp08_summary_${DATASET}.md" > /dev/null
    echo "  summary: $RESULTS_DIR/exp08_summary_${DATASET}.md"
done
