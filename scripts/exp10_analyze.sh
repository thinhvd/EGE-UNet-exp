#!/usr/bin/env bash
# EXP-10 evaluation. ON THE BOX after scripts/exp10_train.sh (one dataset per box), or locally on the synced
# batch: RESULTS_DIR=results/EGE-UNet-results-exp10 DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp10_analyze.sh
# Per dataset:
#   1. per-image metrics of every finished run;
#   2. paired per-image DSC tests, one folder per comparison:
#        significance_vs-B0    every arm against the original model (the decision)
#        significance_vs-B1    the refinement arms against the fusion-only model
#        significance_B2-vs-B3 gated + supervised residual vs the plain residual
#   3. the pooled-DSC summary (B0 is the baseline);
#   4. the mechanism readouts (analysis/exp10_mechanism.py): same-checkpoint knockouts of the residual and
#      of the gate, boundary-map quality, leverage, errors by distance to the contour.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
MECH_DEVICE=${MECH_DEVICE:-$DEVICE}
RESULTS_DIR=${RESULTS_DIR:-results}
THRESHOLDS=${THRESHOLDS:-0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset (skips the image count check)
RUN_SUFFIX=${RUN_SUFFIX:-}
RADIUS=${RADIUS:-10}
RUNS=(
    "B0|learnable"
    "B1|learnable_fuse-sum-shallow3-d8"
    "B2|learnable_fuse-sum-shallow3-d8_brr-gate_loss-zone10"
    "B3|learnable_fuse-sum-shallow3-d8_brr-plain"
)
for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017; N_VAL=650 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018; N_VAL=808 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    if [ -n "$DATA_PATH_OVERRIDE" ]; then
        DATA_PATH=$DATA_PATH_OVERRIDE
    else
        n_found=$( (ls "$DATA_PATH/val/images" 2>/dev/null || true) | wc -l)
        [ "$n_found" -eq "$N_VAL" ] || { echo "SKIP $DATASET: expected $N_VAL val images, found $n_found"; continue; }
    fi
    echo "#================ $DATASET ================#"
    OUT="$RESULTS_DIR/exp10_${DATASET}"
    declare -A DIR=()
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label suffix <<<"$entry"
        dir="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        if [ ! -f "$RESULTS_DIR/$dir/test_results.json" ]; then echo "  not finished: $dir"; continue; fi
        echo "  per-image metrics: $dir"
        python analysis/eval_per_image.py --checkpoint "$RESULTS_DIR/$dir" --data-path "$DATA_PATH" \
            --dataset "$DATASET" --device "$DEVICE" --thresholds "$THRESHOLDS" \
            --out-csv "$RESULTS_DIR/$dir/analysis/per_image_metrics_full.csv" | tail -2
        DIR[$label]="$RESULTS_DIR/$dir"
    done

    # compare <folder name> <reference label> <variant label>...: skipped unless the reference and one variant exist
    compare() {
        local name=$1 ref=$2; shift 2
        [ -n "${DIR[$ref]:-}" ] || { echo "  skip $name: $ref missing"; return 0; }
        local args=(--run "$ref=${DIR[$ref]}") v n=0
        for v in "$@"; do
            [ -n "${DIR[$v]:-}" ] && { args+=(--run "$v=${DIR[$v]}"); n=$((n + 1)); }
        done
        [ "$n" -gt 0 ] || { echo "  skip $name: no variant present"; return 0; }
        echo "  significance: $name"
        python analysis/significance_tests.py "${args[@]}" --csv-name per_image_metrics_full.csv \
            --metric dsc --out-dir "$OUT/significance_$name" | tail -2
    }
    compare vs-B0 B0 B1 B2 B3
    compare vs-B1 B1 B2 B3
    compare B2-vs-B3 B3 B2

    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-10 — kết quả theo DSC pooled ($DATASET)" \
        --variant "B0 Gốc=egeunet_{ds}_learnable_s{seed}${RUN_SUFFIX}=model gốc + loss gốc (như paper)" \
        --variant "B1=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s{seed}${RUN_SUFFIX}=+ CSF thường dec3–5 (8 kênh), loss gốc" \
        --variant "B2=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_brr-gate_loss-zone10_s{seed}${RUN_SUFFIX}=+ BRR: sửa logit dec5 có gate B5, loss vùng ±10 px" \
        --variant "B3=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_brr-plain_s{seed}${RUN_SUFFIX}=+ cùng head sửa logit, không gate, không B5" \
        --out "$RESULTS_DIR/exp10_summary_${DATASET}.md" > /dev/null
    echo "  summary: $RESULTS_DIR/exp10_summary_${DATASET}.md"

    mech=()
    for label in B0 B1 B2 B3; do
        [ -n "${DIR[$label]:-}" ] && mech+=(--run "$label=${DIR[$label]}")
    done
    pairs=()
    for pv in B1=B0 B2=B1 B3=B1 B2=B3; do
        [ -n "${DIR[${pv%=*}]:-}" ] && [ -n "${DIR[${pv#*=}]:-}" ] && pairs+=(--pair "$pv")
    done
    if [ -n "${DIR[B0]:-}" ]; then
        echo "  mechanism readouts"
        python analysis/exp10_mechanism.py --dataset "$DATASET" --data-path "$DATA_PATH" --device "$MECH_DEVICE" \
            --radius "$RADIUS" "${mech[@]}" "${pairs[@]}" --out-dir "$OUT/mechanism" | tail -3
    fi
    unset DIR
done
