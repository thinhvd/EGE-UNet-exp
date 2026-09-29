#!/usr/bin/env bash
# EXP-9 evaluation. ON THE BOX after scripts/exp09_train.sh (one dataset per box), or locally on the synced
# batch: RESULTS_DIR=results/EGE-UNet-results-exp9 DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp09_analyze.sh
# Per dataset:
#   1. per-image metrics of every finished run;
#   2. paired per-image DSC tests, one folder per comparison (the script writes a fixed file name):
#        significance_vs-A0     every arm against the original model + original loss (the decision)
#        significance_vs-A0e    the architectures against the original model + E3a (same loss)
#        significance_vs-A1e    boundary-guided vs plain fusion (both with E3a)
#        significance_A3e-vs-A4e  supervised boundary heads vs free gate
#        significance_A3e-vs-A3   E3a on the boundary-guided model
#   3. the pooled-DSC summary (A0 is the baseline);
#   4. the mechanism readouts (analysis/exp09_mechanism.py).
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
RUNS=(
    "A0|learnable"
    "A0e|learnable_loss-bl"
    "A1e|learnable_fuse-sum-shallow3-d8_loss-bl"
    "A3|learnable_fuse-bg_stage-shallow3-d8_loss-bnd"
    "A3e|learnable_fuse-bg_stage-shallow3-d8_loss-bl+bnd"
    "A4e|learnable_fuse-bg_stage-shallow3-d8_loss-bl"
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
    OUT="$RESULTS_DIR/exp09_${DATASET}"
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
    compare vs-A0 A0 A0e A1e A3 A3e A4e
    compare vs-A0e A0e A1e A3e A4e
    compare vs-A1e A1e A3e A4e
    compare A3e-vs-A4e A4e A3e
    compare A3e-vs-A3 A3 A3e

    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-9 — kết quả theo DSC pooled ($DATASET)" \
        --variant "A0 Gốc=egeunet_{ds}_learnable_s{seed}${RUN_SUFFIX}=model gốc + loss gốc (như paper)" \
        --variant "A0e=egeunet_{ds}_learnable_loss-bl_s{seed}${RUN_SUFFIX}=model gốc + loss gốc + 0,095 × bl (E3a)" \
        --variant "A1e=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_loss-bl_s{seed}${RUN_SUFFIX}=+ CSF thường dec3–5 (8 kênh) + E3a" \
        --variant "A3=egeunet_{ds}_learnable_fuse-bg_stage-shallow3-d8_loss-bnd_s{seed}${RUN_SUFFIX}=+ BG-CSF + loss dải viền, loss gốc" \
        --variant "A3e=egeunet_{ds}_learnable_fuse-bg_stage-shallow3-d8_loss-bl+bnd_s{seed}${RUN_SUFFIX}=+ BG-CSF + loss dải viền + E3a" \
        --variant "A4e=egeunet_{ds}_learnable_fuse-bg_stage-shallow3-d8_loss-bl_s{seed}${RUN_SUFFIX}=+ BG-CSF gate tự do + E3a" \
        --out "$RESULTS_DIR/exp09_summary_${DATASET}.md" > /dev/null
    echo "  summary: $RESULTS_DIR/exp09_summary_${DATASET}.md"

    mech=()
    for label in A0 A0e A1e A3 A3e A4e; do
        [ -n "${DIR[$label]:-}" ] && mech+=(--run "$label=${DIR[$label]}")
    done
    pairs=()
    for pv in A1e=A0e A3e=A0e A3e=A1e A4e=A1e A3e=A4e A3e=A3; do
        [ -n "${DIR[${pv%=*}]:-}" ] && [ -n "${DIR[${pv#*=}]:-}" ] && pairs+=(--pair "$pv")
    done
    if [ -n "${DIR[A0]:-}" ]; then
        echo "  mechanism readouts"
        python analysis/exp09_mechanism.py --dataset "$DATASET" --data-path "$DATA_PATH" --device "$MECH_DEVICE" \
            "${mech[@]}" "${pairs[@]}" --out-dir "$OUT/mechanism" | tail -3
    fi
    unset DIR
done
