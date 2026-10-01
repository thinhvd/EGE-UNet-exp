#!/usr/bin/env bash
# EXP-13 evaluation. ON THE BOX after scripts/exp13_train.sh (one dataset per box), or locally on the synced
# batch: RESULTS_DIR=results/EGE-UNet-results-exp13 DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp13_analyze.sh
# Per dataset:
#   1. per-image metrics of every finished run (threshold sweep 0.1-0.9);
#   2. paired per-image DSC tests, one folder per comparison:
#        significance_vs-D0     every arm against the original model (the decision)
#        significance_vs-D1     the propagation arms against the fusion-only model
#        significance_E2-vs-E1  gradients through the propagation vs edge head trained by its own loss only
#        significance_E3-vs-E1  no relative cap vs cap
#   3. the pooled-DSC summary (D0 is the baseline);
#   4. errors by position (analysis/exp11_errors.py): val with the heavy-image set of EXP-10
#      (analysis/lists/exp11_heavy_*.csv) and without the suspect-GT image, E1 / E2 / E3 against D1's
#      threshold curve; the same on the train split (test transform);
#   5. the mechanism readouts (analysis/exp13_mechanism.py): edge statistics, growth, knockouts, polygon split.
# Steps 4-5 print WARN and go on if they fail, so a readout bug never blocks the box from finishing.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
MECH_DEVICE=${MECH_DEVICE:-$DEVICE}
RESULTS_DIR=${RESULTS_DIR:-results}
THRESHOLDS=${THRESHOLDS:-0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9}
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset (skips the image count check and the heavy set)
RUN_SUFFIX=${RUN_SUFFIX:-}
EXP11_DIR=${EXP11_DIR:-results/EGE-UNet-results-exp11}   # locally: EXP-11 D1 joins the error readout as a noise ruler for D1
RUNS=(
    "D0|learnable"
    "D1|learnable_fuse-sum-shallow3-d8"
    "E1|learnable_fuse-sum-shallow3-d8_crp_loss-aff"
    "E2|learnable_fuse-sum-shallow3-d8_crp-coupled_loss-aff+comp"
    "E3|learnable_fuse-sum-shallow3-d8_crp-nocap_loss-aff"
)
for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718/isic2017; N_VAL=650; DROP=ISIC_0014979 ;;
        isic18) DATA_PATH=./data/data_isic1718/isic2018; N_VAL=808; DROP=390 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    focus=(--focus "heavy=analysis/lists/exp11_heavy_${DATASET}.csv" --drop "$DROP")
    if [ -n "$DATA_PATH_OVERRIDE" ]; then
        DATA_PATH=$DATA_PATH_OVERRIDE
        focus=()
    else
        n_found=$( (ls "$DATA_PATH/val/images" 2>/dev/null || true) | wc -l)
        [ "$n_found" -eq "$N_VAL" ] || { echo "SKIP $DATASET: expected $N_VAL val images, found $n_found"; continue; }
    fi
    echo "#================ $DATASET ================#"
    OUT="$RESULTS_DIR/exp13_${DATASET}"
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
    compare vs-D0 D0 D1 E1 E2 E3
    compare vs-D1 D1 E1 E2 E3
    compare E2-vs-E1 E1 E2
    compare E3-vs-E1 E1 E3

    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-13 — kết quả theo DSC pooled ($DATASET)" \
        --variant "D0 Gốc=egeunet_{ds}_learnable_s{seed}${RUN_SUFFIX}=model gốc + loss gốc (như paper)" \
        --variant "D1=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s{seed}${RUN_SUFFIX}=+ CSF thường dec3–5 (8 kênh), loss gốc" \
        --variant "E1=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_crp_loss-aff_s{seed}${RUN_SUFFIX}=+ propagation theo cạnh S, cap tương đối, edge head chỉ học từ loss cạnh" \
        --variant "E2=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_crp-coupled_loss-aff+comp_s{seed}${RUN_SUFFIX}=E1 + gradient segmentation / completion qua phép lan" \
        --variant "E3=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_crp-nocap_loss-aff_s{seed}${RUN_SUFFIX}=E1 không cap (chỉ pairwise)" \
        --out "$RESULTS_DIR/exp13_summary_${DATASET}.md" > /dev/null
    echo "  summary: $RESULTS_DIR/exp13_summary_${DATASET}.md"

    err=()
    for label in D0 D1 E1 E2 E3; do
        [ -n "${DIR[$label]:-}" ] && err+=(--run "$label=${DIR[$label]}")
    done
    b1="$EXP11_DIR/egeunet_${DATASET}_learnable_fuse-sum-shallow3-d8_s${SEED}"
    extra=()
    [ -z "$RUN_SUFFIX" ] && [ -f "$b1/test_results.json" ] && extra=(--run "D1e11=$b1")
    pairs=()
    for pv in D1=D0 E1=D1 E2=D1 E3=D1 E2=E1 E3=E1; do
        [ -n "${DIR[${pv%=*}]:-}" ] && [ -n "${DIR[${pv#*=}]:-}" ] && pairs+=(--pair "$pv")
    done
    [ ${#extra[@]} -gt 0 ] && [ -n "${DIR[D1]:-}" ] && pairs+=(--pair D1=D1e11)
    curve=()
    [ -n "${DIR[D1]:-}" ] && curve=(--curve-ref D1)
    if [ ${#err[@]} -gt 0 ]; then
        echo "  errors by position (val)"
        python analysis/exp11_errors.py --dataset "$DATASET" --data-path "$DATA_PATH" --device "$MECH_DEVICE" \
            --thresholds "$THRESHOLDS" "${err[@]}" "${extra[@]}" "${pairs[@]}" "${curve[@]}" "${focus[@]}" \
            --out-dir "$OUT/errors_val" | tail -1 || echo "  WARN: errors (val) failed"
        echo "  errors by position (train)"
        drop=()
        [ -z "$DATA_PATH_OVERRIDE" ] && drop=(--drop "$DROP")
        python analysis/exp11_errors.py --dataset "$DATASET" --data-path "$DATA_PATH" --device "$MECH_DEVICE" --split train \
            --thresholds 0.5 "${err[@]}" "${extra[@]}" "${pairs[@]}" "${drop[@]}" \
            --out-dir "$OUT/errors_train" | tail -1 || echo "  WARN: errors (train) failed"
    fi

    mech=()
    for label in D1 E1 E2 E3; do
        [ -n "${DIR[$label]:-}" ] && mech+=(--run "$label=${DIR[$label]}")
    done
    if [ -n "${DIR[E1]:-}${DIR[E2]:-}${DIR[E3]:-}" ]; then
        ref=()
        [ -n "${DIR[D1]:-}" ] && ref=(--ref D1)
        echo "  mechanism readouts"
        python analysis/exp13_mechanism.py --dataset "$DATASET" --data-path "$DATA_PATH" --device "$MECH_DEVICE" \
            "${mech[@]}" "${ref[@]}" "${focus[@]}" --out-dir "$OUT/mechanism" | tail -1 || echo "  WARN: mechanism failed"
    fi
    unset DIR
done
