#!/usr/bin/env bash
# EXP-14 evaluation on the held-out TEST split of the 70:10:20 protocol. ON THE BOX after scripts/exp14_train.sh (one
# dataset per box), or locally on the synced batch: RESULTS_DIR=results/EGE-UNet-results-exp14 DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp14_analyze.sh
# Per dataset:
#   1. per-image metrics of every finished run on the test split, with the interior-miss columns (normalized depth
#      >= TAU) and a threshold sweep -> <run>/analysis/per_image_metrics_test.csv;
#   2. the headline summary (analysis/exp14_summary.py): per-image mean DSC / IoU / precision / recall, per size
#      tertile, interior-miss rate, FP, pooled DSC for reference; paired per-image t-tests of every arm vs A0 (the
#      decision), vs A1 (CSF), and A2 vs A3 / A4;
#   3. per-image DSC significance tables (analysis/significance_tests.py, with Holm) for the same comparisons;
#   4. the pooled-DSC table of the author's protocol, for reference (analysis/exp06_summary.py on test_results.json);
#   5. errors by position on the test split (analysis/exp11_errors.py): FP / interior miss / threshold curves,
#      heavy-image set where its images fall in the test split.
# Steps 3-5 print WARN and go on if they fail.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
MECH_DEVICE=${MECH_DEVICE:-$DEVICE}
RESULTS_DIR=${RESULTS_DIR:-results}
TAU=${TAU:-0.15}
THRESHOLDS=${THRESHOLDS:-0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9}
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset (skips the image count check and the heavy set)
RUN_SUFFIX=${RUN_SUFFIX:-}
CSV=per_image_metrics_test.csv
RUNS=(
    "A0|learnable|model gốc, loss gốc"
    "A1|learnable_fuse-sum-shallow3-d8|+ CSF, loss gốc"
    "A2|learnable_fuse-sum-shallow3-d8_loss-ihp|+ CSF, + interior hard-positive"
    "A3|learnable_fuse-sum-shallow3-d8_loss-focal|+ CSF, + focal loss"
    "A4|learnable_fuse-sum-shallow3-d8_loss-tversky|+ CSF, + Tversky loss"
)
for DATASET in $DATASETS; do
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718_701020/isic2017; N_TEST=434; DROP=ISIC_0014979 ;;
        isic18) DATA_PATH=./data/data_isic1718_701020/isic2018; N_TEST=538; DROP=390 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    focus=(--focus "heavy=analysis/lists/exp11_heavy_${DATASET}.csv" --drop "$DROP")
    if [ -n "$DATA_PATH_OVERRIDE" ]; then
        DATA_PATH=$DATA_PATH_OVERRIDE
        focus=()
    else
        n_found=$( (ls "$DATA_PATH/test/images" 2>/dev/null || true) | wc -l)
        [ "$n_found" -eq "$N_TEST" ] || { echo "SKIP $DATASET: expected $N_TEST test images, found $n_found"; continue; }
    fi
    echo "#================ $DATASET ================#"
    OUT="$RESULTS_DIR/exp14_${DATASET}"
    mkdir -p "$OUT"
    declare -A DIR=()
    summ=() ; desc=()
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label suffix what <<<"$entry"
        dir="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        if [ ! -f "$RESULTS_DIR/$dir/test_results.json" ]; then echo "  not finished: $dir"; continue; fi
        echo "  per-image metrics (test): $dir"
        python analysis/eval_per_image.py --checkpoint "$RESULTS_DIR/$dir" --data-path "$DATA_PATH" --dataset "$DATASET" \
            --device "$DEVICE" --split test --interior-tau "$TAU" --thresholds "$THRESHOLDS" \
            --out-csv "$RESULTS_DIR/$dir/analysis/$CSV" | tail -2
        DIR[$label]="$RESULTS_DIR/$dir"
        summ+=(--run "$label=$RESULTS_DIR/$dir")
        desc+=(--describe "$label=$what")
    done
    if [ -n "${DIR[A0]:-}" ]; then
        python analysis/exp14_summary.py "${summ[@]}" "${desc[@]}" --ref A0 --ref2 A1 --pair A2=A3 --pair A2=A4 \
            --title "EXP-14 ($DATASET)" --out "$RESULTS_DIR/exp14_summary_${DATASET}.md" > /dev/null
        echo "  summary: $RESULTS_DIR/exp14_summary_${DATASET}.md"
    fi
    compare() {
        local name=$1 ref=$2; shift 2
        [ -n "${DIR[$ref]:-}" ] || { echo "  skip $name: $ref missing"; return 0; }
        local args=(--run "$ref=${DIR[$ref]}") v n=0
        for v in "$@"; do
            [ -n "${DIR[$v]:-}" ] && { args+=(--run "$v=${DIR[$v]}"); n=$((n + 1)); }
        done
        [ "$n" -gt 0 ] || { echo "  skip $name: no variant present"; return 0; }
        for metric in dsc iou; do
            python analysis/significance_tests.py "${args[@]}" --csv-name "$CSV" --metric "$metric" \
                --out-dir "$OUT/significance_$name" > "$OUT/significance_${name}_${metric}.txt" || echo "  WARN: significance $name $metric failed"
        done
        echo "  significance: $name"
    }
    compare vs-A0 A0 A1 A2 A3 A4
    compare vs-A1 A1 A2 A3 A4
    compare A2-vs-A3 A3 A2
    compare A2-vs-A4 A4 A2
    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-14 — DSC gộp trên tập test (tham khảo, cách tính của paper gốc) ($DATASET)" \
        --variant "A0 Gốc=egeunet_{ds}_learnable_s{seed}${RUN_SUFFIX}=model gốc, loss gốc" \
        --variant "A1=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s{seed}${RUN_SUFFIX}=+ CSF" \
        --variant "A2=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_loss-ihp_s{seed}${RUN_SUFFIX}=+ CSF + interior hard-positive" \
        --variant "A3=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_loss-focal_s{seed}${RUN_SUFFIX}=+ CSF + focal" \
        --variant "A4=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_loss-tversky_s{seed}${RUN_SUFFIX}=+ CSF + Tversky" \
        --out "$OUT/pooled_reference.md" > /dev/null || echo "  WARN: pooled reference failed"
    err=()
    for label in A0 A1 A2 A3 A4; do
        [ -n "${DIR[$label]:-}" ] && err+=(--run "$label=${DIR[$label]}")
    done
    pairs=()
    for pv in A1=A0 A2=A1 A3=A1 A4=A1 A2=A3 A2=A4; do
        [ -n "${DIR[${pv%=*}]:-}" ] && [ -n "${DIR[${pv#*=}]:-}" ] && pairs+=(--pair "$pv")
    done
    curve=()
    [ -n "${DIR[A1]:-}" ] && curve=(--curve-ref A1)
    if [ ${#err[@]} -gt 0 ]; then
        echo "  errors by position (test)"
        python analysis/exp11_errors.py --dataset "$DATASET" --data-path "$DATA_PATH" --device "$MECH_DEVICE" --split test \
            --thresholds "$THRESHOLDS" "${err[@]}" "${pairs[@]}" "${curve[@]}" "${focus[@]}" \
            --out-dir "$OUT/errors_test" | tail -1 || echo "  WARN: errors (test) failed"
    fi
    unset DIR
done
