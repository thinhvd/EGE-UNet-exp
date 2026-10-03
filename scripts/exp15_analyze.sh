#!/usr/bin/env bash
# EXP-15 evaluation on the held-out TEST split of the 70:10:20 protocol. ON THE BOX after scripts/exp15_train.sh (one
# dataset per box), or locally on the synced batch: RESULTS_DIR=results/EGE-UNet-results-exp15 DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp15_analyze.sh
# Per dataset:
#   1. SWA: the uniform average of the checkpoints epoch200..300 of every finished run (analysis/swa_average.py)
#      -> <run>/checkpoints/swa-epoch200-300.pth;
#   2. per-image metrics on the test split of the selected checkpoint (the decision) and of the SWA checkpoint, with the
#      interior-miss columns (normalized depth >= TAU) -> <run>/analysis/per_image_metrics_test{,_swa}.csv;
#   3. the headline summary (analysis/exp14_summary.py) for both: per-image mean DSC, per size tertile, paired
#      per-image t-tests vs E0 (the decision), vs E1 (CSF), and E3 vs E2 (router, probe), E4 vs E3 (expert
#      supervision, router), E4 vs E5p (router, supervised), E5p vs E2 (expert supervision, fixed gates);
#   4. per-image DSC significance tables with Holm (analysis/significance_tests.py), selected checkpoint;
#   5. pooled DSC of the author's protocol, for reference (analysis/exp06_summary.py on test_results.json);
#   6. mechanism (analysis/exp15_mechanism.py): LSA criterion + epoch-swing over the 11 checkpoints, gates by region,
#      forced gates / knockouts, oracle routing, aux heads; SWA model on test (all parts), selected model on test
#      (no criterion: it is in the SWA file), SWA model on val (oracle only).
# Steps 4-6 print WARN and go on if they fail.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
MECH_DEVICE=${MECH_DEVICE:-$DEVICE}
RESULTS_DIR=${RESULTS_DIR:-results}
TAU=${TAU:-0.15}
SWA_FIRST=${SWA_FIRST:-200}
SWA_LAST=${SWA_LAST:-300}
SWA_EVERY=${SWA_EVERY:-10}
THRESHOLDS=${THRESHOLDS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-}       # local smoke tests: a mini dataset (skips the image count check)
RUN_SUFFIX=${RUN_SUFFIX:-}
SWA="swa-epoch${SWA_FIRST}-${SWA_LAST}.pth"
RUNS=(
    "E0|learnable|model gốc"
    "E1|learnable_fuse-sum-shallow3-d8|+ CSF"
    "E2|learnable_fuse-sum-shallow3-d8_lsa-fixed-probe|+ 2 expert, cổng cố định 0,5 / 0,5 / 0, aux probe"
    "E3|learnable_fuse-sum-shallow3-d8_lsa-route-probe|+ 2 expert, router học, aux probe"
    "E4|learnable_fuse-sum-shallow3-d8_lsa-route-supervise|+ 2 expert, router học, aux supervise"
    "E5p|learnable_fuse-sum-shallow3-d8_lsa-fixed-supervise|+ 2 expert, cổng cố định, aux supervise"
)
for DATASET in $DATASETS; do
    # lesion-area tertile edges (fraction of the image) from the train split: step 0 S0-4 (results/exp15_step0/)
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718_701020/isic2017; N_TEST=434; EDGES=0.0779,0.2402 ;;
        isic18) DATA_PATH=./data/data_isic1718_701020/isic2018; N_TEST=538; EDGES=0.0671,0.2324 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    if [ -n "$DATA_PATH_OVERRIDE" ]; then
        DATA_PATH=$DATA_PATH_OVERRIDE
    else
        n_found=$( (ls "$DATA_PATH/test/images" 2>/dev/null || true) | wc -l)
        [ "$n_found" -eq "$N_TEST" ] || { echo "SKIP $DATASET: expected $N_TEST test images, found $n_found"; continue; }
    fi
    echo "#================ $DATASET ================#"
    OUT="$RESULTS_DIR/exp15_${DATASET}"
    mkdir -p "$OUT"
    declare -A DIR=()
    summ=() ; summ_swa=() ; desc=() ; mech=()
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label suffix what <<<"$entry"
        dir="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        d="$RESULTS_DIR/$dir"
        if [ ! -f "$d/test_results.json" ]; then echo "  not finished: $dir"; continue; fi
        python analysis/swa_average.py --run "$d" --first "$SWA_FIRST" --last "$SWA_LAST" --every "$SWA_EVERY" | tail -1
        for which in selected swa; do
            if [ "$which" = selected ]; then ck="$d"; csv=per_image_metrics_test.csv; else ck="$d/checkpoints/$SWA"; csv=per_image_metrics_test_swa.csv; fi
            echo "  per-image metrics (test, $which): $dir"
            python analysis/eval_per_image.py --checkpoint "$ck" --data-path "$DATA_PATH" --dataset "$DATASET" \
                --device "$DEVICE" --split test --interior-tau "$TAU" --thresholds "$THRESHOLDS" \
                --out-csv "$d/analysis/$csv" | tail -1
        done
        DIR[$label]="$d"
        summ+=(--run "$label=$d")
        desc+=(--describe "$label=$what")
        mech+=(--run "$label=$d")
    done
    if [ -n "${DIR[E0]:-}" ]; then
        pairs=(--pair E3=E2 --pair E4=E3 --pair E4=E5p --pair E5p=E2)
        python analysis/exp14_summary.py "${summ[@]}" "${desc[@]}" --ref E0 --ref2 E1 "${pairs[@]}" \
            --title "EXP-15 ($DATASET), checkpoint chọn theo val loss" --out "$RESULTS_DIR/exp15_summary_${DATASET}.md" > /dev/null
        python analysis/exp14_summary.py "${summ[@]}" "${desc[@]}" --ref E0 --ref2 E1 "${pairs[@]}" --csv-name per_image_metrics_test_swa.csv \
            --title "EXP-15 ($DATASET), SWA epoch ${SWA_FIRST}-${SWA_LAST}" --out "$RESULTS_DIR/exp15_summary_swa_${DATASET}.md" > /dev/null
        echo "  summary: $RESULTS_DIR/exp15_summary_${DATASET}.md, $RESULTS_DIR/exp15_summary_swa_${DATASET}.md"
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
            python analysis/significance_tests.py "${args[@]}" --csv-name per_image_metrics_test.csv --metric "$metric" \
                --out-dir "$OUT/significance_$name" > "$OUT/significance_${name}_${metric}.txt" || echo "  WARN: significance $name $metric failed"
        done
        echo "  significance: $name"
    }
    compare vs-E0 E0 E1 E2 E3 E4 E5p
    compare vs-E1 E1 E2 E3 E4 E5p
    python analysis/exp06_summary.py --results-dir "$RESULTS_DIR" --datasets "$DATASET" --seed "$SEED" \
        --title "EXP-15 — DSC gộp trên tập test (tham khảo, cách tính của paper gốc) ($DATASET)" \
        --variant "E0 Gốc=egeunet_{ds}_learnable_s{seed}${RUN_SUFFIX}=model gốc" \
        --variant "E1=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s{seed}${RUN_SUFFIX}=+ CSF" \
        --variant "E2=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_lsa-fixed-probe_s{seed}${RUN_SUFFIX}=+ 2 expert, cổng cố định, probe" \
        --variant "E3=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_lsa-route-probe_s{seed}${RUN_SUFFIX}=+ 2 expert, router, probe" \
        --variant "E4=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_lsa-route-supervise_s{seed}${RUN_SUFFIX}=+ 2 expert, router, supervise" \
        --variant "E5p=egeunet_{ds}_learnable_fuse-sum-shallow3-d8_lsa-fixed-supervise_s{seed}${RUN_SUFFIX}=+ 2 expert, cổng cố định, supervise" \
        --out "$OUT/pooled_reference.md" > /dev/null || echo "  WARN: pooled reference failed"
    if [ ${#mech[@]} -gt 0 ]; then
        common=(--dataset "$DATASET" --data-path "$DATA_PATH" --area-edges "$EDGES" --tau "$TAU" --device "$MECH_DEVICE"
                --epochs "$SWA_FIRST,$SWA_LAST,$SWA_EVERY" --out-dir "$OUT/mechanism" "${mech[@]}")
        echo "  mechanism (test, SWA)"
        python analysis/exp15_mechanism.py "${common[@]}" --split test --ckpt swa | tail -1 || echo "  WARN: mechanism test swa failed"
        echo "  mechanism (test, selected)"
        python analysis/exp15_mechanism.py "${common[@]}" --split test --ckpt selected --parts gates,force,oracle,aux | tail -1 \
            || echo "  WARN: mechanism test selected failed"
        echo "  mechanism (val, SWA, oracle)"
        python analysis/exp15_mechanism.py "${common[@]}" --split val --ckpt swa --parts oracle | tail -1 || echo "  WARN: mechanism val failed"
    fi
    unset DIR
done
