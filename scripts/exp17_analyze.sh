#!/usr/bin/env bash
# EXP-17 evaluation on the TEST split of the 80/20 protocol (no validation split). ON THE BOX after
# scripts/exp17_train.sh (one dataset per box), or locally on the synced batch: RESULTS_DIR=results/EGE-UNet-results-exp17
# DEVICE=cpu.
#   DATASETS=isic17 bash scripts/exp17_analyze.sh
# Per dataset:
#   1. SWA (the headline model): uniform average of the checkpoints epoch200..300 of every finished run
#      (analysis/swa_average.py) -> <run>/checkpoints/swa-epoch200-300.pth;
#   2. per-image metrics on the test split of the SWA model and of the last epoch (final-epoch300.pth), with the
#      interior-miss columns (normalized depth >= TAU) -> <run>/analysis/per_image_metrics_test_{swa,final}.csv;
#   3. summaries (analysis/exp14_summary.py) for both: per-image mean DSC, per size tertile, precision, recall,
#      interior miss, pooled DSC; paired per-image t-tests vs A0 (the original model) and A2 vs A1 (effect of E3a on
#      LSA-CSF);
#   4. per-image DSC / IoU significance tables with Holm (analysis/significance_tests.py), SWA model;
#      (pooled DSC, the author's metric, is a column of the step-3 tables, for the SWA model and the last epoch)
#   5. mechanism (analysis/exp15_mechanism.py, labels E0 = A0, E3 = A1, E3bl = A2): metrics of the last epoch, SWA and
#      each of the 11 checkpoints (mean and SD over epochs), router gates by region and forced gates / knockouts of the
#      LSA arms (SWA model).
# Steps 4-5 print WARN and go on if they fail.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASETS=${DATASETS:-"isic17 isic18"}
SEED=${SEED:-123}
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
PROTOCOL="chia 80/20 (train / test, không val), seed ${SEED}"
RUNS=(
    "A0|E0|learnable|model gốc, loss gốc"
    "A1|E3|learnable_fuse-sum-shallow3-d8_lsa-route-probe|LSA-CSF (router + probe), loss gốc"
    "A2|E3bl|learnable_fuse-sum-shallow3-d8_lsa-route-probe_loss-bl|LSA-CSF (router + probe), loss gốc + E3a"
)
for DATASET in $DATASETS; do
    # lesion-area tertile edges (fraction of the image, 256 px) from the 80/20 train split (doc 17)
    case "$DATASET" in
        isic17) DATA_PATH=./data/data_isic1718_8020/isic2017; N_TEST=429; EDGES=0.0684,0.2200 ;;
        isic18) DATA_PATH=./data/data_isic1718_8020/isic2018; N_TEST=539; EDGES=0.0743,0.2497 ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
    esac
    if [ -n "$DATA_PATH_OVERRIDE" ]; then
        DATA_PATH=$DATA_PATH_OVERRIDE
    else
        n_found=$( (ls "$DATA_PATH/test/images" 2>/dev/null || true) | wc -l)
        [ "$n_found" -eq "$N_TEST" ] || { echo "SKIP $DATASET: expected $N_TEST test images, found $n_found"; continue; }
    fi
    echo "#================ $DATASET ================#"
    OUT="$RESULTS_DIR/exp17_${DATASET}"
    mkdir -p "$OUT"
    declare -A DIR=()
    summ=() ; desc=() ; mech=()
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label mlabel suffix what <<<"$entry"
        dir="egeunet_${DATASET}_${suffix}_s${SEED}${RUN_SUFFIX}"
        d="$RESULTS_DIR/$dir"
        if [ ! -f "$d/test_results.json" ]; then echo "  not finished: $dir"; continue; fi
        python analysis/swa_average.py --run "$d" --first "$SWA_FIRST" --last "$SWA_LAST" --every "$SWA_EVERY" | tail -1
        for which in swa final; do
            if [ "$which" = swa ]; then ck="$d/checkpoints/$SWA"; else ck="$d"; fi    # the work dir resolves to final-epoch*.pth
            echo "  per-image metrics (test, $which): $dir"
            python analysis/eval_per_image.py --checkpoint "$ck" --data-path "$DATA_PATH" --dataset "$DATASET" \
                --device "$DEVICE" --split test --interior-tau "$TAU" --thresholds "$THRESHOLDS" \
                --out-csv "$d/analysis/per_image_metrics_test_${which}.csv" | tail -1
        done
        DIR[$label]="$d"
        summ+=(--run "$label=$d")
        desc+=(--describe "$label=$what")
        mech+=(--run "$mlabel=$d")
    done
    if [ -n "${DIR[A0]:-}" ]; then
        for which in swa final; do
            tag=$([ "$which" = swa ] && echo "SWA epoch ${SWA_FIRST}-${SWA_LAST} (số chính)" || echo "epoch cuối")
            python analysis/exp14_summary.py "${summ[@]}" "${desc[@]}" --ref A0 --pair A2=A1 --csv-name "per_image_metrics_test_${which}.csv" \
                --protocol "$PROTOCOL" --title "EXP-17 ($DATASET), $tag" --out "$RESULTS_DIR/exp17_summary_${which}_${DATASET}.md" > /dev/null
        done
        echo "  summary: $RESULTS_DIR/exp17_summary_{swa,final}_${DATASET}.md"
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
            python analysis/significance_tests.py "${args[@]}" --csv-name per_image_metrics_test_swa.csv --metric "$metric" \
                --out-dir "$OUT/significance_$name" > "$OUT/significance_${name}_${metric}.txt" || echo "  WARN: significance $name $metric failed"
        done
        echo "  significance: $name"
    }
    compare vs-A0 A0 A1 A2
    compare A2-vs-A1 A1 A2
    if [ ${#mech[@]} -gt 0 ]; then
        echo "  mechanism (test, SWA)"
        python analysis/exp15_mechanism.py --dataset "$DATASET" --data-path "$DATA_PATH" --area-edges "$EDGES" --tau "$TAU" \
            --device "$MECH_DEVICE" --epochs "$SWA_FIRST,$SWA_LAST,$SWA_EVERY" --out-dir "$OUT/mechanism" "${mech[@]}" \
            --split test --ckpt swa --parts crit,gates,force,aux | tail -1 || echo "  WARN: mechanism failed"
    fi
    unset DIR
done
