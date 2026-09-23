#!/usr/bin/env bash
# EXP-4 post-training evaluation. Run this ON THE SERVER, after scripts/exp04_train.sh finishes.
#
#   bash scripts/exp04_analyze.sh
#
# Everything heavy (650 forward passes per run, per-threshold sweeps, the GPU cost table) happens
# here on the GPU; only small CSV/JSON files then need to cross the link to the local machine,
# which matters when that link runs at a few kB/s.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASET=${DATASET:-isic17}
SEED=${SEED:-42}
DATA_PATH=${DATA_PATH:-./data/data_isic1718/isic2017}
DEVICE=${DEVICE:-cuda}
THRESHOLDS=${THRESHOLDS:-0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}

RUNS=(
    "learnable|egeunet_${DATASET}_learnable_s${SEED}"
    "none|egeunet_${DATASET}_none_s${SEED}"
    "sum-deep3|egeunet_${DATASET}_learnable_fuse-sum-deep3_s${SEED}"
    "sum-all5|egeunet_${DATASET}_learnable_fuse-sum-all5_s${SEED}"
    "concat-deep3|egeunet_${DATASET}_learnable_fuse-concat-deep3_s${SEED}"
    "concat-all5|egeunet_${DATASET}_learnable_fuse-concat-all5_s${SEED}"
    "csaa-deep3|egeunet_${DATASET}_learnable_fuse-csaa-deep3_s${SEED}"
    "csaa-all5|egeunet_${DATASET}_learnable_fuse-csaa-all5_s${SEED}"
    "sum_attn-deep3|egeunet_${DATASET}_learnable_fuse-sum_attn-deep3_s${SEED}"
    "sum_attn-all5|egeunet_${DATASET}_learnable_fuse-sum_attn-all5_s${SEED}"
    "none+sum-deep3|egeunet_${DATASET}_none_fuse-sum-deep3_s${SEED}"
    "none+sum_attn-deep3|egeunet_${DATASET}_none_fuse-sum_attn-deep3_s${SEED}"
    "none+sum_attn-all5|egeunet_${DATASET}_none_fuse-sum_attn-all5_s${SEED}"
)

echo "#---------- per-image metrics ----------#"
present=()
for entry in "${RUNS[@]}"; do
    IFS='|' read -r label dir <<<"$entry"
    if [ ! -f "results/$dir/test_results.json" ]; then
        echo "  SKIP $label - results/$dir has no test_results.json (run unfinished?)"
        continue
    fi
    present+=("$label=results/$dir")
    echo "  $label"
    python analysis/eval_per_image.py \
        --checkpoint "results/$dir" --data-path "$DATA_PATH" --dataset "$DATASET" \
        --device "$DEVICE" --thresholds "$THRESHOLDS" \
        --out-csv "results/$dir/analysis/per_image_metrics_full.csv" \
        | tail -4
done

if [ ${#present[@]} -lt 2 ]; then
    echo "not enough finished runs to compare"; exit 1
fi

args=()
for p in "${present[@]}"; do args+=(--run "$p"); done

echo "#---------- stratified comparison (dsc) ----------#"
python analysis/compare_runs.py "${args[@]}" \
    --csv-name per_image_metrics_full.csv --metric dsc \
    --out-dir results/compare_fusion_dsc | tail -20

echo "#---------- stratified comparison (oracle_dsc) ----------#"
python analysis/compare_runs.py "${args[@]}" \
    --csv-name per_image_metrics_full.csv --metric oracle_dsc \
    --out-dir results/compare_fusion_oracle | tail -20

echo "#---------- GPU cost table ----------#"
# Latency is only meaningful on an idle GPU: measured while other runs are training, the numbers
# reflect the contention, not the model. Refuse rather than record a number that looks fine and is
# not comparable.
if [ "$(ps -eo args | grep -c '[t]rain.py')" -gt 0 ]; then
    echo "  SKIPPED: training processes are still running, so latency would measure the contention"
    echo "  rather than the model. Re-run this script once they finish."
else
bench=()
for entry in "${RUNS[@]}"; do
    IFS='|' read -r label dir <<<"$entry"
    [ -f "results/$dir/test_results.json" ] && bench+=(--checkpoint "results/$dir" --label "$label")
done
python analysis/benchmark_speed.py --device "$DEVICE" --batch-sizes 1,8 \
    --out results/benchmark_gpu.json "${bench[@]}"
fi

echo
echo "#---------- pooled summary ----------#"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r label dir <<<"$entry"
    [ -f "results/$dir/test_results.json" ] || continue
    python - "$label" "results/$dir/test_results.json" <<'PY'
import json, sys
label, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
print(f'{label:16} mIoU {100*d["miou"]:.2f}  DSC {100*d["f1_or_dsc"]:.2f}  '
      f'Se {100*d["sensitivity"]:.2f}  Sp {100*d["specificity"]:.2f}  '
      f'best epoch {d["min_epoch"]}')
PY
done

cat <<'EOF'

#---------- pull to the local machine ----------#
# From LOCAL, into a folder of its own so it cannot overwrite the earlier Colab batch:
#   SERVER=<user>@<host> PORT=<port> LEAN=1 \
#     LOCAL_DIR=$(pwd)/results/EGE-UNet-results-exp4 bash scripts/sync_results.sh
EOF
