#!/usr/bin/env bash
# Pull training results from the rented GPU server down to this machine.
# Run this LOCALLY (not on the server).
#
#   SERVER=user@1.2.3.4 bash scripts/sync_results.sh                    # everything
#   SERVER=user@1.2.3.4 bash scripts/sync_results.sh egeunet_isic17_x   # one run
#   SERVER=user@1.2.3.4 PORT=2222 bash scripts/sync_results.sh          # non-default ssh port
#   DRY=1 SERVER=... bash scripts/sync_results.sh                       # show what would transfer
#   LEAN=1 SERVER=... bash scripts/sync_results.sh                      # only what analysis needs
#
# The link to a rented box can be slow and drop mid-transfer, so rsync runs with --partial inside a
# retry loop: re-running resumes rather than restarting. LEAN=1 fetches only what the analysis
# pipeline reads (best-*.pth, metrics.csv, test_results.json, log/) - about 0.5 MB per run instead
# of 5 MB - and is the project's convention: the best weights are enough for inference and for
# finetuning. What it leaves on the server is latest.pth (optimizer and scheduler state, only needed
# to resume the exact run) and the TensorBoard events; both are gone once the rented box is returned.
set -euo pipefail

SERVER=${SERVER:-}                                   # user@host  (required)
PORT=${PORT:-22}
REMOTE_DIR=${REMOTE_DIR:-/workspace/EGE-UNet/results}  # work-dir parent on the server
LOCAL_DIR=${LOCAL_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/EGE-UNet-results"}
RUN=${1:-}                                           # optional single run name

if [ -z "$SERVER" ]; then
    echo "SERVER is not set. Example: SERVER=user@1.2.3.4 bash scripts/sync_results.sh" >&2
    exit 1
fi

# A batch that retrains baselines reuses their run names, which would overwrite the earlier batch
# sitting in the default folder. Point LOCAL_DIR at a per-batch folder in that case.
if [ -z "${LOCAL_DIR_SET:-}" ] && [ -z "${LOCAL_DIR:-}" ]; then
    for clash in egeunet_isic17_learnable_s42 egeunet_isic17_none_s42; do
        if [ -d "$LOCAL_DIR/$clash" ]; then
            echo "note: $LOCAL_DIR already holds $clash." >&2
            echo "      If this batch retrained the baselines, sync it somewhere else instead:" >&2
            echo "      LOCAL_DIR=\$(pwd)/results/EGE-UNet-results-<batch> SERVER=... bash scripts/sync_results.sh" >&2
            break
        fi
    done
fi

mkdir -p "$LOCAL_DIR"

SRC="$SERVER:$REMOTE_DIR/"
DST="$LOCAL_DIR/"
if [ -n "$RUN" ]; then
    SRC="$SERVER:$REMOTE_DIR/$RUN/"
    DST="$LOCAL_DIR/$RUN/"
    mkdir -p "$DST"
fi

RSYNC_OPTS=(-avz --partial --progress
            -e "ssh -p $PORT -o ServerAliveInterval=20 -o ConnectTimeout=20")
RSYNC_OPTS+=(--exclude '*.tmp' --exclude '__pycache__/')
# LEAN=1 keeps only what the analysis pipeline reads.
[ -n "${LEAN:-}" ] && RSYNC_OPTS+=(--exclude 'latest.pth' --exclude 'summary/'
                                   --exclude 'outputs/' --exclude 'stdout.log')
[ -n "${DRY:-}" ] && RSYNC_OPTS+=(--dry-run)

echo "rsync $SRC -> $DST"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if rsync "${RSYNC_OPTS[@]}" "$SRC" "$DST"; then
        break
    fi
    echo "  transfer interrupted (attempt $attempt) - resuming in 5s..."
    sleep 5
    [ "$attempt" = 10 ] && { echo "giving up after 10 attempts" >&2; exit 1; }
done

cat <<EOF

Done. Analyse locally, e.g.:
  python analysis/eval_per_image.py --checkpoint $LOCAL_DIR/<run> \\
      --data-path ./data/data_isic1718/isic2017 --dataset isic17 \\
      --thresholds 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 \\
      --out-csv $LOCAL_DIR/<run>/analysis/per_image_metrics_full.csv
EOF
