#!/usr/bin/env bash
# Pull training results from the rented GPU server down to this machine.
# Run this LOCALLY (not on the server).
#
#   SERVER=user@1.2.3.4 bash scripts/sync_results.sh                    # everything
#   SERVER=user@1.2.3.4 bash scripts/sync_results.sh egeunet_isic17_x   # one run
#   SERVER=user@1.2.3.4 PORT=2222 bash scripts/sync_results.sh          # non-default ssh port
#   DRY=1 SERVER=... bash scripts/sync_results.sh                       # show what would transfer
#   LEAN=1 SERVER=... bash scripts/sync_results.sh                      # skip latest.pth + tensorboard
#
# Defaults can be edited here once the server is rented.
set -euo pipefail

SERVER=${SERVER:-}                                   # user@host  (required)
PORT=${PORT:-22}
REMOTE_DIR=${REMOTE_DIR:-~/EGE-UNet-exp/results}     # work-dir parent on the server
LOCAL_DIR=${LOCAL_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/EGE-UNet-results"}
RUN=${1:-}                                           # optional single run name

if [ -z "$SERVER" ]; then
    echo "SERVER is not set. Example: SERVER=user@1.2.3.4 bash scripts/sync_results.sh" >&2
    exit 1
fi

mkdir -p "$LOCAL_DIR"

SRC="$SERVER:$REMOTE_DIR/"
DST="$LOCAL_DIR/"
if [ -n "$RUN" ]; then
    SRC="$SERVER:$REMOTE_DIR/$RUN/"
    DST="$LOCAL_DIR/$RUN/"
    mkdir -p "$DST"
fi

RSYNC_OPTS=(-avz --partial --progress -e "ssh -p $PORT")
RSYNC_OPTS+=(--exclude '*.tmp' --exclude '__pycache__/')
# LEAN=1 skips the resume state (latest.pth, only needed on the server) and TensorBoard events.
[ -n "${LEAN:-}" ] && RSYNC_OPTS+=(--exclude 'latest.pth' --exclude 'summary/')
[ -n "${DRY:-}" ] && RSYNC_OPTS+=(--dry-run)

echo "rsync $SRC -> $DST"
rsync "${RSYNC_OPTS[@]}" "$SRC" "$DST"

cat <<EOF

Done. Analyse locally, e.g.:
  python analysis/eval_per_image.py --checkpoint $LOCAL_DIR/<run> \\
      --data-path ./data/data_isic1718/isic2017 --dataset isic17 \\
      --thresholds 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 \\
      --out-csv $LOCAL_DIR/<run>/analysis/per_image_metrics_full.csv
EOF
