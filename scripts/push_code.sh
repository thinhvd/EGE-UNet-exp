#!/usr/bin/env bash
# Push the working tree to the GPU server. Run this LOCALLY.
#
#   SERVER=root@1.2.3.4 PORT=19825 bash scripts/push_code.sh
#
# git stays on the local machine: the server gets a plain copy of the files, plus a `.code_revision`
# stamp so each run's log can still say which commit (and whether a dirty tree) produced it.
# Results, data, notebooks, papers and reports are never pushed.
set -euo pipefail

SERVER=${SERVER:-}
PORT=${PORT:-22}
REMOTE_DIR=${REMOTE_DIR:-/workspace/EGE-UNet}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -z "$SERVER" ]; then
    echo "SERVER is not set. Example: SERVER=root@1.2.3.4 PORT=19825 bash scripts/push_code.sh" >&2
    exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
dirty=""
[ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=" (dirty)"
echo "${branch}@${commit:0:10}${dirty}" > .code_revision
echo "pushing ${branch}@${commit:0:10}${dirty} -> $SERVER:$REMOTE_DIR"

for attempt in 1 2 3 4 5; do
    if rsync -az --partial --delete-after \
            -e "ssh -p $PORT -o ServerAliveInterval=20 -o ConnectTimeout=20" \
            --exclude '.git/' --exclude 'data/' --exclude 'results/' --exclude 'paper/' \
            --exclude 'reports/' --exclude 'exp_docs/' --exclude 'notebooks/' \
            --exclude '__pycache__/' --exclude '*.pyc' \
            ./ "$SERVER:$REMOTE_DIR/"; then
        echo "code pushed"
        exit 0
    fi
    echo "  interrupted (attempt $attempt) - retrying in 5s..."
    sleep 5
done
echo "code push failed after 5 attempts" >&2
exit 1
