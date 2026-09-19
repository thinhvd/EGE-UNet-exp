#!/usr/bin/env bash
# Prepare a rented GPU server for EGE-UNet training.
#
# The repo is pushed from the local machine with rsync (git stays local), so this script only has
# to make the Python environment usable and check that the GPU and dataset are really there:
#
#   # locally
#   rsync -az --exclude '.git/' --exclude 'data/' --exclude 'results/' --exclude 'paper/' \
#         --exclude 'reports/' --exclude '__pycache__/' --exclude 'notebooks/' \
#         -e "ssh -p <port>" ./ <user>@<host>:/workspace/EGE-UNet/
#   # on the server
#   bash /workspace/EGE-UNet/scripts/setup_server.sh
#
# Idempotent: safe to re-run after a disconnect.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Images from vast.ai and similar hosts ship a ready PyTorch env; use it instead of building one.
# Set VENV=/path/to/venv to force a different one, or VENV=new to create ./.venv from scratch.
VENV=${VENV:-auto}
if [ "$VENV" = "auto" ]; then
    for cand in /venv/main "${VIRTUAL_ENV:-}" ./.venv; do
        if [ -n "$cand" ] && [ -x "$cand/bin/python" ]; then VENV="$cand"; break; fi
    done
fi
if [ "$VENV" = "auto" ] || [ "$VENV" = "new" ]; then
    python3 -m venv .venv
    VENV=$(pwd)/.venv
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
echo "#---------- python env: $VENV ----------#"
python --version

echo "#---------- dependencies ----------#"
PIP="pip install -q"
command -v uv >/dev/null 2>&1 && PIP="uv pip install -q"
python - <<'PY' || $PIP torch torchvision
import torch, torchvision  # noqa: F401
PY
# engine.py needs sklearn, utils.py needs matplotlib, train.py needs tensorboard,
# analysis/compare_runs.py needs scipy.
$PIP scikit-learn matplotlib tensorboard scipy tqdm pillow numpy

echo "#---------- torch / CUDA ----------#"
python - <<'PY'
import torch
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY

echo "#---------- dataset ----------#"
DATA_DIR="$REPO_DIR/data/data_isic1718"
missing=0
for d in isic2017 isic2018; do
    if [ -d "$DATA_DIR/$d/train/images" ]; then
        echo "  $d: $(ls "$DATA_DIR/$d/train/images" | wc -l) train / $(ls "$DATA_DIR/$d/val/images" | wc -l) val images"
    else
        echo "  $d: MISSING"
        missing=1
    fi
done
if [ "$missing" = 1 ]; then
    cat <<EOF

The experiments so far only use isic2017 (29 MB), so pushing just that subset is enough:
  # locally
  tar -C data/data_isic1718 -czf /tmp/isic2017.tgz isic2017
  scp -P <port> /tmp/isic2017.tgz <user>@<host>:/workspace/
  # on the server
  mkdir -p $DATA_DIR && tar -C $DATA_DIR -xzf /workspace/isic2017.tgz

If scp crawls or dies part-way (a rented box can be behind a link that shapes each connection to a
few kB/s and drops it after a minute), split the tarball and push the pieces in parallel, resuming
each by sending only its missing tail:
  split -n 16 -d /tmp/isic2017.tgz /tmp/part
  # per piece, repeat until the sizes match:
  have=\$(ssh -p <port> <host> "stat -c%s /workspace/chunks/partNN 2>/dev/null || echo 0")
  tail -c +\$((have+1)) /tmp/partNN | ssh -p <port> <host> "cat >> /workspace/chunks/partNN"
Afterwards verify every piece by md5 before concatenating: a connection killed mid-write can leave
the remote file LONGER than what was sent (the remote end keeps flushing after the local side dies),
so truncate to the expected size and re-check rather than trusting the byte count.
EOF
fi

cat <<'EOF'

#---------- next ----------#
python train.py --help                  # check the flags available on this code revision
PARALLEL=1 bash scripts/exp04_train.sh  # launch the EXP-4 matrix detached (survives an SSH drop)
# pull results to the local machine with scripts/sync_results.sh (run it LOCALLY)
EOF
