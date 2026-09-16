#!/usr/bin/env bash
# Set up a rented GPU server for EGE-UNet training.
#
# Usage (on the server, from the home directory):
#   git clone https://github.com/thinhvd/EGE-UNet-exp.git && cd EGE-UNet-exp
#   git checkout <branch>            # e.g. exp/04-cross-stage-fusion
#   bash scripts/setup_server.sh
#
# Idempotent: safe to re-run after a disconnect.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY=${PY:-python3}
VENV=${VENV:-.venv}

echo "#---------- repo ----------#"
git log --oneline -1
echo "branch: $(git rev-parse --abbrev-ref HEAD)"

echo "#---------- venv ----------#"
if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install scipy            # needed by analysis/compare_runs.py (Wilcoxon)

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
if [ -d "$DATA_DIR/isic2017/train/images" ]; then
    echo "OK: $DATA_DIR/isic2017 found ($(ls "$DATA_DIR/isic2017/train/images" | wc -l) train images)"
else
    cat <<EOF
MISSING: $DATA_DIR/isic2017

Upload the dataset zip to the server and extract it so the layout is:
  data/data_isic1718/isic2017/{train,val}/{images,masks}
  data/data_isic1718/isic2018/{train,val}/{images,masks}

From the local machine, e.g.:
  scp data_isic1718.zip USER@HOST:~/EGE-UNet-exp/data/
  ssh USER@HOST 'cd ~/EGE-UNet-exp/data && mkdir -p data_isic1718 && unzip -q data_isic1718.zip -d data_isic1718'
EOF
fi

cat <<'EOF'

#---------- next ----------#
source .venv/bin/activate
python train.py --help                 # check the available flags on this branch
# training writes to --work-dir; re-running the same dir auto-resumes from checkpoints/latest.pth
# pull the results back to the local machine with scripts/sync_results.sh (run it LOCALLY)
EOF
