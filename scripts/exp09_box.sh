#!/usr/bin/env bash
# EXP-9 runner ON THE BOX (one dataset per box). Launches the 6 runs, waits, relaunches any run that ended
# without test_results.json (train.py auto-resumes from latest.pth; the first stdout.log is kept as
# stdout_attemptN.log; exp09_train.sh refuses to resume a dir whose log records another loss), then
# evaluates. Its last line contains "analysis done" (with the analysis exit code), which
# scripts/finish_box.sh waits for before pulling and destroying the box.
#
#   setsid -f bash scripts/exp09_box.sh isic17 > /workspace/exp09_box.log 2>&1 < /dev/null
set -u
DS=${1:?dataset (isic17 or isic18)}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
log() { echo "$(date -u +%F\ %T) $*"; }
nrun() { pgrep -fc "[t]rain.py" || true; }
tag_index() { case "$1" in
    *_learnable_s42) echo 0 ;;
    *_learnable_loss-bl_s42) echo 1 ;;
    *_fuse-sum-shallow3-d8_loss-bl_s42) echo 2 ;;
    *_fuse-bg_stage-shallow3-d8_loss-bnd_s42) echo 3 ;;
    *_fuse-bg_stage-shallow3-d8_loss-bl+bnd_s42) echo 4 ;;
    *_fuse-bg_stage-shallow3-d8_loss-bl_s42) echo 5 ;;
    esac; }
log "launch $DS"
DATASETS=$DS PARALLEL=1 bash scripts/exp09_train.sh | grep "^launching"
sleep 120; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
for attempt in 1 2; do
    relaunched=0
    for d in results/egeunet_${DS}_learnable*_s42; do
        [ -f "$d/test_results.json" ] && continue
        idx=$(tag_index "$d")
        if [ -z "$idx" ]; then log "unknown run dir, not relaunched: $d"; continue; fi
        [ -f "$d/stdout.log" ] && mv "$d/stdout.log" "$d/stdout_attempt$attempt.log"
        log "relaunch (attempt $attempt): $d"
        DATASETS=$DS PARALLEL=1 bash scripts/exp09_train.sh "$idx" | grep "^launching"
        relaunched=1; sleep 30
    done
    [ $relaunched -eq 0 ] && break
    sleep 90; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
done
log "finished: $(ls results/egeunet_${DS}_*/test_results.json 2>/dev/null | wc -l)/6"
DATASETS=$DS bash scripts/exp09_analyze.sh > "results/exp09_analyze_${DS}.log" 2>&1
rc=$?
log "analysis done (exit $rc)"
