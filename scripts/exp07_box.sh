#!/usr/bin/env bash
# EXP-7 runner ON THE BOX (one dataset per box). Launches the 6 runs, waits, relaunches any run that ended
# without test_results.json (train.py auto-resumes from latest.pth; the first stdout.log is kept as
# stdout_attemptN.log), then evaluates. Its last line is "analysis done", which scripts/finish_box.sh
# waits for before pulling and destroying the box.
#
#   setsid -f bash scripts/exp07_box.sh isic17 > /workspace/exp07_box.log 2>&1 < /dev/null
set -u
DS=${1:?dataset (isic17 or isic18)}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
log() { echo "$(date -u +%F\ %T) $*"; }
nrun() { pgrep -fc "[t]rain.py" || true; }
tag_index() { case "$1" in
    *_learnable_s42) echo 0 ;; *loss-bl_s42) echo 1 ;; *loss-bl+fndp_s42) echo 2 ;; *loss-bl+fndp_s42_rep1) echo 3 ;;
    *loss-fndp_s42) echo 4 ;; *loss-blhalf_s42) echo 5 ;; esac; }
log "launch $DS"
DATASETS=$DS PARALLEL=1 bash scripts/exp07_train.sh | grep "^launching"
sleep 120; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
for attempt in 1 2; do
    relaunched=0
    for d in results/egeunet_${DS}_learnable*_s42*; do
        [ -f "$d/test_results.json" ] && continue
        mv "$d/stdout.log" "$d/stdout_attempt$attempt.log"
        log "relaunch (attempt $attempt): $d"
        DATASETS=$DS PARALLEL=1 bash scripts/exp07_train.sh "$(tag_index "$d")" | grep "^launching"
        relaunched=1; sleep 30
    done
    [ $relaunched -eq 0 ] && break
    sleep 90; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
done
log "finished: $(ls results/egeunet_${DS}_*/test_results.json | wc -l)/6"
DATASETS=$DS bash scripts/exp07_analyze.sh > "results/exp07_analyze_${DS}.log" 2>&1
log "analysis done"
