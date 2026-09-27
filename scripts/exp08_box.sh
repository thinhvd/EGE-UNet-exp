#!/usr/bin/env bash
# EXP-8 runner ON THE BOX (one dataset per box): launch the 5 runs, wait, relaunch any run that ended
# without test_results.json (auto-resume; the first stdout.log is kept as stdout_attemptN.log), evaluate.
# Its last line is "analysis done", which scripts/finish_box.sh waits for.
#
#   setsid -f bash scripts/exp08_box.sh isic17 > /workspace/exp08_box.log 2>&1 < /dev/null
set -u
DS=${1:?dataset (isic17 or isic18)}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
log() { echo "$(date -u +%F\ %T) $*"; }
nrun() { pgrep -fc "[t]rain.py" || true; }
run_index() { case "$1" in
    *_learnable_s42) echo 0 ;; *_none_fuse-sum-deep3_s42) echo 1 ;; *_none_fuse-sum-deep3_loss-blhalf_s42) echo 2 ;;
    *_none_fuse-sum_attn-all5_s42) echo 3 ;; *_none_fuse-sum_attn-all5_loss-blhalf_s42) echo 4 ;; esac; }
log "launch $DS"
DATASETS=$DS PARALLEL=1 bash scripts/exp08_train.sh | grep "^launching"
sleep 120; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
for attempt in 1 2; do
    relaunched=0
    for d in results/egeunet_${DS}_*_s42; do
        [ -f "$d/test_results.json" ] && continue
        mv "$d/stdout.log" "$d/stdout_attempt$attempt.log"
        log "relaunch (attempt $attempt): $d"
        DATASETS=$DS PARALLEL=1 bash scripts/exp08_train.sh "$(run_index "$d")" | grep "^launching"
        relaunched=1; sleep 30
    done
    [ $relaunched -eq 0 ] && break
    sleep 90; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
done
log "finished: $(ls results/egeunet_${DS}_*/test_results.json | wc -l)/5"
DATASETS=$DS bash scripts/exp08_analyze.sh > "results/exp08_analyze_${DS}.log" 2>&1
log "analysis done"
