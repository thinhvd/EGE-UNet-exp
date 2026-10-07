#!/usr/bin/env bash
# EXP-17 runner ON THE BOX (one dataset per box). Builds the 80/20 split from the committed lists, launches the 3
# runs, waits, relaunches any run that ended without test_results.json (train.py auto-resumes from latest.pth; the
# previous stdout.log is kept as stdout_attemptN.log; exp17_train.sh refuses to resume a dir whose log records another
# loss), then evaluates. Its last line contains "analysis done" (with the analysis exit code), which
# scripts/finish_box.sh waits for before pulling and destroying the box.
#
#   setsid -f bash scripts/exp17_box.sh isic17 > /workspace/exp17_box.log 2>&1 < /dev/null
set -u
DS=${1:?dataset (isic17 or isic18)}
SEED=${SEED:-123}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
log() { echo "$(date -u +%F\ %T) $*"; }
nrun() { pgrep -fc "[t]rain[.]py --work" || true; }
tag_index() { case "$1" in
    *_learnable_s${SEED}) echo 0 ;;
    *_lsa-route-probe_s${SEED}) echo 1 ;;
    *_lsa-route-probe_loss-bl_s${SEED}) echo 2 ;;
    esac; }
sub=$([ "$DS" = isic17 ] && echo isic2017 || echo isic2018)
python scripts/make_split_8020.py --datasets "$DS" || { log "split failed"; exit 1; }
log "split: $(ls data/data_isic1718_8020/$sub/train/images | wc -l) train / $(ls data/data_isic1718_8020/$sub/test/images | wc -l) test"
log "launch $DS"
DATASETS=$DS PARALLEL=1 bash scripts/exp17_train.sh | grep "^launching"
sleep 120; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
for attempt in 1 2; do
    relaunched=0
    for d in results/egeunet_${DS}_learnable*_s${SEED}; do
        [ -f "$d/test_results.json" ] && continue
        idx=$(tag_index "$d")
        if [ -z "$idx" ]; then log "unknown run dir, not relaunched: $d"; continue; fi
        [ -f "$d/stdout.log" ] && mv "$d/stdout.log" "$d/stdout_attempt$attempt.log"
        log "relaunch (attempt $attempt): $d"
        DATASETS=$DS PARALLEL=1 bash scripts/exp17_train.sh "$idx" | grep "^launching"
        relaunched=1; sleep 30
    done
    [ $relaunched -eq 0 ] && break
    sleep 90; while [ "$(nrun)" -gt 0 ]; do sleep 60; done
done
log "finished: $(ls results/egeunet_${DS}_*/test_results.json 2>/dev/null | wc -l)/3"
DATASETS=$DS bash scripts/exp17_analyze.sh > "results/exp17_analyze_${DS}.log" 2>&1
rc=$?
log "analysis done (exit $rc)"
