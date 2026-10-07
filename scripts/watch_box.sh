#!/usr/bin/env bash
# LOCAL health watcher for a rented box (added after EXP-15, where a GPU died mid-run and the box sat idle ~3 h).
# Every POLL seconds it reads, over SSH, the last finished epoch of every run and greps the runs' stdout for CUDA /
# Python errors. It EXITS (so whoever launched it in the background is notified) when
#   - a run log shows an error while the box has no finished analysis yet ("ALERT error ..."), or
#   - no run advanced for STALL seconds while some run is still unfinished ("ALERT stalled ..."), or
#   - the box log reports "analysis done" ("DONE").
# SSH failures are tolerated (counted, reported after 6 in a row).
#
#   POLL=1200 STALL=2400 bash scripts/watch_box.sh root@1.2.3.4 12345 isic17 /workspace/exp17_box.log [run glob]
set -u
HOST=${1:?user@host}; PORT=${2:?port}; DS=${3:?dataset}; BOXLOG=${4:?box log}; GLOB=${5:-egeunet_${DS}_*}
POLL=${POLL:-1200}; STALL=${STALL:-2400}
SSH=(ssh -p "$PORT" -o ConnectTimeout=20 -o ServerAliveInterval=20 "$HOST")
last_sum=-1; last_change=$(date +%s); fails=0
while true; do
    out=$(timeout 120 "${SSH[@]}" "cd /workspace/EGE-UNet/results 2>/dev/null || exit 3
        grep -q 'analysis done' $BOXLOG 2>/dev/null && echo DONE
        for d in $GLOB; do
            [ -d \"\$d\" ] || continue
            n=\$(grep -c 'val epoch:\\|train: epoch .*, iter:0' \"\$d/log/train.info.log\" 2>/dev/null || echo 0)
            e=\$(ls \"\$d\"/checkpoints/epoch*.pth 2>/dev/null | wc -l)
            f=\$([ -f \"\$d/test_results.json\" ] && echo 1 || echo 0)
            err=\$(grep -l -m1 'CUDA error\\|CUDA unknown error\\|Traceback' \"\$d/stdout.log\" 2>/dev/null | head -1)
            echo \"RUN \$d \$n \$f \${err:-none}\"
        done" 2>/dev/null)
    if [ -z "$out" ]; then
        fails=$((fails + 1)); [ $fails -ge 6 ] && { echo "$(date -u +%T) ALERT ssh failed $fails times in a row"; exit 2; }
        sleep "$POLL"; continue
    fi
    fails=0
    echo "$out" | grep -q '^DONE' && { echo "$(date -u +%T) DONE"; exit 0; }
    sum=$(echo "$out" | awk '/^RUN/ {s += $3} END {print s + 0}')
    unfinished=$(echo "$out" | awk '/^RUN/ && $4 == 0' | wc -l)
    errs=$(echo "$out" | awk '/^RUN/ && $4 == 0 && $5 != "none" {print $2}')
    echo "$(date -u +%T) progress $sum, unfinished $unfinished"
    if [ -n "$errs" ]; then echo "$(date -u +%T) ALERT error in: $errs"; echo "$out"; exit 1; fi
    now=$(date +%s)
    if [ "$sum" != "$last_sum" ]; then last_sum=$sum; last_change=$now
    elif [ "$unfinished" -gt 0 ] && [ $((now - last_change)) -ge "$STALL" ]; then
        echo "$(date -u +%T) ALERT stalled: no progress for $((now - last_change)) s"; echo "$out"; exit 1
    fi
    sleep "$POLL"
done
