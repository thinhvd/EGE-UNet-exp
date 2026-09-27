#!/usr/bin/env bash
# LOCAL. Wait until a rented box has finished its runs and its box-side analysis, pull everything needed,
# verify every pulled file by md5 against the box, check that the vast.ai instance id matches the SSH
# host:port, then DESTROY the instance (the user's standing rule since 2026-09-26: once results are
# pulled and verified, destroy the box without asking). Nothing is destroyed if a run is missing or a
# checksum differs after 3 pulls.
#
#   LOCAL_DIR=$(pwd)/results/EGE-UNet-results-exp7 EXPECTED=6 PREFIX=exp07 \
#     bash scripts/finish_box.sh isic17 root@1.2.3.4 12345 <instance_id> /workspace/exp07_box.log
set -u
DS=${1:?dataset}; HOST=${2:?user@host}; PORT=${3:?ssh port}; ID=${4:?vast instance id}; DONE_LOG=${5:?log on the box}
LOCAL_DIR=${LOCAL_DIR:?set LOCAL_DIR}
EXPECTED=${EXPECTED:-6}
PREFIX=${PREFIX:-exp07}
POLL=${POLL:-300}
VAST=${VAST:-/home/thinhvd/.local/bin/vastai}
PY=${PY:-/home/thinhvd/miniconda3/bin/python}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP=$(mktemp -d)
IP=${HOST#*@}
SSH=(ssh -p "$PORT" -o ConnectTimeout=20 -o ServerAliveInterval=20 "$HOST")
log() { echo "$(date -u +%F\ %T) [$DS] $*"; }
mkdir -p "$LOCAL_DIR"

log "waiting for 'analysis done' in $DONE_LOG"
until timeout 120 "${SSH[@]}" "grep -q 'analysis done' $DONE_LOG" 2>/dev/null; do sleep "$POLL"; done
n=$(timeout 120 "${SSH[@]}" "ls /workspace/EGE-UNet/results/egeunet_${DS}_*/test_results.json 2>/dev/null | wc -l" 2>/dev/null)
[ "${n:-0}" -eq "$EXPECTED" ] || { log "ERROR: ${n:-0}/$EXPECTED runs finished - NOT destroying"; exit 1; }
log "all $EXPECTED runs finished; pulling"

for attempt in 1 2 3; do
    (cd "$REPO" && SERVER=$HOST PORT=$PORT LEAN=1 LOCAL_DIR=$LOCAL_DIR bash scripts/sync_results.sh > "$TMP/sync.log" 2>&1)
    rsync -az -e "ssh -p $PORT -o ConnectTimeout=20" --include='*/' --include='stdout*.log' --exclude='*' \
        "$HOST:/workspace/EGE-UNet/results/" "$LOCAL_DIR/" 2>>"$TMP/sync.log"
    mkdir -p "$LOCAL_DIR/box_logs_$DS"
    rsync -az -e "ssh -p $PORT -o ConnectTimeout=20" "$HOST:$DONE_LOG" "$LOCAL_DIR/box_logs_$DS/" 2>>"$TMP/sync.log"
    timeout 300 "${SSH[@]}" "cd /workspace/EGE-UNet/results && find egeunet_${DS}_* ${PREFIX}_${DS} ${PREFIX}_analyze_${DS}.log \
        ${PREFIX}_summary_${DS}.md -type f ! -name latest.pth ! -path '*/summary/*' ! -path '*/outputs/*' 2>/dev/null \
        | LC_ALL=C sort | xargs md5sum" > "$TMP/box.md5" 2>/dev/null
    (cd "$LOCAL_DIR" && cut -c35- "$TMP/box.md5" | xargs md5sum 2>/dev/null) > "$TMP/local.md5"
    nbox=$(wc -l < "$TMP/box.md5")
    if [ "$nbox" -gt $((EXPECTED * 5)) ] && cmp -s "$TMP/box.md5" "$TMP/local.md5"; then
        log "md5 OK on all $nbox files"; break
    fi
    log "md5 check failed (box $nbox files) - attempt $attempt"
    [ "$attempt" = 3 ] && { log "ERROR: pull not verified - NOT destroying"; exit 1; }
    sleep 30
done

match=$("$VAST" show instances --raw 2>/dev/null | "$PY" -c "
import json,sys
for o in json.load(sys.stdin):
    p=(o.get('ports') or {}).get('22/tcp') or [{}]
    if str(o['id'])=='$ID' and o.get('public_ipaddr')=='$IP' and str(p[0].get('HostPort'))=='$PORT': print('yes')")
[ "$match" = "yes" ] || { log "ERROR: instance $ID does not match $IP:$PORT - NOT destroying"; exit 1; }
log "destroying instance $ID ($IP:$PORT)"
"$VAST" destroy instance "$ID" -y
sleep 20
if "$VAST" show instances --raw 2>/dev/null | "$PY" -c "import json,sys; sys.exit(0 if any(str(o['id'])=='$ID' for o in json.load(sys.stdin)) else 1)"; then
    log "WARNING: instance $ID still listed after destroy"
else
    log "instance $ID destroyed"
fi
