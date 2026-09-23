#!/bin/bash
# Launch GLM-5.3-Flash (8xA100 TP8, SM80 fallbacks) in the background.
# Usage: ./serve_glm_flash.sh {start|stop|status|restart}
set -u

MODEL="zai-org/GLM-5.3-Flash"
ALIAS="local-glm-flash"
PORT=30000
LOG_FILE="$(dirname "$0")/log-glm-flash.txt"
PID_FILE="$(dirname "$0")/.glm-flash.pid"
LOG_MAX_BYTES=$((100 * 1024 * 1024)) # truncate the log at 100MB

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME=/nvme2data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$REPO/.venv/bin/python"

truncate_watchdog() {
    # Truncate (in place) once the log exceeds the cap. The server writes in
    # append mode, so writes continue at the new EOF after truncation.
    while kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; do
        sleep 30
        local size
        size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$size" -ge "$LOG_MAX_BYTES" ]; then
            : > "$LOG_FILE"
        fi
    done
}

start_watchdog() {
    setsid nohup bash -c '
        while kill -0 "$(cat "'"$PID_FILE"'" 2>/dev/null)" 2>/dev/null; do
            sleep 30
            size=$(stat -c%s "'"${LOG_FILE:?}"'" 2>/dev/null || echo 0)
            if [ "$size" -ge '"${LOG_MAX_BYTES:?}"' ]; then : > "'"${LOG_FILE:?}"'"; fi
        done' > /dev/null 2>&1 < /dev/null &
}

do_start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "already running (pid $(cat "$PID_FILE"))"
        return 0
    fi
    echo "starting $MODEL as '$ALIAS' on port $PORT, log: $LOG_FILE"
    setsid nohup "$PY" -m sglang.launch_server \
        --model-path "$MODEL" \
        --served-model-name "$ALIAS" \
        --tp 8 --kv-cache-dtype bfloat16 \
        --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
        --host 0.0.0.0 --port "$PORT" \
        --mem-fraction-static 0.85 \
        --decode-log-interval 10 \
        >> "$LOG_FILE" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    start_watchdog
    echo "server pid $pid; waiting for health..."
    for _ in $(seq 1 60); do
        sleep 10
        if curl -s -m 3 "http://127.0.0.1:$PORT/health" -o /dev/null; then
            echo "READY: http://127.0.0.1:$PORT (model alias: $ALIAS)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "FAILED — tail of $LOG_FILE:"; tail -5 "$LOG_FILE"
            return 1
        fi
    done
    echo "timeout waiting for health; see $LOG_FILE"
    return 1
}

do_stop() {
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null) || true
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid"
        echo "stopped (pid $pid)"
    else
        echo "not running"
    fi
    rm -f "$PID_FILE"
}

case "${1:-start}" in
    start) do_start ;;
    stop) do_stop ;;
    restart) do_stop; sleep 5; do_start ;;
    status)
        pid=$(cat "$PID_FILE" 2>/dev/null) || true
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            echo "running (pid $pid), log $(stat -c%s "$LOG_FILE" 2>/dev/null) bytes"
        else
            echo "not running"
        fi ;;
    *) echo "usage: $0 {start|stop|status|restart}" >&2; exit 1 ;;
esac
