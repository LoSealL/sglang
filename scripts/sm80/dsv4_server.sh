#!/bin/bash
set -euo pipefail

MODEL=/nvme2data/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
LOG=${LOG:-/tmp/opencode/sgl_server.log}
# CTX: 102400 = validated bound (needles 22/22, decode/prefill gates). >102400
# crashed once at ~186k ctx (sampler NaN, cause unconfirmed) -- see
# /tmp/opencode/repro_e2e_1m.py for the pending repro.

.venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --served-model-name dsv4 \
  --tp 8 \
  --attention-backend dsv4 \
  --page-size 256 \
  --moe-runner-backend marlin \
  --context-length "${CTX:-102400}" \
  --port "${PORT:-8000}" \
  --host 0.0.0.0 \
  --mem-fraction-static "${MEM:-0.72}" \
  --chat-template scripts/sm80/dsv4_chat_template.jinja \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --default-chat-template-kwargs '{"thinking": true}' \
  >"$LOG" 2>&1 &
echo "server pid $! -> $LOG"
