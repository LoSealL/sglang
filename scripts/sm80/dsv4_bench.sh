#!/usr/bin/env bash
# DSV4-Flash A100 (sm80) prefill/decode benchmark (in-process, no HTTP).
# Gates: prefill >= 1000 tok/s (bs=32 x input 1536); decode >= 50 tok/s
#        (bs=1, 8k context, 256 steps). Reference: docs/sm80_dsv4_launch.md.
#
# Runs sglang.benchmark.one_batch as a single TP=8 sweep over
# bs {1,32} x input {1536,8192,16384} x output {1,256}, REPEATS times
# (each repeat reloads weights; results append to $RESULT jsonl).
# output-len 1 => prefill-only run (prefill throughput is the metric).
#
# Usage: REPEATS=3 scripts/sm80/dsv4_bench.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=/nvme2data/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
RESULT=${RESULT:-/tmp/opencode/dsv4_bench_result.jsonl}
REPEATS=${REPEATS:-3}

mkdir -p "$(dirname "$RESULT")"

gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1; }

for i in $(seq 1 "$REPEATS"); do
  echo "=== repeat $i/$REPEATS ==="
  echo "GPU util before: $(gpu_util)"
  # ponytail: mem-fraction 0.55 keeps headroom under the foreign training job
  .venv/bin/python -m sglang.benchmark.one_batch \
    --model-path "$MODEL" --trust-remote-code \
    --tp-size 8 --attention-backend dsv4 --page-size 256 \
    --moe-runner-backend marlin --context-length 49152 \
    --mem-fraction-static 0.55 --run-name dsv4-sm80 \
    --batch-size 1 32 --input-len 1536 8192 16384 --output-len 1 256 \
    --result-filename "$RESULT" \
    > "/tmp/opencode/dsv4_bench_run${i}.log" 2>&1 \
    || { tail -40 "/tmp/opencode/dsv4_bench_run${i}.log"; exit 1; }
  grep -E "Prefill\.|Decode\.  median|Total\.|skipping" "/tmp/opencode/dsv4_bench_run${i}.log"
  echo "GPU util after: $(gpu_util)"
done

echo "=== results appended to $RESULT ==="
