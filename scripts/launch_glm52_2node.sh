#!/usr/bin/env bash
# Launch the 2x8 RTX 6000D GLM-5.2-FP8 cluster.
#
# Topology: node0 = 192.168.1.10 (master, HTTP on 172.16.50.10:8000)
#           node1 = 192.168.1.20 (worker, 200GbE pair ens3*/ens2* on 192.168.1.0/24)
# Layout:   TP16 + DP-attention x2 (node-aligned) + EAGLE MTP (NextN layer 78),
#           breakable prefill CUDA graphs, FP8 KV. Hostnames have no IPv4
#           mapping on these boxes, so GLOO/NCCL are pinned to explicit NICs.
#
# Usage (on each node):  ./launch_glm52_2node.sh <node-rank 0|1>
# Stop:                  pkill -f 'sglang.launch_serve[r]'; pkill -f 'sglang::[s]cheduler'; pkill -f 'sglang::[d]etokenizer'
# Model answer check:    curl http://172.16.50.10:8000/v1/chat/completions ... -d '{"model":"local-glm",...}'
#
# Note: rank0 weights load takes ~25 min on node0 (NVMe read path); node1 ~3 min.

set -euo pipefail

NODE_RANK=${1:?usage: $0 <node-rank 0|1>}
MODEL=/home/tangwy/models_src/GLM-5.2-FP8
LOG_DIR=/tmp/opencode/sgl
mkdir -p "$LOG_DIR"

case $NODE_RANK in
  0) NCCL_IF=ens3 GLOO_IF=ens3f0np0 EXTRA=(--host 172.16.50.10 --port 8000) ;;
  1) NCCL_IF=ens2 GLOO_IF=ens2f0np0 EXTRA=() ;;
  *) echo "node-rank must be 0 or 1" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/../.."   # repo root, sglang runs from the source tree
PATH=$HOME/.venv/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_SOCKET_IFNAME=$NCCL_IF \
GLOO_SOCKET_IFNAME=$GLOO_IF \
setsid nohup $HOME/.venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name local-glm \
  "${EXTRA[@]}" \
  --tp 16 --dp 2 --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 \
  --nnodes 2 --node-rank "$NODE_RANK" --dist-init-addr 192.168.1.10:25000 \
  --context-length 524288 --mem-fraction-static 0.7 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --reasoning-parser glm45 --tool-call-parser glm45 --disable-shared-experts-fusion \
  > "$LOG_DIR/node$NODE_RANK.log" 2>&1 < /dev/null &

echo "launched node-rank $NODE_RANK (pid $!), log: $LOG_DIR/node$NODE_RANK.log"
