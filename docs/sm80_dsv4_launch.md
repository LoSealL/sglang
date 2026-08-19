# DSV4-Flash on A100 (sm80) — launch recipe

First end-to-end port of DeepSeek-V4-Flash to sm80 (branch `feat/deepseek-v4-sm80`).
Status: **smoke-tested end-to-end on A100 (TP=8) — PASS** (2026-08-18, under GPU
contention from a foreign training job). See `.superpowers/sdd/task-7-report.md` for
the debugging trail. Long-context validation (2026-08-18, post-fix): see
[Long-context validation](#long-context-validation--2026-08-18) — **100k is
validated; `--context-length 102400` is allowed** after the paged MQA logits layout
fix (commit `372c6f289a`).

## Launch command (final, as smoke-tested)

```bash
MODEL=/nvme2data/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
.venv/bin/python -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code \
  --tp 8 --attention-backend dsv4 --page-size 256 \
  --moe-runner-backend marlin \
  --context-length 102400 --port 8000 --host 0.0.0.0 \
  --mem-fraction-static 0.5 \
  --chat-template scripts/sm80/dsv4_chat_template.jinja \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  > /tmp/opencode/sgl_server.log 2>&1 &
```

No environment variables required. Notes:

- **CoT requires thinking mode, not just the template flag.** The checkpoint ships no
  chat_template; `scripts/sm80/dsv4_chat_template.jinja` reproduces the official
  thinking-mode encoder frame for the raw-completions path. But with
  `--tool-call-parser deepseekv4`, `/v1/chat/completions` uses sglang's native DSV4
  encoder and does not consult `--chat-template`; thinking defaults to OFF (frame ends
  `</think>`), so responses have empty `reasoning_content`. To get CoT, send
  `"chat_template_kwargs": {"thinking": true}` per request (or launch with
  `SGLANG_DEFAULT_THINKING=1`). Verified 2026-08-18: math/Paris/multi-turn-haiku probes
  all return structured `reasoning_content` + correct `content`, no `<think>` leakage.
  Budget `max_tokens` for the reasoning tokens (a 512 cap can be consumed by thinking
  alone). Evidence: `.superpowers/sdd/task-10-report.md`.

- `--moe-runner-backend marlin` is **required on sm80**: DSV4-Flash experts are
  FP4-packed (`expert_dtype: fp4`) and the default auto→Triton runner cannot touch
  fp4/fp8 tensors on sm80 (Triton has no fp8 pointer support below sm89; deep_gemm /
  cutlass / flashinfer are sm90+). The MXFP4 Marlin MoE path handles them and is
  validated on A100 against real weights (rel err 0.3% vs the repo's
  `cast_e2m1fn_to_e4m3fn` reference).
- `--mem-fraction-static 0.5` was used for the smoke test because a foreign
  training job held 14-22 GB per GPU; it left ~38 GB free after graph capture.
  On idle GPUs `0.72` is the validated maximum (leaves headroom for the
  indexer's first-request Triton compile and NCCL buffers; 0.819 default died
  with 0.03 GiB free at the warmup request).
- Parser flags: `--reasoning-parser deepseek-v4`, `--tool-call-parser deepseekv4`.
- Weight load ~7-9 min (48 shards). Decode cuda graphs capture a few minutes more.
- Do NOT use `--kv-cache-dtype`: dsv4 pools are always the packed layout.

## Issues hit and fixes

| Symptom | Root cause | Fix |
|---|---|---|
| argparse rejects `deepseek_v4` | choices are hyphenated / concatenated | use `deepseek-v4` / `deepseekv4` |
| Triton fused-MoE `Hidden size mismatch` at graph capture | experts are FP4-packed (`w13.shape[2] = hidden/2`); Triton runner has no fp4 path | `--moe-runner-backend marlin` |
| Triton `fp8e4nv not supported in this architecture` (with `SGLANG_DSV4_FP4_DEQUANT=1`) | sm80 Triton cannot compile fp8 pointers at all — no fp8 W8A8 Triton MoE on Ampere | use marlin (fp4) instead of dequant→fp8 |
| `MXFP4 Marlin requires SM90 or SM120` | Python-only arch gate; kernel itself runs on sm80 | commit `7006772c00` allows SM80 |
| NCCL `unhandled cuda error` at warmup logits all-gather, 0.03 GiB free | memory headroom exhausted (and an external job had started grabbing GPUs) | `--mem-fraction-static 0.72` |

## Smoke test — PASS (2026-08-18 03:37–03:47 CST, under GPU contention)

Paris probe (`max_tokens=512, temperature=0`, latency 0.3 s):

```json
{"choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is **Paris**.","reasoning_content":null},"finish_reason":"stop","matched_stop":1}],"usage":{"prompt_tokens":9,"completion_tokens":9,"reasoning_tokens":0}}
```

Haiku probe (`max_tokens=256, temperature=0`):

```json
{"choices":[{"index":0,"message":{"role":"assistant","content":"Silicon rivers flow,\nParallel streams of numbers,\nFrames bloom on the screen."},"finish_reason":"stop","matched_stop":1}],"usage":{"prompt_tokens":11,"completion_tokens":18,"reasoning_tokens":0}}
```

Both clean: correct factual completion, natural language, no repetition loops.
Startup: weights + graphs ready in ~5 min (page cache warm); server reported
`The server is fired up and ready to roll!` at 03:37:06. `/health` returns
HTTP 200 with an empty body (not the string `ok`).

## Long-context validation (2026-08-18)

**Post-fix (commit `372c6f289a`): 100k context is validated.** The sm80 Triton
`paged_fp8_mqa_logits` kernel previously read the indexer cache with an interleaved
row layout while the writer stores a block layout ([64×128B values][64×4B scales]
per page) — misread scales (±1e30/NaN) poisoned top-k row selection, losing needles
from 16k tokens onward. After the fix, a needle sweep (chat-encoded via the vLLM
reference encoder, varied filler, depths 5% and 50%, temperature 0, token counts
verified) passes **22/22 runs from 8192 through 98304 tokens** — no failure onset
within `--context-length 102400`, exceeding the vLLM sm80 reference (which fails at
49k@50% and ≥65k). Evidence trail: `.superpowers/sdd/task-10-report.md`.

100k perf (98k-token prompt, cache flushed): TTFT 40.5 s; decode 13.3 tok/s at 98k
context (server-side gen throughput, cuda graphs on).

Streaming decode @98k (2026-08-18, chat API, thinking on, idle GPUs): steady-state
**13.2–13.4 tok/s** across cold, radix-cached, and long-output runs (server log:
13.39–13.42 tok/s). TTFT 40.1 s cold vs **0.44 s** on an identical-prefix rerun
(radix cache). The user-reported "~1 tps" is TTFT amortization over a short output:
58 completion tokens / 44.5 s wall = 1.30 tok/s. Same request cached: 12.0 tok/s
amortized. No decode regression; prefix reuse dominates perceived speed.
**Post KV-split fix (commit `b45190e6f0`): steady decode @98k is 53.5 tok/s (4.0×)**
— the paged fp8 MQA logits kernel went from a 1-CTA grid to
`(rows, ceil(max_len/128))`; see
[KV-split logits optimization](#kv-split-logits-optimization-2026-08-18).

Earlier pre-fix numbers (kept for the record): onset of retrieval loss at 16384 with
coherent-denial failures, degenerate decode at 100k — fully explained by the layout
bug above; the fp8-KV and chunked-prefill suspects were ruled out (same reports).

Idle-GPU benchmark medians (3 repeats, `scripts/sm80/dsv4_bench.sh`, post-layout-fix):
prefill 32×1536 **6358 tok/s** (gate ≥1000, PASS); decode bs=1 @8k ctx **49.0 tok/s**
(gate ≥50, marginal FAIL — improved from 40.6 pre-fix). Profiling verdict:
predominantly GPU-bound (80.7% kernel coverage per decode step; sparse-MLA +
paged-MQA-logits kernels ≈ 62% of busy time, ~19% host-side segment tax under the
piecewise cuda graph). Other medians: decode bs=1 @1.5k ctx 62.9 tok/s; @16k ctx
39.4 tok/s; decode bs=32 @1536 1115 tok/s.

## KV-split logits optimization (2026-08-18)

Profiling @98k ctx showed `_paged_fp8_mqa_logits` = 43.3 ms/step (70.9% of GPU busy):
the sm80 Triton kernel launched grid `(B*next_n,)` = **1 CTA at bs=1** (1/108 SMs)
and sequentially scanned ctx/4 = 24.5k rows in BLOCK_KV=64 chunks. The n-dimension is
embarrassingly parallel (each logits[m,n] is written by exactly one CTA), so the grid
is now `(B*next_n, ceil(max_len/SPLIT_KV))`, `SPLIT_KV=128` (tuned: fastest at decode
ctx 8k/24.5k; splits past a row's ctx exit immediately; the top-k consumer scans
bounded by seq_lens). Micro-bench idle A100: ctx 8192 **1.068 → 0.068 ms (16×)**,
ctx 24576 **3.091 → 0.070 ms (44×)**.

Post-fix numbers (idle GPUs): decode @98k steady **53.5 tok/s** (server-side; was
13.2–13.4); decode bs=1 @8k gate **63.2 tok/s** median of 3 (was 49.0; gate ≥49 PASS).
Prefill untouched. Evidence: `.superpowers/sdd/task-10-report.md`.

### Prefill: BLOCK_M-tiled paged MQA logits (commit `588a5c8d26`, 2026-08-19)

Prefill chunks re-decoded every KV byte per query row and ran the indexer logits
as `[64,128]x[128,64]` dots (~8-10% TC peak). The tiled kernel (`M >= 32` rows
routes to it; decode keeps the per-row kernel) decodes each 64-row KV block once
and runs one `[BLOCK_M*64,128]x[128,64]` GEMM per block. Bitwise-equal to the
per-row kernel (fixed pairwise head-reduction tree + `enable_fp_fusion=False`;
see `.superpowers/sdd/task-10-report.md`).

Micro-bench M=8192 (kernel-side, c4 ctx 2048/6144/24576): **13.5/37.5/131.5 ms
→ 5.4/16.0/63.7 ms (2.1-2.5x)**. E2E @~100k prompt: per-chunk input tok/s
3516→3920 (8k ctx), 2094→2657 (49k), 1582→2115 (81k, +34%); **TTFT 40.5 s →
35.6-38.3 s cold**. Decode unchanged: 53.2 tok/s @98k steady.

### Prefill: sparse-MLA head-tile fusion, BLOCK_H=32 (2026-08-19)

The sparse MLA prefill kernel ran one CTA per (token, 16-head tile) — with
64 replicated heads/rank that is 4 CTAs per token each re-gathering and
re-dequantizing the SAME KV rows. `BLOCK_H=32` fuses two head-tiles per CTA
(halving the redundant gather+decode work): c4-layer kernel 45.6→31.9 ms
(with `BLOCK_N=32` for ≤640-row lists), c128@94k 61.9→40.7 ms
(`BLOCK_N=64`). Enabled for `T ≥ 32` with `H % 32 == 0`; decode (`T ≤ 8`)
keeps `BLOCK_H=16` — measured faster there (215 vs 277 µs/call). A split-K
variant over the flattened per-token index lists (explicit `num_splits=`)
is implemented and tested but measured NEVER faster on A100 at prefill T
(partial write+read traffic dominates), so it stays auto-disabled below
4096 rows/token. Micro-bench + trail: `.superpowers/sdd/task-10-report.md`.

E2E @~100k prompt (2 runs): chunk@8k ctx **5039 tok/s** (was 3920),
@49k **3266** (was 2657), deepest@81k **2590** (was 2115); vs the original
pre-optimization baseline +43/+56/+64%. **TTFT 40.5 s → 28.5-28.6 s**.
Decode @98k: 52.5 tok/s steady (gate ≥50 PASS, unchanged).
