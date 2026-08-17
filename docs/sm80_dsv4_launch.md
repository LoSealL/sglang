# DSV4-Flash on A100 (sm80) — launch recipe

First end-to-end port of DeepSeek-V4-Flash to sm80 (branch `feat/deepseek-v4-sm80`).
Status: **smoke-tested end-to-end on A100 (TP=8) — PASS** (2026-08-18, under GPU
contention from a foreign training job). See `.superpowers/sdd/task-7-report.md` for
the debugging trail.

## Launch command (final, as smoke-tested)

```bash
MODEL=/nvme2data/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
.venv/bin/python -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code \
  --tp 8 --attention-backend dsv4 --page-size 256 \
  --moe-runner-backend marlin \
  --context-length 49152 --port 8000 --host 0.0.0.0 \
  --mem-fraction-static 0.5 \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  > /tmp/opencode/sgl_server.log 2>&1 &
```

No environment variables required. Notes:

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
