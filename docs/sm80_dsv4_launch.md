# DSV4-Flash on A100 (sm80) — launch recipe

First end-to-end port of DeepSeek-V4-Flash to sm80 (branch `feat/deepseek-v4-sm80`).
Status: launch recipe validated up to serving (weights, cuda graphs, warmup forward);
final smoke outputs pending — GPUs were lost to an external job before the smoke
curls could run. See `.superpowers/sdd/task-7-report.md` for the debugging trail.

## Launch command

```bash
MODEL=/nvme2data/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
.venv/bin/python -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code \
  --tp 8 --attention-backend dsv4 --page-size 256 \
  --context-length 49152 --port 8000 --host 0.0.0.0 \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --moe-runner-backend marlin --mem-fraction-static 0.72 \
  > /tmp/opencode/sgl_server.log 2>&1 &
```

No environment variables required. Notes:

- `--moe-runner-backend marlin` is **required on sm80**: DSV4-Flash experts are
  FP4-packed (`expert_dtype: fp4`) and the default auto→Triton runner cannot touch
  fp4/fp8 tensors on sm80 (Triton has no fp8 pointer support below sm89; deep_gemm /
  cutlass / flashinfer are sm90+). The MXFP4 Marlin MoE path handles them and is
  validated on A100 against real weights (rel err 0.3% vs the repo's
  `cast_e2m1fn_to_e4m3fn` reference).
- `--mem-fraction-static 0.72` leaves headroom for serving-time JIT compiles (the
  indexer's `_paged_fp8_mqa_logits` Triton kernel compiles on first request) and NCCL
  buffers. 0.819 (default) died with 0.03 GiB free at the warmup request.
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

## Smoke test (PENDING — GPUs occupied by an external training job)

To run once GPUs are free:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","max_tokens":512,"temperature":0,"messages":[{"role":"user","content":"The capital of France is"}]}'
# expect "Paris" in content or reasoning_content (reasoning model)

curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","max_tokens":512,"temperature":0,"messages":[{"role":"user","content":"Write a haiku about GPUs"}]}'
# expect natural language (garble detection)
```
