# DeepSeek-V4.1-Flash on NVIDIA A100 (SM80)

DeepSeek-V4.1-Flash ships for Hopper/Blackwell. Its kernels lean on instructions
Ampere does not have, so a stock SGLang checkout cannot serve it on an A100 at
all — the server dies during warmup. This document lists what was changed, how
to launch the result, what it costs, and what is still missing.

Verified on 8×A100-SXM4-80GB (81.9 GB each), `sglang 0.5.18.dev689`,
torch 2.13.0+cu130, Triton 3.7.1.

## What Ampere lacks

| Feature | Where V4.1 used it | Replacement here |
| --- | --- | --- |
| FP8 / FP4 tensor cores | routed + shared expert GEMMs | Marlin (`mma.sync` + bf16 dequant in registers) |
| TMA, wgmma | FlashMLA sparse decode/prefill, mega prefill | Triton SDPA-style kernel, torch reference |
| `torch._scaled_mm` (sm_89+) | `Fp8LinearMethod` block path | bf16 dequant + `torch.matmul` |
| Triton `fp8e4nv` dtype | KV dequant, engram gather, rope fake-quant | bytewise E4M3 decode / emulated E4M3 rounding |
| DeepGEMM `fp8_einsum` | wo_a projection | bf16 einsum fallback |

## Quick start

```bash
source ~/.zshrc
cd /path/to/sglang

# sgl-kernel is pinned to a build that gates FlashMLA on sm90/sm100; the
# version check only compares strings, so skip it for an Ampere box.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

python -m sglang.launch_server \
  --model-path /nvme2data/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<sha> \
  --tp 8 \
  --moe-runner-backend marlin \
  --context-length 512 \
  --max-total-tokens 1024 \
  --mem-fraction-static 0.97 \
  --max-running-requests 4 \
  --cuda-graph-backend-decode=disabled \
  --port 30000 --host 0.0.0.0 --trust-remote-code
```

Flags that are not optional on this hardware:

- `--moe-runner-backend marlin` — the only MXFP4 MoE path that does not need FP4
  tensor cores. Without it the expert GEMM fails during warmup.
- `--mem-fraction-static 0.97` — the fp4 experts plus the two engram tables
  (190 GiB, 22.9 GiB/card at TP8) leave very little slack; the default fraction
  trips the "minimum viable memory" check.
- `--context-length` / `--max-total-tokens` small — start here. The fp8 KV pool
  is squeezed between the expert shards and the engram tables, and an oversized
  SWA pool fails with "DSV4 SWA pool cap ... leaves no room".
- `--cuda-graph-backend-decode=disabled` — the shipped FlashMLA sparse-decode
  capsule has no sm80 image, so the decode graph cannot be captured. Decode
  instead runs the portable Triton kernel described under "Decode path".

## Memory budget

The checkpoint is 734.7 GiB on disk with a very uneven shape:

| Component | GiB | Share |
| --- | --- | --- |
| Routed experts | 535.1 | 72.8% |
| `embed` + `head` (bf16) | 190.1 | 25.9% |
| Attention | 5.5 | 0.7% |
| Everything else | 4.0 | 0.5% |

Experts are `expert_dtype: fp4` on disk, so 267.6 GiB survive; the rest is
unquantized bf16. That puts the resident set at ≈468 GiB, i.e. 58.6 GiB/card at
TP8 and ≈21 GiB/card left for KV. TP4 does not fit.

The 190 GiB of `embed`/`head` is the surprise: layers 1 and 14 each own an
engram table of 384M rows × 256 fp8 values (plus a per-32 e8m0 scale), and
`embed`/`head` themselves stay bf16. Any plan that assumes "the experts are the
memory problem" will under-provision.

## What changed in the tree

Kernel-level:

- `kernels/jit/csrc/gemm/marlin/` — instantiate the fp8 and mxfp4 Marlin configs
  DSv4.1 actually needs (`COMMON_GET_IF(kFE4M3fn)`, `GROUP_BLOCKS=2` for a
  32-element group, `m_block_size_8` variants) and add an `half2`/`kFE8M0fnu`
  scale-dequant specialization. Without these, Marlin rejects the MoE tile
  shapes with "Invalid thread config".
- `kernels/ops/attention/dsv4/dequant_k_cache.py` — decode E4M3 out of a uint8
  view (`BYTEWISE_FP8`) because Triton on SM80 refuses to build a kernel that
  mentions `fp8e4nv`, even for a dead branch. The bytewise decoder is bit-exact
  against torch on all 254 finite bytes.
- `kernels/ops/attention/dsv4/fp4_rope_fake_quant.py` — `_round_to_e4m3_fp32`
  reproduces an E4M3 cast in the value domain; bit-exact on every grid point and
  on 65536 log-uniform samples in [2⁻⁹, 448].
- `kernels/ops/embeddings/engram_gather.py` — same bytewise fp8 treatment.
- `kernels/ops/layernorm/mhc.py` — replace a `{16384, 28672}` hidden-size
  whitelist with a divisibility test so the 20480-wide DSv4.1 split works.
- `kernels/ops/attention/sparse_mla_prefill_sm80.py` (new) — portable Triton
  sparse MLA. Used for prefill.

Model/backend level:

- `layers/attention/deepseek_v4_backend.py` — route prefill to the Triton
  sparse kernel and decode to `_sparse_mla_decode_torch` on SM80; make the mega
  prefill path raise a clear error instead of failing inside FlashMLA.
- `layers/quantization/{fp8,marlin_utils_fp8,marlin_utils_fp4,mxfp4_marlin_moe}.py`
  — MXFP4-vs-NVFP4 scale handling, the MoE `global_scale` restriction, and a
  bf16 dequant fallback for block-fp8 layers whose shard width is not a Marlin
  tile multiple.
- `models/deepseek_v4.py` — bf16 fallback for the wo_a `fp8_einsum`.
- `runtime_context.py` — expose `get_platform().is_sm80`.

## Decode path

`--cuda-graph-backend-decode=disabled` aside, decode runs the portable Triton
sparse-MLA kernel from `sparse_mla_prefill_sm80.py`. Two things make that work:

- **The paged cache has to be folded back to one byte row per page.** The
  attention caller hands the kernel a `(pages, page_size, 1, bytes_per_token)`
  view, which is the shape FlashMLA detects the layout from. The paged
  dequantizer instead takes its page stride from `shape[-1]`, so passing the
  view straight through steps `page_index * 584` and reads unrelated bytes.
  `_flatten_paged_view` converts it to `(pages, page_size * bytes_per_token)`
  first. Slots confined to page 0 hide the mistake; any request whose KV spans
  pages gets garbage, so the regression test uses cross-page ids.
- **Both segments are written back-to-back** into one bf16 workspace: the SWA
  window first, then the compressed rows, matching the layout
  `_sm120_sparse_decode_fwd` builds with `cat([gathered_kv, extra_kv])`.

Measured on one request, 64 generated tokens, greedy:

| Decode implementation | Throughput |
| --- | --- |
| eager fp32 reference (`_sparse_mla_decode_torch`) | 2.3 token/s |
| Triton sparse-MLA kernel | 6.4 token/s |

The eager reference is kept as the readable oracle the kernel is checked
against.

## Known gaps

- **No CUDA graph for decode.** The shipped FlashMLA sparse-decode capsule has
  no sm80 image and the error text is compiled into the `.so`, so the graph
  cannot be captured. Every token still pays Python dispatch.
- **Prefill is Triton, not FlashMLA.** Correct, and after the tensor-core
  rewrite it sustains 3.3k-8.2k tok/s single stream, but it is still a
  hand-rolled two-pass kernel rather than FlashMLA's warp-specialized pipeline.
  Short prompts are dominated by fixed overhead rather than the kernel.
- **Decode is far from FlashMLA rates.** 6.4 token/s single-stream reflects both
  the missing graph and the portable kernel's lack of tensor-core `tl.dot`.
- **Fixed SWA window.** The header of `deepseek_v4_backend.py` hard-codes
  `SWA_WINDOW = 128`; the sparse kernels assume the same width.
- **DSpark is checked for correctness, not for speedup on real traffic.** The
  11.4 tok/s figure comes from one prompt; accept length, accuracy and DSpark
  over PD were not measured.

## Heterogeneous PD (A100 prefill + RTX 6000D decode)

The useful split is A100 prefill + SM120 decode (the RTX 6000D class, which
`deepseek_v4_memory_pool.py` and `deep_gemm_wrapper/configurer.py` name
explicitly): prefill is compute-bound and this tree runs it at bf16
tensor-core rates, while decode wants FlashMLA and CUDA graphs, which Ampere
cannot run. Hopper SM90 behaves the same way apart from the SWA note below,
since the KV bytes carry no architecture.

### The KV bytes need no conversion

`KVLayout.V4` is a pure byte format; `kv_layout.py` has no architecture
branches, and the store kernel writes the same offsets everywhere. Measured on
this tree (`test_dsv4_pd_kv_layout.py`), a page holds, per token slot:

| Byte offset within the page | Content |
| --- | --- |
| `slot * 576 + 0 .. 448` | NoPE, 448 e4m3 values |
| `slot * 576 + 448 .. 576` | RoPE, 64 bf16 values (bit-exact round trip) |
| `page_size * 576 + slot * 8` | 7 ue8m0 scale bytes, then 1 zero pad byte |
| page stride | `page_bytes(page_size)`, 149760 for page_size 256 |

Those are the offsets `sgl_kernel/deepseek_v4/kv_layout.cuh` declares
(`PagedKV::data_offset` / `scale_offset`), i.e. what a FlashMLA peer reads. So
an A100 prefill can hand its pages to an H100/B200/RTX-6000-class decode with
no repack and no requantization.

What the A100 reader must *not* do is rebase the pages. The pool pads each V4
page up to a 576-byte multiple, so the per-token view attention passes around
(`(pages, page_size, 1, 584)`) is strided, not contiguous. Taking the page
stride from `tensor.stride(0)` — the same rule `flash_mla_sm120` uses — keeps
the reader on the bytes FlashMLA and PD use. Slicing to the live span instead
rebases page N at `N * page_size * 584`, which silently reads another page.

### The four numbers that must match

A page index means nothing without the page geometry, so both roles must agree
on all of:

1. `--page-size` (the DSV4 backend pins this to 256 for this model);
2. `KVLayout` (`SGLANG_DSV4_KV_LAYOUT`, `v4` here);
3. `--tp` and the layer count — the transfer is per-layer and index-matched;
4. `SGLANG_OPT_SM120_DIRECT_SWA_KV` — **must stay off on both sides** (the
   A100 side never sets it, so in practice this means: do not enable it on the
   6000D side either). It stores
   a 256-token logical page as four 64-token physical rows (37440 bytes each),
   so a decode node that enables it alone would read page N out of page N/4.
   The compressed pools take their page size from `page_size // ratio`
   (`c4` = 64 tokens, `c128` = 2 tokens here), so they inherit the same rule.

The PD handshake now compares a descriptor rather than the layout name:
`v4;swa=v4@256:149760;c4=v4@64:37440;c128=v4@2:1728`, i.e.
`<layout>@<page tokens>:<bytes per page>` per component. Any of the four
mismatches above changes it and aborts the connection with a message naming the
offending components, instead of decoding garbage.

### Launch

Prefill (A100 host):

```bash
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 python -m sglang.launch_server \
  --model-path <DeepSeek-V4.1-Flash> --tp 8 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.97 --max-running-requests 4 \
  --cuda-graph-backend-decode=disabled \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --port 30000
```

Decode (FlashMLA host) takes the same `--tp`, `--page-size`,
`--dsv4-attn-backend flashmla` and `--disaggregation-mode decode`, plus
whatever CUDA-graph settings that GPU supports.

### Still unverified

- **No transfer was run over a real fabric.** Neither mooncake nor NIXL is
  installed in the environment used here, and
  `--disaggregation-transfer-backend fake` is rejected for
  `--disaggregation-mode prefill` outright ("Prefill server does not support
  'fake'"); it is a warmup stub that never moves page bytes either. A
  correctness run needs two hosts and a real backend.
- **Sizing.** The prefill side still holds the whole ~468 GiB resident set;
  there is no prefill-only launch profile that drops decode-only state
  (`is_prefill_only` only affects logits and attention). Whether a prefill rank
  can free the engram tables (190 GiB) is the largest single lever on the cost
  of adding this tier, and it is not investigated.

## Speculative decoding (DSpark)

DSpark runs on Ampere, but it needed one more fallback than plain serving.
The checkpoint bundles the draft head (`dspark_block_size: 5`,
`dspark_target_layer_ids: [37, 38, 39]`), so
`--speculative-algorithm DSPARK` picks it up from `--model-path` and the draft
MoE runs on the same MXFP4 Marlin path fixed above
(`speculative_moe_runner_backend: marlin`).

What broke was the indexer. `PagedIndexerMetadata` plans paged-MQA-logits
metadata with the in-tree Triton planner only while
`rows_per_request == 1`, and DSpark verify has one row per draft token (6
here). The other branch calls `deep_gemm.get_paged_mqa_logits_metadata`, whose
DeepGEMM build has no Ampere attention kernels:

```
tvm.error.InternalError: Assertion error
  (/deepgemm/csrc/apis/attention.hpp:243): Unsupported architecture
  ... deepseek_v4_backend.py:1473 init_forward_metadata_indexer
  ... dsv4/metadata.py:281 __post_init__
  ... dspark_verify.py:325 _forward_prepared_verify
```

`model_hook.py` now turns on `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` for the
Ampere family, as the SM120 branch already did for its own arch. That drops the
DeepGEMM planner (`deep_gemm_metadata = None`) and routes logits to the torch
implementation, which needs no per-arch kernel.

Measured on this host (TP8, page_size 256, single stream, 64 generated tokens):

| Path | Decode | Prefill (626-token prompt) |
| --- | --- | --- |
| no speculation, torch logits off | 6.38 tok/s | 1,560 tok/s |
| no speculation, torch logits on | 6.19 tok/s | 3,405 tok/s |
| DSPARK | 11.4 tok/s | — |

Decode is unchanged within noise; prefill is faster because the torch logits
path skips the DeepGEMM/JIT metadata planning entirely. Outputs were coherent
and matched the non-speculative run on the prompts tested, which is not a
substitute for an acceptance-rate or accuracy measurement.

PD with DSpark is a further step that was not run: it requires two SWA states
(target and draft) and describes itself through `get_dsv41_spec_layout`, whose
`kv_item_lens` / `state_item_lens` carry the byte geometry for the draft
pools, alongside the per-component descriptor this pool reports.

## Prefill performance

The first portable version reduced each `(token, head)` pair with `tl.sum`,
which cannot reach the tensor cores: one 512-wide multiply-add stream per
program. Profiling put 99.7% of prefill time there (19.5 s of a 20 s request),
with dequantization at 0.3%.

Rewriting it so each program covers a block of query heads turns the token's
scores into `[BLOCK_H, D] @ [TOPK, D]^T`, which the MMA units consume. V4's MLA
shares one K/V row across heads, so batching heads is free numerically. The
kernel runs two passes over the KV blocks — maximum first, then normalizer plus
the value product — because folding the online-softmax rescale into an MMA
accumulator loses precision.

Single stream, TP8, greedy, prompts generated with fresh random text so no
request hits the prefix cache:

| Prompt tokens | Before | After |
| --- | --- | --- |
| ~610 | 194 | 3,300 tok/s |
| ~1,240 | 194 | 5,340 tok/s |
| ~2,040 | 194 | 6,860 tok/s |
| ~2,850 | 194 | 7,600 tok/s |
| ~4,050 | — | 8,240 tok/s |

Two later changes took the 4k figure to 10,800 tok/s: tiling the sparse-MLA
kernel for the 8 heads a TP8 rank owns, and merging its two softmax passes into
one. `hc_mix_stats` then got its own SM80 schedule. Cumulative, on the same prompt:

| Change | 3,884-token TTFT | tok/s |
| --- | ---: | ---: |
| Portable kernel, tiles for 8 heads | 0.4235 s | 9,171 |
| + `hc_mix_stats` SM80 schedule | 0.4055 s | 9,598 |
| + single-pass sparse MLA | 0.3580 s | 10,841 |
| + `wo_a` matmul for G == 1, no-op output fill | 0.3490 s | 11,127 |

Measure the first request twice before believing any of these: the first
requests after a launch still pay Triton JIT for the new specializations, and
they read 30-40% slow (5,393 tok/s where the warmed server gives 9,598).

The absolute figures only hold with `#cached-token: 0`; repeated prompts reuse
the prefix cache and inflate the number several-fold. Verify against the
server's own `Prefill batch ... input throughput` line before quoting any
prefill measurement.

Note that the `input throughput` line divides by wall time since the previous
batch, so it also counts any idle gap. A prompt that arrives after a quiet spell
reports ~20 tok/s for the same work that measures 3,400 tok/s from the client.
Time-to-first-token is the measurement to trust.

### Where the time goes, and whether it is compute bound

`torch.profiler` on a 4,000-token prefill (TP8, three steps captured, so 120
launches per layer set). Shares are of summed GPU time; the last column is what
each part sustains against the A100-SXM4 bf16 peak of 312 TFLOPS.

| Kernel group | GPU time | Share | FLOPs | Achieved |
| --- | ---: | ---: | ---: | ---: |
| `_sparse_mla_prefill_kernel` | 183 ms | 18% | 1.68 TFLOP | 28 TFLOPS (9% peak) |
| TP all-reduce (`2shot_pull` + NCCL) | 282 ms | 28% | — | not communication, see below |
| MoE (`fused_marlin_moe`) | 127 ms | 13% | 8.5 TFLOP | 200 TFLOPS (64% peak) |
| Dense fp8 Marlin (attention projections) | 106 ms | 11% | 5.3 TFLOP | 121 TFLOPS (39% peak) |
| `hc_mix_stats` + `mhc_post` + combine | 109 ms | 11% | — | — |
| Everything else (fill, wo_a bf16, top-k) | 90 ms | 9% | — | — |

So prefill is **not** compute bound: the run sustains ~48 TFLOPS aggregate, 15%
of the bf16 ceiling. Only the MoE GEMM is near a practical limit, at 64%. The
attention kernel is the furthest from any roofline at 9% of compute peak and
~38% of HBM: the tiles arrive through an index list rather than a contiguous
copy, the `BLOCK_H x 512` fp32 accumulator keeps occupancy at one program per
SM, and every KV block rescales that whole accumulator for the online softmax.

Retuning for the 8 heads a TP8 rank owns is what moved it: `BLOCK_H` 16 -> 8
(padding 8 real heads to a 16-row MMA wastes more than it saves) and
`BLOCK_KV` 64 -> 32 (room for three pipeline stages), 3.66 ms -> 2.03 ms per
layer, and 8.2k -> 9.2k tok/s end to end. The end-to-end gain is smaller than
the kernel gain because the fixed per-step cost, below, does not shrink with it.

### The all-reduce share is not communication

The 28% row above is a sum of kernel durations off one rank, and it does not mean
the collectives move that much data. Two measurements say so.

Isolated, with the algorithm forced on each call (8 ranks, 50 iterations, the
production tensor shapes):

| Tensor | Algorithm | Time | Bandwidth |
| --- | --- | ---: | ---: |
| `[128, 5120]` = 1.28 MB | `2shot_pull` (what prefill picks) | 38.6 us | 33.9 GB/s |
| `[128, 5120]` = 1.28 MB | `1shot_pull` | 97.5 us | 13.4 GB/s |
| `[4004, 5120]` = 40 MB | falls through to NCCL past the 16 MB cap | — | — |

Forty of the small ones are 1.5 ms of a 358 ms step, 0.4%.

In situ, per rank over the same three profiled steps (GPU ms):

| rank | attention | `marlin_moe` | fp8 Marlin | `hc_mix` | other | all-reduce | GPU busy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 183.4 | 140.0 | 128.4 | 67.6 | 168.0 | 282.3 | 969.6 |
| 1 | 183.8 | 140.6 | 129.0 | 67.8 | 168.2 | 298.5 | 987.9 |
| 3 | 183.4 | 140.2 | 128.7 | 67.7 | 164.4 | 132.0 | 816.5 |
| 7 | 183.5 | 140.1 | 128.1 | 67.6 | 167.9 | 257.5 | 944.7 |

Every compute family agrees across ranks to under 1%. Only the all-reduce column
moves (132 ms to 298 ms), while the step takes the same ~358 ms on all eight
ranks, and the arrival spread at a shared barrier is ~2 us with a single kernel
measuring 47 us on one rank and 2,352 us on another. So that column is not data
movement, and it is not one rank running late either: it is where a rank that is
ahead spins, and the rank that is behind shows the same lost time as gaps
between kernels (86 ms/step of GPU idle on rank 3 against 29 ms on rank 1).

What the eight processes agree on is the total: 252 ms/step of compute and
106 ms/step of everything else, on every rank. And the "everything else" is CPU:
all eight TP workers sit at 100% of exactly one core through a prefill burst,
8.0 cores in total.

Batch-size scaling puts a number on the fixed part:

| Prompt tokens | TTFT | tok/s |
| ---: | ---: | ---: |
| 943 | 0.180 s | 5,195 |
| 1,903 | 0.191 s | 9,953 |
| 2,803 | 0.302 s | 9,267 |
| 3,884 | 0.360 s | 10,784 |

943 tokens costs the same 180 ms as an empty step, so ~120 ms of every step is
fixed and the marginal rate is 16.3k tok/s. At 4,004 tokens that predicts
120 + 245 = 365 ms against the 358 ms measured.

It cannot be amortized from the client either: two concurrent 3,800-token
prefills take 0.722 s against 0.364 s for one, 10,539 tok/s against 10,402, so
the scheduler runs them one at a time and charges the fixed cost to each.

### What the fixed cost leaves as levers

Faster collectives cannot help; the collectives are 1.5 ms of the 358. The
106 ms is dispatch, so the levers are, in order:

1. Fewer Python ops per layer. The three-step trace logs 875,920
   `python_function` and 58,222 `cpu_op` events, ~485 CPU ops per layer, all
   issued from one saturated thread per rank.
2. Prefill CUDA graphs, the flag-level version of (1).
   `--cuda-graph-backend-prefill full` is not wired for this attention backend:
   `deepseek_v4_backend._GraphBucket.of` has `DECODE_OR_IDLE`, `TARGET_VERIFY`
   and `DRAFT_EXTEND` branches and raises `NotImplementedError` on `EXTEND`.

   `--cuda-graph-backend-prefill breakable` does capture, and the capture fits
   once it is given room — the default bucket set does not. A full run captures
   50 shapes and eats the headroom one at a time (4.52 GB -> 0.35 GB by the ninth
   bucket); at `--max-total-tokens 4608 --mem-fraction-static 0.95
   --cuda-graph-bs-prefill 4096 --cuda-graph-prefill-max-context 4096` with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, a single bucket captures
   and the server serves. Reserving the graph's memory before the KV pool is
   sized is the same lever from the other side: the pool is only ever sized to
   `mem-fraction-static`, so capping `--max-total-tokens` is what leaves the
   ~320 MiB a capture needs.

   Replay needs the batch to be exactly its bucket, and both halves of that are
   now in the tree: `token_req_indices` sizes `repeat_interleave` from
   `extend_seq_lens_cpu` (the same lengths, so no sync, and a capture no longer
   trips), and the runner rejects a padded batch for this model instead of
   replaying it into a fixed `result_size` (`Repeat.cu: result_size ==
   cumsum_ptr[size - 1]`). Pinning `--chunked-prefill-size` to the captured
   bucket then does replay: the batch line reports `cuda graph: True` for the
   full chunk and `False` for the tail.

   It buys nothing. The same ~2,050-token prompt, chunked at 2,048 so that the
   first chunk is exactly the bucket, costs 0.330 s with the graph and 0.332 s
   with `--cuda-graph-backend-prefill disabled`. A 2,048-token replay is no
   faster than an eager step of the same size, so `breakable` captures
   fragments and breaks around the rest, and the 120 ms is not in the part it
   captures.

   `full` is not a wiring change here, and this is the finding that closes it
   on Ampere. Everything that makes the V4.1 prefill capturable lives behind
   `low_ratio_prefill_graph`, and that property requires
   `has_dense_fp4_indexer() and is_sm100_or_newer()`:

   ```python
   return (bool(self.low_ratios) and has_dense_fp4_indexer()
           and is_sm100_or_newer() and get_parallel().attn_cp_size == 1)
   ```

   On A100 it is False by construction, because the capturable variant rides on
   DeepGEMM's fp4 MQA-logits indexer, which has no SM80 kernels -- the same
   absence that forces the torch indexer and `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`
   here. That is why `breakable` fell through to the eager `token_req_indices`
   branch instead of the hoisted one: the hoisted metadata (`low_ratio_req_indices`,
   `_low_ratio_prefill_indexer_metadata`, `_prefill_graph_full_topk`) was never
   built. So `full` needs a torch-indexer port of that metadata and top-k, whose
   shapes are data-dependent -- a feature, not a flag.

   What that port would cost, measured by un-gating the property on SM80 and
   capturing: it reaches `full_topk.topk_prefill_captured` ->
   `_deep_gemm_prefill_captured` -> `deep_gemm.fp8_fp4_paged_mqa_logits`, which
   raises `TypeError: Mismatched type on argument #6 ... Expected DLTensor* but
   got None` because the fp4 quant half does not exist here. The DeepGEMM-free
   replacement does exist and is documented capture-compatible
   (`fp8_paged_mqa_logits_torch`), but it reads the paged fp8 index-K cache,
   while the Ampere low-ratio path scores through `dsv41_sparse.scores` driven by
   `prefill_requests`, which walks requests and their variable-length chunks in
   Python (`for request, chunks in requests: ... chunk.scores.topk(...)`). A
   graph cannot replay that. Capturing it means rewriting the V4.1 prefill top-k
   for the torch indexer in the fixed row-chunk form the fp4 path uses
   (`metadata.row_chunks()`, `out_page_indices`), not swapping one call.

3. Two-batch overlap is what the 28% row suggested, and it would overlap the
   collectives with compute — but there is nothing to overlap. It is also
   blocked for DSV4.1 by `deepseek_v4_hook.validate_deepseek_v41_features`, and
   the DSV4 path it would reuse asserts on the `hc_pre_from_prev_sublayer`
   scheme.

The ceiling: dropping the 120 ms puts a 4,004-token prefill at 16.8k tok/s.

## Tests

```bash
python -m pytest test/registered/kernels/ops/attention/dsv4/test_sparse_mla_sm80.py
```

Covers agreement with an fp32 reference, `-1` sentinel handling, the
all-invalid row, the attention sink, `topk_length` prefix truncation, and the
paged-view flattening that decode depends on (with cross-page slot ids, since
page-0-confined ids hide a wrong page stride).
