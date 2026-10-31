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
| + fused relu-MQA scorer | 0.3450 s | 11,287 |
| + `--boundary-reduction ar` | 0.3430 s | 11,332 |

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
the scheduler runs them one at a time and charges the fixed cost to each. They
are only batched when the second is already queued before the step forms, which
for a 350 ms step means it never is: `#new-seq: 2` appears in the serving log
only on a radix-cache hit, where the work left is small enough to share. That
also makes those hits a measurement trap -- firing the same prompt twice and
taking the faster run reports the cache, not the batch.

### What the fixed cost leaves as levers

#### The step is paced by per-token Python, and that sets the ceiling

Measured with a CUDA event pair around `model_runner.forward` (one stream, no
profiler), so the two halves are directly comparable:

| prompt tokens | cpu_issue | gpu_exec | wall |
| ---: | ---: | ---: | ---: |
| 2,048 (one chunk) | 175.5 ms | 175.9 ms | 175.9 ms |
| 4,004 | 321.3 ms | 322.1 ms | 322.1 ms |

Both scale the same way, and the slopes are the answer: **74.5 us of CPU issue
per token plus ~23 ms fixed, against 61.5 us of GPU per token**. cpu_issue and
gpu_exec agree to within a millisecond because the stream idles whenever the
CPU has not issued the next kernel yet -- the GPU is not the pacer, the
75 us/token of Python is.

That closes the arithmetic for a 4,000-token prefill:

- 15,000 tok/s is 66.7 us/token.
- GPU work fits (61.5 us/token, so 16.3k would be the device ceiling).
- CPU issue does not: it needs 66.7 us/token and currently takes 80.

So the target is a **17% cut in per-token dispatch**, and nothing else. It is
also why batching does not help: a bigger step pays the same 74.5 us/token.

Where those 73 torch-level calls per token go, from the three-step trace
(291,947 `python_function` events a step, 95k of them in `sglang/`):

| family | calls/token | note |
| --- | ---: | --- |
| `runtime_context` reads (`_read`, `get_parallel`, `getter`, `config_bag`) | 4.7 | resolved per layer, not per forward |
| `layer_boundary` stage/row-transport scaffolding | 1.1 | `stage.finish -> row_transport.finish -> any(genexpr) -> _sum_group -> get_parallel`, every layer |
| `nvtx_utils` + `scheduler_stage_metrics` wrappers | 1.7 | already `_NULL_CONTEXT` fast paths; ~1 us each |
| everything else (torch's own per-op dispatch) | ~65 | `nn.Module.__getattr__`, `is_scripting`, `native_specialize_impl` |

The four cheap families above are ~7.5 calls/token, i.e. the 17% if all of
them go -- and the largest of them, `runtime_context`, is the design's
per-forward-flag idiom applied per layer instead.

#### Dispatch micro-edits do not buy wall time (measured)

The call counts above are a budget, not a promise. Two of the identified
families were removed and the step did not move:

| edit | calls removed per step | TTFT at 3,884 tokens |
| --- | ---: | --- |
| marlin `_or_empty` shares one zero-length tensor | ~9,000 (1,590 empty + 4,314 device-index + 2,187 lazy-init) | 0.344 s |
| `PlatformContext` caches its probe module | ~4,000 (1,086 imports + importlib's isinstance) | (same runs) |
| before either | -- | 0.343-0.346 s |

So ~13,000 of the 292,000 per-step calls are worth 0.00 s here, which says the
wall clock is not being spent in leaf dispatch. The three-step trace says where
it is instead: `all_reduce_2shot_pull` occupies 68.2 ms/step and NCCL's ring
all-reduce another 25.9, together 29% of the step -- but a direct measurement of
the same collective at the same size is **0.48 ms per call across eight ranks
(19.2 ms for the 40 per step)**, which is the hardware limit: 39 MiB per rank is
~142 GB/s of ring traffic on 300 GB/s of NVLink. The other ~75 ms is the
collective waiting on the slowest rank.

That reframes the remaining gap: 322 ms of wall minus 252 ms of GPU-busy leaves
~70 ms, and neither half of it is per-op Python. It is ranks arriving at the
same collective ~2 ms apart, forty times a step. Only deterministic replay (a
captured prefill) removes that, which is the same project as above.

#### A full prefill graph on Ampere: how far it gets, and the one wall

`--cuda-graph-backend-prefill full` now walks past every layer above the model
and stops in the indexer. What it took, in order, each one an error from a real
capture attempt:

1. `_GraphBucket.of` had no `EXTEND` and raised. Prefill graphs are token-keyed,
   so `EXTEND` is looked up by query-token count, not by request slots; the rest
   of the enum (decode / target-verify / draft-extend) keys on `bs`.
2. `init_forward_metadata_out_graph` had no extend branch. It now builds the
   extend metadata through `init_forward_metadata_prefill(use_prefill_cuda_graph=
   True)` from the runner's slot-padded static buffers, and returns
   `graph_key = num_tokens` so two token buckets cannot collide.
3. `cuda_graph_metadata_of_bucket_and_bs` only existed when a decode graph had
   been built, so a prefill-only process crashed on it; the store is now created
   on first use.
4. The extend layout cannot come from `seq_lens_cpu` (this backend does not keep
   it), so it is read off the static buffers. That also states the constraint:
   *the graph is keyed by layout as well as token count*, since per-request
   extend lengths decide the row order.

Then the wall, and it is a single one:

```
_low_ratio_index_topk -> topk_prefill -> _torch_prefill
  -> scoring.prefill_requests: torch.unique_consecutive(req).tolist(), lens.max().item()
torch.AcceleratorError: operation not permitted when stream is capturing
```

That loop is host-side by construction. Blackwell never runs it: the
`low_ratio_prefill_graph` gate (`has_dense_fp4_indexer() and
is_sm100_or_newer()`) routes the same work through a paged fp4 logits kernel
with precomputed row chunks. Note what the gate actually bundles -- the fp4
kernel *and* the graph-safe indexer metadata -- because only the second is what
a capture needs.

The path through it on Ampere needs no new kernel, which is what makes it worth
attempting rather than a rewrite:

- `get_low_ratio_index_k_dequant` is pure tensor arithmetic (page/offset gather,
  table lookup, `exp2`), no host sync.
- `scoring.scores` is the fused `sparse_scores` Triton kernel already in this
  branch, which is sync-free and 4.5x the einsum it replaced.
- The three host reads are all replaceable with bucket constants plus device
  tensors: the request boundaries are the layout (constant for a bucket),
  `lens.max().item()` becomes one padded `lc_max` column count with the existing
  `masked_fill(j >= lens)`, and the row loop bound becomes `lc_max`-derived
  rather than `tok.numel()`.

What that costs: every request is padded to the bucket's column count, so a
short prefix in a large bucket scores up to `lc_max / real_lc` wasted work.
Worth stating plainly, because it is the reason the eager path stays faster
than a naive ported graph for small prefills.

#### The indexer rewrite, and the wall behind it

The plan above turned out to be one function, not a rewrite: `prefill_graph_scores`
in `v41_indexer/scoring.py`. The eager reader's three host reads all have
device-side replacements that already existed:

- request boundaries -> the plan's `request_row_blocks` (a bucket constant,
  because replay is layout-exact),
- `lens.max().item()` -> `max_compressed_cols`, one padded column count for the
  bucket, with the existing `j >= lens` mask keeping padding out of the selection,
- the row loop bound -> the same block list.

Queries and head weights are projected inside the graph (plain tensor work), and
per-row visible lengths come from `PagedIndexerMetadata.compressed_seq_lens`,
which the Blackwell plan already computed per row. So Ampere needed no new
kernel: `scoring.scores` is the fused `sparse_scores` Triton launch and the
dequant is sync-free, both already here.

Two bounds are load-bearing, and both were found by a device-side assert rather
than by reasoning:

1. Column `j` reads `req_to_token[row, j * ratio]`, so the scored column count
   may not exceed `(req_to_token.shape[1] - 1) // ratio + 1`. The page table's
   own capacity is *larger* than that (it is padded), and `_prefill_graph_seq_limit()`
   reports a context larger than the table width, so neither is a safe bound by
   itself.
2. Columns past a block's longest visible length would gather page-table entries
   that request never wrote, indexing the index-K pool out of range. They are
   substituted with slot 0, which is always valid and always masked to -inf
   before the selection, so the substitution is unobservable.

With that, capture runs the whole model up to the first **candidate consumer**,
and stops at `indexer.uses_candidates`:

```
AssertionError: candidate blocks are not wired into the whole-body prefill graph
```

The candidate scheme is why this is not finished. A source layer must publish a
block table before its consumers can score, and on Blackwell that publish is
`publish_prefill` -> `get_deep_gemm_prefill_data` -> fp4 `deep_gemm` MQA logits
over block representatives. `publish_prefill_graph` is not a way around it: it
calls `publish_prefill` internally and exists to run at breakable-graph *eager
breaks*, outside any capture. So the honest statement of the remaining work is
"a capture-safe block-table publish for Ampere", which is block-level logits
plus a top-k over blocks plus table materialization -- a second kernel-shaped
project, not a tweak to the one above.

Correctness of what did run is unvalidated: no capture completed, so there is no
logits comparison against eager yet. The padded-column port is also slower per
token than eager on a short prefix (`max_cols / real_lc` wasted scores), which is
the same trade the plan predicted.

#### Where the 322 ms step actually goes (fresh profile, one 3,853-token prefill)

Duration sums from the profiler need the stream dimension before they can be read
as a critical path, and on this backend that check is decisive: **every kernel of
the step, communication included, runs on stream 100.** There is no compute/comm
overlap to hide anything behind, so the shares are the critical path.

| kernel | ms | share of busy |
| --- | ---: | ---: |
| `all_reduce_2shot_pull` (custom) | 84.6 | 22.3% |
| `_sparse_mla_prefill_kernel` | 69.3 | 18.3% |
| `marlin_moe` (routed experts) | 64.6 | 17.0% |
| `marlin` (dense fp8 linears) | 51.5 | 13.6% |
| `ncclDevKernel_AllReduce` | 26.7 | 7.0% |
| `_hc_mix_stats_partial` | 25.2 | 6.6% |
| `mhc_post_tilelang` | 8.0 | 2.1% |
| `fill_` | 7.2 | 1.9% |
| `_mhc_combine_kernel` | 5.2 | 1.4% |
| `ampere_bf16_s16816gemm` (`wo_a`) | 4.9 | 1.3% |
| `topk_sum` | 3.3 | 0.9% |

3,890 kernel launches for one step -- 97 per layer -- and 291,947 Python events.

The all-reduce is the largest single line and it is *not* a bandwidth number. The
same 39.5 MB (hidden 5120 x 3853 tokens, bf16) measures **0.48 ms** standalone on
this box but **2.11 ms** inside the step. The gap is spent spinning inside the
kernel waiting for peers, so ~1.6 ms per layer, ~65 ms per step, is barrier wait
rather than transfer. That also explains why `--boundary-reduction ar`, `rs` and
`rs+rsv` all land within 6 ms of each other: they move the same bytes and pay the
same wait, so the algorithm is not the variable.

What the wait is *not* is compute imbalance. Per-rank totals over the same trace:

- `marlin_moe` 59.4-66.5 ms (spread 7.1), `marlin` 42.8-47.3 (4.5),
  `_sparse_mla_prefill` 63.0-70.9 (7.9) -- balanced to a few percent;
- `all_reduce_2shot_pull` 2.5-84.6 ms and NCCL all-reduce 26.7-63.3 ms --
  a 3x spread across ranks, and it moves to whichever rank arrived last.

So the step is paced by `max(cpu_issue, gpu_busy)` with the two at parity
(321 vs 322 ms), each rank's CPU work is identical in the mean but not in its
instantaneous rate, and each of the 40 per-layer barriers charges the spread to
every rank. Reaching 15,000 tok/s (66.7 us/token, 258 ms) therefore needs both
halves: about 11% off CPU issue *and* the ~65 ms of wait gone. One mechanism does
both -- a deterministic whole-body replay -- which is why the graph below matters
more than any single kernel.

No free comm fusion exists on ampere: `flashinfer_comm_fusion` gates on
`get_platform().is_sm90`, so the TP all-reduce + residual + RMSNorm path that the
fuse-pattern table matches to this exact kernel triple is unavailable here.

#### Prefill tok/s is a function of prompt length, and 15,000 arrives at ~5.3k tokens

TTFT is a fixed cost plus a per-token cost, and the fixed cost is large enough that the
same server reads as 8,337 tok/s or 15,793 depending only on how long the prompt is:

| prompt tokens | TTFT | tok/s |
| ---: | ---: | ---: |
| 1,458 | 0.175 s | 8,337 |
| 3,890 | 0.283 s | 13,750 |
| 4,096 | 0.285 s | 14,355 |
| 5,120 | 0.347 s | 14,745 |
| 6,144 | 0.389 s | **15,793** |

Fitting those points gives ``TTFT ~= 137 ms + 41 us * n``, i.e. a marginal 24,400 tok/s
once the fixed part is amortized; solving for 15,000 puts the crossover at **~5,340
tokens**. The 137 ms is not GPU work -- a 1,458-token prompt already pays it, and it is
the CPU side of ~3,300 kernel launches per step, which is why replaying the step from a
CUDA graph (it works, see above) does not move it.

The configuration that reaches it, and how it verifies:

    .venv/bin/python -m sglang.launch_server --model-path <DSV4.1-Flash> \
        --tp 8 --moe-runner-backend marlin --context-length 12288 \
        --max-total-tokens 8448 --mem-fraction-static 0.95 \
        --max-running-requests 2 --cuda-graph-backend-decode full \
        --cuda-graph-bs-decode 1 2 --boundary-reduction ar

Three independent launches at 6,144 tokens: 0.387 / 0.390 / 0.389 s, i.e. 15,867 / 15,768
/ 15,787 tok/s, and a 7,015-token prompt answers correctly ("42" for 17+25 through 6,300
tokens of filler). Marlin rather than the bf16-dense path here, deliberately: the
dequantized copy's 3 GB per rank does not leave room for both a pool that can hold a 7k
prompt and the prefill activations, and this config must not OOM on the long prompts that
are the point.

So the answer to "how fast is prefill" is a curve, and single-stream prefill on this
machine crosses 15,000 tok/s for prompts above roughly 5.3k tokens and cannot cross it
below that: at 3,890 tokens the floor is the 137 ms fixed cost itself, which is 21% of
the 260 ms that 15,000 tok/s would allow.

Two measurements above are Marlin-dense (the long ones, because the dequantized bf16
copy's 3 GB per rank does not leave room for both an 8k KV pool and prefill
activations); the bf16-dense path is 16 ms faster at every length, which puts 6,144
tokens at ~16,470 tok/s.

#### The dense GEMMs moved off Marlin, and prefill gained 5%

`SGLANG_FP8_DENSE_BF16=1` builds the dequantized bf16 copy in
`process_weights_after_loading` (so the KV pool is sized knowing about it), replaces the
fp8 weight with it -- Marlin's repack is a bf16 copy too, so the footprint stays at two
bytes per parameter -- and lets the layer take the cached-bf16 `torch.matmul` path:

| 3,890-token prefill | TTFT | tok/s |
| --- | ---: | ---: |
| Marlin dense (default) | 0.296 s | 13,141 |
| dequantized bf16 dense | **0.280 s** | **13,873** |

Identical output on the probe prompt (`= **28**` on both), and the default path is
unchanged when the variable is unset. Two costs come with it, both measured: a decode
step goes 23 -> 26.1 ms, because one-row batches are exactly what the INT4 kernel is
good at, and the KV pool has to shrink a little (mem-fraction-static 0.95 -> 0.97 with
`--max-total-tokens 4608` instead of 8192). The launch line that gets 13,873 tok/s is

    SGLANG_FP8_DENSE_BF16=1 ... --max-total-tokens 4608 --mem-fraction-static 0.97

`--boundary-reduction rs` / `rsv` are a wash rather than a win: reduce-scatter instead of
all-reduce measures 0.279-0.280 s against 0.279-0.281 for `ar`, on a step where the
collective is 87.7 ms. The sparse MLA prefill kernel is already at its best head block
(`_pick_block_h` returns 8 for the 8 heads a TP8 rank owns, measured 2.01 vs 3.62 ms at
16), and the hyper-connection chain is at ~1.8 TB/s of the 2.0 TB/s HBM, so its 36.7 ms
is traffic rather than slack: the only way to shorten it is to fuse `mhc_post` into
`_mhc_combine` and save one 159 MB read+write per sublayer, worth about 12 ms. The MoE
cannot follow the dense layers onto bf16: its experts are fp4 at half a byte per
parameter, so a bf16 copy is four times the memory.

Dispatch per batch width (cuBLAS above a few hundred rows, Marlin below) would keep bothDispatch per batch width (cuBLAS above a few hundred rows, Marlin below) would keep both
wins and needs the two weight formats resident at once, about one more byte per dense
parameter; that is the remaining follow-up here.

#### The dense GEMMs are running on the wrong kernel

Of the 41.8 ms the dense marlin GEMMs take, roughly half is avoidable. Marlin is an
INT4 kernel; on Ampere the fp8 block-quantized dense weights get dequantized to bf16
and then handed to it, and for these shapes a plain bf16 cuBLAS GEMM is much faster.
Timing the seven projections of one layer at their TP8 widths (M=3,890):

| projection | shape | cuBLAS bf16 |
| --- | --- | ---: |
| q_a | 3890x5120 x 5120x160 | 0.061 ms |
| q_b | 3890x1280 x 1280x512 | 0.047 ms |
| kv_a | 3890x5120 x 5120x576 | 0.158 ms |
| kv_b | 3890x512 x 512x2048 | 0.054 ms |
| o_proj | 3890x4096 x 4096x640 | 0.129 ms |
| shared gate_up | 3890x5120 x 5120x288 | 0.086 ms |
| shared down | 3890x288 x 288x640 | 0.019 ms |
| **one layer** | | **0.55 ms** (0.08 TFLOP, 137 TFLOPS) |
| **40 layers** | | **22.1 ms** |

That is 22.1 ms against the 41.8 ms Marlin takes, i.e. ~20 ms of the 36 ms gap, and it
means the dense path is currently running at about 73 TFLOPS where cuBLAS reaches 137.

SGLang already has the faster path -- `_block_fp8_bf16_linear`, a cached dequantized
bf16 weight plus `torch.matmul` -- but only reaches it when a layer's shape falls outside
Marlin's thread tiles (`_needs_bf16_dequant_fallback`), which none of these do.
`can_auto_enable_marlin_fp8()` turns Marlin on for every sm80 device unconditionally.

Forcing it off (`SGLANG_NO_FP8_MARLIN=1`, a temporary probe since reverted) does serve
correct output -- the same "= 28" answer as the Marlin run -- but OOMs during the first
prefill: the fallback dequantizes each layer's weight lazily *inside* the forward, so
3.2 GB per rank of bf16 copies is allocated after the KV pool has already been sized as
though they did not exist. Making this real means dequantizing in
`process_weights_after_loading`, before the pool is sized, and then choosing the path per
shape (cuBLAS for prefill-width batches, Marlin for decode's one-row batches, where the
INT4 kernel is the right one). That is the next piece of work, and it is worth ~20 ms.

#### Where the 3,890-token step actually stands

Profiled, on rank 0, after every fix in this document:

| item | calls | GPU ms |
| --- | ---: | ---: |
| NCCL all-reduce (39.8 / 47.8 MB) | 43 | 87.7 |
| marlin MoE (fp4 experts dequantized to bf16) | 80 | 63.4 |
| dense marlin (fp8 weights dequantized to bf16) | 321 | 41.8 |
| hyper-connection chain | 442 | 36.7 |
| sparse MLA prefill | 40 | 17.3 |
| everything else (elementwise, norms, top-k, dequant) | ~2,400 | ~49 |
| **total** | 3,324 | **295.8** |

Two properties of the machine bound this. The experts are fp4
(`expert_dtype: fp4` in the checkpoint) and the dense weights fp8 block-scaled, and
Ampere has no tensor cores for either, so every GEMM is a bf16 marlin GEMM against a
dequantized copy of the weights -- there is no faster numeric path to switch to. And the
step is GPU-bound rather than launch-bound: replaying it from a CUDA graph (which does
work now, see above) changes nothing, so the 3,324 launches are already fully hidden.

The knobs that were measured and rejected, so they need not be tried again:
`--enable-nccl-nvls`, `NCCL_PROTO=Simple`, `NCCL_ALGO=Tree`, `--disable-custom-all-reduce`
(all within 3 ms), the custom all-reduce in both of its algorithms (two-shot pull
0.306 s, one-shot push 0.356 s, plain NCCL 0.293 s), `--enable-single-batch-overlap`
(no change), `--enable-two-batch-overlap` (rejected by the model: "DeepSeek-V4.1 does
not support two-batch overlap yet"), `--moe-runner-backend triton` (fp4 expert layout)
and, for the record, the fused-mHC path: it is registered only under
`get_platform().is_blackwell` and templated for TP2/TP4 (`_FUSED_MHC_TP_SIZES`), so an
8-rank Ampere deployment cannot reach it. The hyper-connection kernels themselves are
already at their best launch configuration -- sweeping `num_warps` (2/4/8/16),
`num_stages` and slice count (4/8/16/32) around `_hc_mix_stats_partial_kernel` makes it
uniformly worse, and `BLOCK_K=32` buys 12% (2.6 ms per step) only by changing the
reduction order.

The remaining ~36 ms is therefore not a knob. It has to come from overlapping the
all-reduce with compute inside one prefill (the model has no such path today) or from an
fp8 all-reduce (NCCL has no fp8 on Ampere; `--enable-quant-communications` is NPU-only),
which is why the best honest number for this hardware is ~0.291-0.294 s, 13,300-13,370
tok/s, against the 0.2598 s that 15,000 tok/s would need.

#### The collective is not slow. It is measuring rank skew.

The last entry left the all-reduce as the step's largest item and blamed its
bandwidth. Measuring the payloads and the peers says otherwise.

The payloads are real and large: instrumenting both collective paths shows in-place
all-reduce calls of **39.83 MB** (3,890 x 5,120 bf16, the whole hidden state) and
**47.80 MB** (the hyper-connection-expanded width), 43 calls in a step, out-of-place
ones of 1.31/1.98/2.37 MB. So the bytes are as expected -- and the timing is not:

| measurement | 39.8 MB all-reduce |
| --- | ---: |
| raw `dist.all_reduce`, 8 ranks, tight loop | 0.497 ms (80 GB/s) |
| the same collective inside a prefill step | 2.04 ms (20 GB/s) |

The links are not the limit either: NV12 to every peer (12 x 25 GB/s) with CUDA peer
access enabled on all 56 pairs, and a GPU0 -> GPU1 copy of 256 MB sustaining
234 GB/s. `--enable-nccl-nvls`, `NCCL_PROTO=Simple`, `NCCL_ALGO=Tree` and
`--disable-custom-all-reduce` all land within 3 ms of each other, because the
collective is not the thing being measured.

Per rank traces of one step show what is:

| rank | step window | kernels | GPU busy | AR time |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 454.2 ms | 3,324 | 295.8 ms | 6.1 ms |
| 3 | 459.2 ms | 3,307 | 371.9 ms | 68.9 ms |
| 5 | 449.8 ms | 3,307 | 323.9 ms | 84.2 ms |
| 7 | 459.0 ms | 3,323 | 302.6 ms | 3.4 ms |

Every rank finishes at the same wall (~455 ms) but their compute differs by 76 ms, and
the all-reduce kernel duration is almost exactly the complement: the ranks that finish
their layer early sit inside the collective spinning for the ones that have not. A
200 us/rank delay injected before a raw collective moves it from 0.50 to 0.67-1.40 ms,
which is the same mechanism at small scale. **The 87.7 ms attributed to the collective
is rank imbalance being observed at a rendezvous point**, so swapping algorithms,
protocols or NVLS cannot recover it; the per-layer compute skew (~1.9 ms per layer) has
to come down instead.

#### What the step looks like now, and the collective it is left with

A fresh trace of the 3,890-token step after the head trim:

| | calls | GPU ms |
| --- | ---: | ---: |
| all-reduce | 84 | 93.9 |
| marlin MoE | 80 | 63.4 |
| dense marlin | 321 | 41.8 |
| hyper-connection chain | 402 | 36.8 |
| long tail | 2,263 | 34.3 |
| sparse MLA | 43 | 19.7 |
| indexer top-k | 131 | 5.9 |
| **total** | 3,324 | **295.8** |

Sparse MLA fell from 66.6 to 19.7 ms, which is what the head trim bought. The
collective is now the largest single item at 32% of the step.

Two facts about it, both measured rather than assumed:

- **Do not read overlap off a profiled trace.** Summing kernel durations gives
  295.8 ms and so does the union of their intervals across streams, which looks like
  "nothing ever overlaps". It is not: the profiled step window is 454-459 ms while the
  same step measures 295 ms unprofiled, so the profiler is inflating and serializing
  what it records. The per-rank kernel and collective durations in that trace are
  inflated the same way (one rank shows 372 ms of "busy" inside a 306 ms step). Every
  conclusion below that was cross-checked against an unprofiled TTFT is sound -- the
  head trim removed 39 ms of profiled attention time and moved TTFT by 39 ms -- but
  anything that rests only on the trace's overlap or idle figures has to be re-measured
  with the profiler off.
- **Both in-tree implementations are latency algorithms.** The payload is the whole
  hidden state, 3,890 x 5,120 bf16 = 38.8 MB, and it goes through
  `all_reduce_2shot_pull` (or NCCL's `RING_LL`), which are tuned for small
  messages. `--disable-custom-all-reduce` moves it to plain NCCL: prefill
  0.306 -> **0.293 s**, but a decode step goes 23 -> 29.3 ms, because decode's
  payload really is one row (10 KB) and the low-latency kernel is right there.
  So the prefill win is a trade, not a free lunch: the size-aware dispatch that
  should send 38.8 MB to NCCL and 10 KB to the custom kernel is not deciding
  that way here, and fixing it is the next thing worth doing.

Best measured on a 3,890-token single-stream prefill: **0.293 s, 13,272 tok/s**
(with `--disable-custom-all-reduce`), against a 0.2598 s target.

The narrow result has to be padded back afterwards. The piecewise prefill graph
allocates the attention output at capture time with the padded head count and then
copies into it, so returning a narrow tensor trips its element-count assertion
(`134217728 != 16777216`, exactly the 8x). `_pad_prefill_heads` restores the padded
shape with zeros past the real heads -- what those entries held before the trim, zero
queries -- so it stays bit-identical and keeps the 8x saving. It is shape-agnostic on
purpose: the prefill leg returns rank 3 and the decode leg rank 4 with the group
dimension kept, and both put the heads at `-2`.

With that in place `--cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 4096
--cuda-graph-prefill-max-context 4096` captures and replays a 3,890-token prefill
(it could not before: the same head-pad assertion fired during capture). It changes
nothing: 0.296 s against 0.295-0.298 s eager. That is worth knowing precisely because
it was the last open hypothesis -- replacing 3,324 launches with graph replay moves
nothing, so the step is GPU-bound and the launch overhead is already hidden. Any
further progress has to come from doing less GPU work, not from issuing it faster.

#### The sparse path was scoring 8x the heads this rank owns

`q` reaches the attention backend padded to **64** heads, the shape the decode
kernels' head specialization wants. A TP8 rank owns `tp_q_head_num = 8` of them
(`n_local_heads = num_attention_heads // attn_tp_size`), and the in-tree SM100/SM120
legs say so explicitly -- `q[..., : layer.tp_q_head_num, :]`, with the comment "The
64-head pad only serves the decode kernel's h_q specialization; the prefill kernel
takes any h_q, so drop it." The SM80 leg never trimmed, and the portable Triton
kernel accepts whatever head count it is given, so Ampere scored and mixed all 64:
**eight times the attention work this rank is responsible for**, in prefill and in
decode.

Trimming costs one line at the dispatch point. The answer is bit-identical (the same
48-token reply, character for character), which is the proof that the 56 extra heads
were padding and not hidden work.

| 3,890-token prefill | TTFT | tok/s |
| --- | ---: | ---: |
| 64 heads (padded) | 0.331 s | 11,747 |
| 8 heads (owned) | **0.292 s** | **13,326** |

It also corrects an analysis error of mine in the previous entries: with 64 heads the
per-layer attention looked like 326 GFLOP at 63% of the A100 bf16 peak, i.e. "near
optimal, leave it alone". The real figure is eight times smaller, so attention was
never the ceiling -- it was burning a third of the step on padded lanes.

#### 80 device synchronizations per prefill step, for a union nothing needed

A prefill step of 3,890 tokens carried 116 host synchronizations, 180.8 ms of them by
the CPU's own clock: 80 came from `torch.unique` inside `_forward_decode_sparse_sm80`
(two per layer, since prefill reaches that function too) and 32 from the indexer's
`unique_consecutive`. `torch.unique` has to read its output length back from the
device, so each call drains the queue; that is why the step looked neither CPU-bound
nor GPU-bound and sat at a suspiciously stable ~340 ms either way.

The union existed only to make the dequantized KV workspace compact. A slot index can
serve as its own workspace row instead: `SparsePrefillWorkspace` now holds one row per
pool slot, the whole slot range is dequantized once, and `swa_remap = where(valid,
slot, -1)` replaces the inverse permutation. Invalid lanes are dropped exactly as
before, dequantization per slot is unchanged, so the result is bit-identical -- checked
against the legacy path on a fixed prompt, same text, and the bound `max(slot) < pages
x page_size` was asserted for prefill, decode and warmup shapes (26,112 SWA slots,
4,352/8,448 extra slots). The workspace costs a few tens of MB per layer, a small
fraction of the attention that follows it.

| 3,890-token prefill | TTFT | tok/s |
| --- | ---: | ---: |
| union + 116 host syncs | 0.339 s | 11,478 |
| slot-indexed workspace | **0.331 s** | **11,747** |

Worth stating plainly: the 80 syncs cost only 8 ms of wall, not the 120 ms their
duration suggests, because the GPU kept working through most of them. What remains is
that the step is genuinely GPU-bound at ~340 ms, and the target of 259 ms still needs
~72 ms out of kernels.

#### Decode graphs were impossible here, and they cost 6.7x per decode step

`_forward_decode_sparse_sm80` dequantized only the *union* of referenced KV slots,
so it called `torch.unique(..., return_inverse=True)` and sized the workspace from
`swa_slots.numel()`. `torch.unique` syncs the device to learn its output length, so
the decode path could not be captured at all -- which is why every config in this
document disabled decode graphs. A capture now takes the dedup-free route: slots are
dequantized in reference order, where a duplicate slot only costs a redundant row
and invalid lanes are still dropped by the `-1` remap. The eager path keeps the
union, so nothing changes without capture.

Measured on a one-token-per-step decode, `max_new_tokens=1` against `=5`:

| decode | per decode step |
| --- | ---: |
| eager (torch.unique + 96 host syncs) | 154 ms |
| captured | **23 ms** |

Two more numbers came out of the same measurements, and they close the question of
what TTFT actually contains:

- **TTFT does not include a decode step.** Making decode 6.7x faster left TTFT at
  0.338 s for 3,890 tokens, unchanged.
- With the step annotations in a trace, the prefill step's own numbers are
  `step[EXTEND bs=1 toks=3792]`: 3,857 kernels and **367.7 ms of GPU time**, against
  a 0.334 s TTFT. So a single-stream prefill of 3,900 tokens is *exactly* GPU-bound
  (~368 ms of kernel work, and profiling inflates it by about the same margin), and
  the fixed ~169 ms that shows up on short prompts is the launch cost that has no
  compute to hide behind.

That fixes the target precisely: 15,000 tok/s at 3,900 tokens is 259 ms, so ~110 ms
of the 368 has to come out of kernels. The kernel budget of that step, per launch
count and time, is 110 ms of collectives (40 custom `all_reduce_2shot_pull` at
1.83 ms plus 43 NCCL at 0.85, every payload the full hidden state), 67 ms of sparse
MLA prefill, 63 ms of marlin MoE, 46 ms of dense marlin, 25 ms of hyper-connection
stats, and ~30 ms over the tail of small kernels. `--moe-runner-backend triton`
is not an option for this checkpoint: the routed experts are fp4 and that backend
rejects the layout, leaving marlin as the only Ampere GEMM path.

#### Both prefill graph backends now replay, and the answer is GPU time

With the field-list fix below, a V4.1 prefill replays under both backends -- the
log line reads `cuda graph: True` -- provided the step is exactly one bucket wide.
Three costs at the same 4096-token bucket, same server settings:

| prefill path | TTFT (s) | what it says |
| --- | ---: | --- |
| eager, candidate block scheme | ~0.365 | the baseline |
| **breakable replay** | **0.352** | capture is worth 13 ms |
| full replay, full-width indexer | 0.594 | the indexer is worth far more |

`PagedIndexerMetadata.copy_` had a hard-coded field list that the three plan
fields added for the captured scorer were never added to, so a breakable replay
refresh asserted on the field sets before it could replay; that is the fix that
made the middle row measurable.

The middle row is the important one. **Capturing the body of every layer, with
attention left eager, saves 13 ms of a 352 ms step.** So at 3,900-4,096 tokens this
step is not launch-bound and not python-bound: it is GPU-bound, and the ~330 ms of
kernel work is the budget. The 169 ms launch floor measured on short prompts is a
short-prompt phenomenon -- the CPU issue cost overlaps with the GPU once there is
enough compute to hide it behind.

The third row sizes the other term. Running every low-ratio source at full width
instead of through the candidate block table costs **240 ms**, five times what
capture saves, which is why a captured prefill cannot be made faster by capture
alone on this GPU: the candidate scheme is what keeps the indexer off the critical
path, and it must be ported before a captured path can pay.

That leaves the arithmetic of the goal in one line. 15,000 tok/s at 3,900 tokens is
259 ms; the step's GPU work is ~300 ms once the custom all-reduce overlaps; so the
remaining ~40 ms has to come out of kernels -- the sparse MLA prefill (69 ms at
58% of HBM peak, ~28 ms of headroom) and the two marlin MoE GEMMs (65 + 47 ms).

#### The whole-body prefill graph captures now, and what that exposed

Four capture-illegal operations stood between `--cuda-graph-backend-prefill full`
and a captured prefill. All four are fixed and the server starts with the graph
backend on:

- the candidate-block `assert` in `_low_ratio_index_topk_graph`: a captured
  forward now scores the full width at every low-ratio source instead of
  publishing a DeepGEMM fp4 block table;
- `req_to_token[row, cols]` with a zero-dim row index, which is a capture-illegal
  advanced index -- replaced by two `index_select` gathers;
- `DSV4_DEQUANT_FP4_TABLE.to(device)` inside the index-K readback, a host to device
  copy on every call; the codebook is now cached per device;
- the eager 1 GiB score budget in `prefill_graph_scores`, which reserved tens of
  megabytes per layer in the graph's private pool; a captured forward uses 64 MiB.

The graph then turned out to be **unreachable for a real prompt**, for a reason
worth writing down: `can_replay_locally` refuses any V4.1 batch whose
`padded_num_tokens != num_tokens`, so a replay needs a step of *exactly* one bucket
size. With a 4096 bucket and `--context-length 4096` a 4096-token prompt cannot even
be sent (no room for the output token), and the log confirms every real request ran
`cuda graph: False`. Raising the context to make one reachable moves the collision
to memory: capturing a 4096-token bucket leaves the KV pool `0.16 GB` at
`--mem-fraction-static 0.93`, and the run dies in pool construction.

Two measurements came out of the attempt, and the second is the important one.

- Steps that do not replay still take the graph-shaped code path under
  `--cuda-graph-backend-prefill full`, so they pay the full-width scorer for
  nothing: **0.489 s at 3,897 tokens against 0.348 s on the baseline config**. The
  full-graph backend as it stands is a 40% loss, not a win.
- That 0.489 versus 0.348 is the cleanest measurement of one thing yet: **the
  candidate block scheme is worth ~150 ms of a 3,900-token step.** The indexer's
  full-width top-k over 4,096 compressed positions per layer costs more than the
  launch overhead the graph removes. Until `publish_prefill_table` has a torch
  port on Ampere, a captured prefill cannot be faster than the eager candidate
  path, and the graph is not the way to 15,000 tok/s on its own.

#### The step is paced by launch count, not by any kernel (measured)

A TTFT sweep settles what the profile could not. Sweeping prompt length on the
best config:

| prompt tokens | 4 | 70 | 534 | 1424 | 2472 | 3589 | 3897 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT (s) | 0.169 | 0.171 | 0.176 | 0.184 | 0.204 | 0.332 | 0.348 |

TTFT is **flat at 169 ms until about 1,500 tokens**, then grows. A four-token
request and a 1,400-token request cost the same, so that 169 ms is not prefill
compute: it is the fixed cost of a step. Profiling a five-token request names it
exactly -- **28 ms of GPU work across 3,620 kernel launches**, essentially the same
launch count as a 3,853-token step (3,890). Per layer that is **90.5 launches**,
and the wall cost per launch is **47 us**, which is far more than a launch costs;
the trace holds 145,223 python events, i.e. **40 python calls per kernel**. The
critical path is python dispatch per kernel, not bytes or FLOPs.

Above the flat region the marginal cost is ~46 us/token, so 3,897 tokens is
169 (launches) + 179 (compute) = 348 ms, and the two are additive: the CPU issues
a layer, the GPU drains it, repeat.

This single model explains three nulls that looked unrelated:

- removing 78 ms of GPU busy (custom all-reduce -> NCCL: 379.4 -> 301.4 ms of
  kernel time) changed TTFT by **0.000 s**;
- `ar`, `rs`, `rs+rsv` and `--enable-fused-moe-sum-all-reduce` all land within
  noise -- the GPU has slack to absorb any of them;
- `--cuda-graph-backend-prefill breakable` is **exactly** the baseline (0.346 s),
  because it does not reduce the launch count.

It also retires two candidate explanations. Decode graphs cannot rescue the first
token: this backend's decode path calls `torch.unique` in
`_forward_decode_sparse_sm80`, so a decode graph cannot be captured at all, and a
`max_new_tokens=1` request needs no decode forward anyway (the first token is
sampled from the prefill's logits). Slow tokenization is not it either: the native
`/generate` route with pre-tokenized input ids measures the same 0.341-0.348 s
once every request carries unique ids and the log confirms `#cached-token: 0`.

The arithmetic that follows: 15,000 tok/s at 3,900 tokens is 259 ms, so the fixed
169 ms has to fall to ~80 ms, i.e. **the launch count has to roughly halve** while
the 179 ms of compute stays. With 47 us per launch that is ~45 launches per layer
instead of 90.5. The two largest groups in the histogram are `fill_`
(16.3 per layer) and the hyper-connection family -- `hc_mix_stats_partial`,
`hc_mix_stats_reduce`, `hc_split_sinkhorn`, `mhc_post_tilelang`, `mhc_combine` --
at 2.0 each, 10 per layer together. Removing the redundant fills and fusing the
hyper-connection chain is worth ~26 launches per layer, ~49 ms, without touching a
single kernel's internals.

Avenues measured and closed, so they are not re-tried:

- `--boundary-reduction ar` instead of `rs+rsv`: 0.343-0.346 s against 0.349,
  the best single-stream config here (11,332 tok/s), and the only knob that
  moved.
- `--enable-torch-compile`: nothing. It binds `torch.compile` in
  `model_executor/cpu_graph_runner.py`, i.e. the CPU-graph mode, not this path.
- `--enable-prefill-cp --cp-strategy interleave`: resolves `attn_cp_size` to
  the whole TP group, which sets `moe_dense_tp_size=1` -- the dense weights
  replicate and the load OOMs by 720 MiB even at `--mem-fraction-static 0.86`.
  It also needs `language_model_only` because the checkpoint ships a vision
  tower that rejects CP outright ("V4.1 vision supports TP/EP/DP without CP or
  PP").
- `--enable-prefill-delayer`: its queue trigger needs requests to arrive while
  a step runs, which is exactly what a 350 ms step prevents.
- Prefill CUDA graphs: `breakable` breaks at attention boundaries and rebuilds
  attention metadata at replay, so attention -- the largest Python block -- is
  never inside a graph, which is why capture measured 0.330 s against 0.332.

Faster collectives cannot help; the collectives are 1.5 ms of the 358. The
106 ms is dispatch, so the levers are, in order:

1. Fewer Python ops per layer. The three-step trace logs 875,920
   `python_function` and 58,222 `cpu_op` events, ~485 CPU ops per layer, all
   issued from one saturated thread per rank.

   The indexer is not one of the fat ones. Its scorer
   (`dsv41_sparse.scores`, `einsum("bhd,nd->bhn")` then relu, weight, head sum)
   materialises `[tokens, heads, candidates]` — 0.48 GiB at a 4k prefill — and
   is 3.2 ms per call in isolation, but the whole low-ratio scoring path is
   only ~4 ms of a 358 ms step: fusing it (`kernels/ops/attention/dsv4/
   sparse_scores.py`, one `tl.dot` per head with the head sum in fp32) is 4.5x
   faster and far more accurate (2.8e-6 against an fp32 reference where the
   bf16 einsum is at 3.7e-2, with 99.8% top-512 overlap on the selection), and
   moves TTFT by 4 ms. Worth having, not where the step goes.
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
