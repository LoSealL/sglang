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

The absolute figures only hold with `#cached-token: 0`; repeated prompts reuse
the prefix cache and inflate the number several-fold. Verify against the
server's own `Prefill batch ... input throughput` line before quoting any
prefill measurement.

## Tests

```bash
python -m pytest test/registered/kernels/ops/attention/dsv4/test_sparse_mla_sm80.py
```

Covers agreement with an fp32 reference, `-1` sentinel handling, the
all-invalid row, the attention sink, `topk_length` prefix truncation, and the
paged-view flattening that decode depends on (with cross-page slot ids, since
page-0-confined ids hide a wrong page stride).
