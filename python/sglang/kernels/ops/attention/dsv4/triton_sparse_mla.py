"""Gather-free Triton sparse MLA for DeepSeek V4 on sm_80, reading sglang's
packed paged KV (448 fp8 nope bytes + 64 bf16 rope bytes per token, ue8m0
scales at the page tail). Attention sinks enter the softmax as one extra
zero-value logit per head (Gemma-style); -inf means no sink.

Ported from vllm/v1/attention/ops/triton_sparse_mla.py; the row loader is
replaced with on-the-fly dequant of the packed layout (fp8 bytes are
software-decoded — Triton rejects fp8e4nv pointers on sm_80)."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    TILE_SIZE,
)
from sglang.kernels.ops.attention.dsv4.triton_fp8_mqa_logits_cuda import (
    _fp8e4m3_to_f32,
)

# Layout constants visible inside the jit kernels (Triton only admits
# tl.constexpr-instantiated globals).
_DIM_NOPE = tl.constexpr(DIM_NOPE)
_NOPE_ROPE_BYTES = tl.constexpr(NOPE_ROPE_BYTES)
_PADDED_SCALE_PER_TOKEN = tl.constexpr(PADDED_SCALE_PER_TOKEN)
_TILE_SIZE = tl.constexpr(TILE_SIZE)


@triton.jit
def _attend_packed(
    CacheU8_ptr,
    CacheBf16_ptr,
    idx_ptr,
    lens_ptr,
    t,
    stride_idx,
    bytes_per_page,
    slots_per_page,
    sm_scale,
    q,
    mask_h,
    m,
    denom,
    acc,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    length = tl.load(lens_ptr + t)
    offs_d = tl.arange(0, HEAD_DIM)
    nope_mask = offs_d < _DIM_NOPE
    for start_n in tl.range(0, length, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < length
        # Index lists carry global slot ids (-1 padded); slot =
        # page * slots_per_page + in-page offset.
        idx = tl.load(idx_ptr + t * stride_idx + offs_n, mask=n_mask, other=-1).to(
            tl.int64
        )
        valid = (idx >= 0) & n_mask
        safe = tl.maximum(idx, 0)
        page = safe // slots_per_page
        off = safe % slots_per_page
        page_base = page * bytes_per_page
        data_base = page_base + off * _NOPE_ROPE_BYTES
        scale_base = (
            page_base
            + slots_per_page * _NOPE_ROPE_BYTES
            + off * _PADDED_SCALE_PER_TOKEN
        )

        # Build the [BLOCK_N, HEAD_DIM] bf16 kv block from the packed layout.
        # Triton cannot assign block slices, so both halves are loaded
        # HEAD_DIM-wide, each masked to its own columns, and merged with a
        # plain elementwise tl.where. The nope half loads raw fp8 bytes and
        # software-decodes + applies the per-64-column ue8m0 scale; the rope
        # half loads bf16 through a second, bf16-typed view of the buffer.
        fmask = valid[:, None] & nope_mask[None, :]
        fp8b = tl.load(
            CacheU8_ptr + data_base[:, None] + offs_d[None, :],
            mask=fmask,
            other=0,
        )
        # ponytail: 2D scale gather (offs_d // TILE_SIZE) beats a
        # [BLOCK_N,8]-load + broadcast_to + reshape (measured 10x slower —
        # the reshape forces a shared-memory layout conversion).
        scale = tl.load(
            CacheU8_ptr + scale_base[:, None] + offs_d[None, :] // _TILE_SIZE,
            mask=fmask,
            other=127,
        ).to(tl.int32)
        nope = _fp8e4m3_to_f32(fp8b) * tl.exp2((scale - 127).to(tl.float32))
        rope = tl.load(
            CacheBf16_ptr
            + (data_base[:, None] + _DIM_NOPE) // 2
            + (offs_d[None, :] - _DIM_NOPE),
            mask=valid[:, None] & (offs_d[None, :] >= _DIM_NOPE),
            other=0.0,
        )
        kv = tl.where(nope_mask[None, :], nope.to(tl.bfloat16), rope)

        qk = tl.dot(q, tl.trans(kv)) * sm_scale
        qk = tl.where(mask_h[:, None] & valid[None, :], qk, float("-inf"))
        n_e_max = tl.maximum(tl.max(qk, 1), m)
        re_scale = tl.exp(m - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc = acc * re_scale[:, None] + tl.dot(p.to(kv.dtype), kv).to(tl.float32)
        denom = denom * re_scale + tl.sum(tl.where(valid, p, 0.0), 1)
        m = n_e_max
    return m, denom, acc


@triton.jit
def _sparse_mla_fwd_kernel(
    Q_ptr,
    Out_ptr,
    SwaU8_ptr,
    SwaBf16_ptr,
    SwaIdx_ptr,
    SwaLens_ptr,
    CompU8_ptr,
    CompBf16_ptr,
    CompIdx_ptr,
    CompLens_ptr,
    Sink_ptr,
    sm_scale,
    stride_q_t,
    stride_q_h,
    stride_o_t,
    stride_o_h,
    stride_swaidx_t,
    stride_compidx_t,
    swa_bytes_per_page,
    swa_slots_per_page,
    comp_bytes_per_page,
    comp_slots_per_page,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_COMP: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_h = offs_h < H

    q = tl.load(
        Q_ptr + t * stride_q_t + offs_h[:, None] * stride_q_h + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=float("-inf"))
    m = sink  # running max (incl. sink logit)
    denom = tl.where(sink > float("-inf"), 1.0, 0.0)
    acc = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)

    m, denom, acc = _attend_packed(
        SwaU8_ptr,
        SwaBf16_ptr,
        SwaIdx_ptr,
        SwaLens_ptr,
        t,
        stride_swaidx_t,
        swa_bytes_per_page,
        swa_slots_per_page,
        sm_scale,
        q,
        mask_h,
        m,
        denom,
        acc,
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
    )
    if HAS_COMP:
        m, denom, acc = _attend_packed(
            CompU8_ptr,
            CompBf16_ptr,
            CompIdx_ptr,
            CompLens_ptr,
            t,
            stride_compidx_t,
            comp_bytes_per_page,
            comp_slots_per_page,
            sm_scale,
            q,
            mask_h,
            m,
            denom,
            acc,
            HEAD_DIM=HEAD_DIM,
            BLOCK_N=BLOCK_N,
        )

    denom = tl.maximum(denom, 1e-20)
    out = (acc / denom[:, None]).to(Out_ptr.dtype.element_ty)
    tl.store(
        Out_ptr + t * stride_o_t + offs_h[:, None] * stride_o_h + offs_d[None, :],
        out,
        mask=mask_h[:, None],
    )


def triton_sparse_mla_fwd(
    q: torch.Tensor,
    out: torch.Tensor,
    swa_cache_u8: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    comp_cache_u8: torch.Tensor | None,
    comp_indices: torch.Tensor | None,
    comp_lens: torch.Tensor | None,
    sm_scale: float,
    sink: torch.Tensor,
    *,
    swa_page: int,
    comp_page: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over index-selected rows of the packed paged KV.

    q/out: [T, H, 512] bf16. swa_cache_u8/comp_cache_u8: [num_pages,
    bytes_per_page_padded] uint8 pools in the DSV4 packed layout (see
    dequant_k_cache). swa_indices/comp_indices: [T, W] int32 with global slot
    ids, -1 padded; swa_lens/comp_lens: [T] int32 segment lengths.
    comp_cache_u8=None selects the SWA-only single-segment specialization.
    Writes and returns ``out``.
    """
    T, H, head_dim = q.shape
    assert head_dim == 512, f"expected head_dim 512, got {head_dim}"
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert swa_cache_u8.dtype == torch.uint8 and swa_cache_u8.is_contiguous()
    swa_bf16 = swa_cache_u8.view(torch.bfloat16)
    has_comp = comp_cache_u8 is not None
    if has_comp:
        assert comp_cache_u8.dtype == torch.uint8 and comp_cache_u8.is_contiguous()
        comp_bf16 = comp_cache_u8.view(torch.bfloat16)
    else:
        # Dummy 1-page tensors keep the kernel signature uniform; the comp
        # segment is dead code under HAS_COMP=False.
        comp_cache_u8, comp_bf16 = swa_cache_u8[:1], swa_bf16[:1]
        comp_indices, comp_lens = swa_indices[:1], swa_lens[:1]
        comp_page = swa_page

    BLOCK_H, BLOCK_N = 16, 64
    grid = (T, triton.cdiv(H, BLOCK_H))
    _sparse_mla_fwd_kernel[grid](
        q,
        out,
        swa_cache_u8,
        swa_bf16,
        swa_indices,
        swa_lens,
        comp_cache_u8,
        comp_bf16,
        comp_indices,
        comp_lens,
        sink,
        sm_scale,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        swa_indices.stride(0),
        comp_indices.stride(0),
        swa_cache_u8.shape[-1],
        swa_page,
        comp_cache_u8.shape[-1],
        comp_page,
        H=H,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        HAS_COMP=has_comp,
        num_warps=8,
        # num_stages=1: default pipelining (3) triples the [BLOCK_N,512]
        # load buffers and exceeds sm80 shared memory (246KB > 167KB).
        num_stages=1,
    )
    return out
