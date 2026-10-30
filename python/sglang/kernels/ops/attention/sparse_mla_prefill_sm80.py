"""Portable Triton sparse-MLA prefill for pre-SM90 GPUs (e.g. A100 / SM80).

``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` is the production sparse prefill
kernel for DeepSeek-V4, but the shipped fatbins only contain ``sm_90a`` and
``sm_100`` images, and the kernel itself leans on TMA / wgmma. GPUs without
those instructions therefore cannot run DeepSeek-V4 prefill at all.

This module provides a numerically equivalent path built from ``tl.dot`` on
bf16 tiles, so it runs on any Triton target including SM80. It consumes exactly the same inputs
``flash_mla_sparse_fwd`` does:

* ``q``     -- ``[num_tokens, num_heads, 512]`` bf16 query rows
* ``kv``    -- ``[num_kv_tokens, 512]`` bf16, already dequantized by the caller
  (DeepSeek-V4 stores one shared K/V row per token, so K and V are the same
  matrix and only the first ``d_v`` columns participate in the V product)
* ``indices`` -- ``[num_tokens, 1, topk]`` int32 row indices into ``kv``
* ``topk_length`` -- optional ``[num_tokens]`` int32 valid lengths

Each program covers a block of query heads for one token. DeepSeek-V4's MLA
shares one K/V row across all heads, so a token's scores are a plain
``[BLOCK_H, D] @ [TOPK, D]^T`` and the MMA units do the work; a per-head
program would only produce 512-wide dot products no tensor-core shape can
consume. Two passes over the KV blocks -- one for the softmax maximum, one for
the normalizer and the value product -- keep the accumulator free of the
per-block rescale that an online softmax needs.

Measured on 8x A100-SXM4-80GB (TP8), single stream, no prefix-cache hits:
about 3.3k tok/s at 600-token prompts rising to 8.2k tok/s at 4k, versus
194 tok/s for the earlier scalar-reduction version. Retuning the tiles for the
8 heads a TP8 rank actually owns (see ``_pick_block_h``) took the kernel from
3.66 ms to 2.03 ms per layer at 4k tokens and moved the same measurement to
9.2k tok/s.

Even after that, one layer of this kernel sustains ~17 TFLOPS, about 5% of the
A100 bf16 peak, and reads KV at ~580 GB/s, about 28% of HBM. It is neither
compute nor bandwidth bound: the two passes double the gathered KV loads, each
``[BLOCK_KV, D]`` tile arrives through an index list so it cannot use one
contiguous copy, and the accumulator (BLOCK_H x 512 fp32) is large enough to
keep occupancy at one program per SM. Those, not the MMA count, are what a
faster version would have to attack.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _sparse_mla_prefill_kernel(
    Q,  # [num_tokens, H, D] bf16
    KV,  # [num_kv_tokens, D] bf16
    INDICES,  # [num_tokens, topk] int32
    TOPK_LEN,  # [num_tokens] int32 or dummy: live length of the head segment
    PREFIX_LEN,  # [num_tokens] int32 or dummy: extra upper bound on kv_len
    OUT,  # [num_tokens, H, DV] bf16
    SINK,  # [H] float32 or dummy
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kvt,
    stride_it,
    stride_ot,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_TOPK_LEN: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_PREFIX_LEN: tl.constexpr,
):
    """One program handles ``BLOCK_H`` query heads of a single token.

    DeepSeek-V4's MLA shares one K/V row across every head, so the score matrix
    for a token is just ``q[BLOCK_H, D] @ kv[TOPK, D]^T``. Batching the heads
    into one tile is what lets this run on tensor cores; a per-head program
    would be a stream of 512-wide dot products no MMA shape can consume.

    Two passes over the KV blocks: the first finds the softmax maximum, the
    second accumulates the normalizer and the value product. Keeping the max
    fixed between passes removes the per-block rescale of the accumulator,
    which cannot be folded into an MMA result without losing precision.
    """
    token = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    offs_dv = tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < DV

    q = tl.load(
        Q + token * stride_qt + offs_h[:, None] * stride_qh + offs_d[None, :],
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # ``INDICES`` holds the SWA window followed by the compressed top-k; the
    # per-row live length comes from TOPK_LEN, optionally capped by PREFIX_LEN.
    kv_len = TOPK
    if HAS_TOPK_LEN:
        kv_len = tl.load(TOPK_LEN + token)
    if HAS_PREFIX_LEN:
        kv_len = tl.minimum(tl.load(PREFIX_LEN + token), kv_len)
    kv_len = tl.minimum(kv_len, TOPK)

    # q stays bf16 for the MMA and the scale is applied to each score tile:
    # pre-scaling q would round the scale itself into bf16.

    # Pass 1: running maximum per head.
    m_i = tl.full((BLOCK_H,), -1.0e30, dtype=tl.float32)
    for start in range(0, TOPK, BLOCK_KV):
        offs_kv = start + tl.arange(0, BLOCK_KV)
        in_range = offs_kv < kv_len
        idx_raw = tl.load(
            INDICES + token * stride_it + offs_kv, mask=in_range, other=-1
        )
        # -1 is the "no KV here" sentinel; it must not alias row 0.
        valid = in_range & (idx_raw >= 0)
        idx = tl.where(valid, idx_raw, 0).to(tl.int64)
        kv_tile = tl.load(
            KV + idx[:, None] * stride_kvt + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(kv_tile), out_dtype=tl.float32)
        scores = scores * (sm_scale * LOG2E)
        # Invalid lanes are pushed to a finite floor rather than -inf so a
        # fully masked block still leaves the maximum usable.
        scores = tl.where(valid[None, :], scores, -1.0e30)
        m_i = tl.maximum(m_i, tl.max(scores, axis=1))

    # Pass 2: normalizer and the P@V product.
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    for start in range(0, TOPK, BLOCK_KV):
        offs_kv = start + tl.arange(0, BLOCK_KV)
        in_range = offs_kv < kv_len
        idx_raw = tl.load(
            INDICES + token * stride_it + offs_kv, mask=in_range, other=-1
        )
        valid = in_range & (idx_raw >= 0)
        idx = tl.where(valid, idx_raw, 0).to(tl.int64)
        kv_tile = tl.load(
            KV + idx[:, None] * stride_kvt + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(kv_tile), out_dtype=tl.float32)
        scores = scores * (sm_scale * LOG2E)
        scores = tl.where(valid[None, :], scores, -1.0e30)
        p = tl.exp2(scores - m_i[:, None])
        l_i += tl.sum(p, axis=1)
        # K and V are the same row. The product spans the whole head width and
        # the store below keeps the first DV columns; DV == D for DeepSeek-V4,
        # and Triton cannot slice a tensor dimension to a constexpr.
        acc += tl.dot(p.to(tl.bfloat16), kv_tile, out_dtype=tl.float32)

    # Attention sink: an always-present logit that only moves the normalizer,
    # matching FlashMLA's sink semantics.
    if HAS_SINK:
        sink = tl.load(SINK + offs_h, mask=h_mask, other=-1.0e30)
        sink = sink * LOG2E
        m_final = tl.maximum(m_i, sink)
        l_i = l_i * tl.exp2(m_i - m_final) + tl.exp2(sink - m_final)
        acc = acc * tl.exp2(m_i - m_final)[:, None]

    # A row with no live slot has l_i == 0; emit zeros instead of NaN so a
    # request whose window has not been written yet cannot poison the layer.
    out = tl.where(l_i[:, None] > 0.0, acc / tl.maximum(l_i, 1e-30)[:, None], 0.0)
    if DV == D:
        store_offs = offs_d
        store_mask = d_mask
    else:
        store_offs = offs_dv
        store_mask = dv_mask
    tl.store(
        OUT + token * stride_ot + offs_h[:, None] * stride_oh + store_offs[None, :],
        out.to(tl.bfloat16),
        mask=h_mask[:, None] & store_mask[None, :],
    )


def _pick_block_kv(topk: int) -> int:
    """KV block height.

    The tile is ``[BLOCK_KV, D]`` bf16 with D=512, so 32 rows are 32 KiB of
    shared memory per staged copy and 64 rows are 64 KiB (128 does not fit
    alongside the MMA operands on a 164 KiB part). 32 wins at every head count
    measured on A100: it leaves room for three pipeline stages, and the extra
    loop iterations cost less than the occupancy the smaller tile buys.
    """
    return 32


def _pick_block_h(num_heads: int) -> int:
    """Query-head block height.

    MMA needs at least 16 rows, and beyond 32 the accumulator tile (BLOCK_H x
    512 fp32) starts crowding out the KV staging buffer. A tensor-parallel rank
    sees only ``num_heads / tp`` of them (8 at TP8), and padding those 8 up to
    16 costs more than the wasted MMA lanes: measured 2.01 ms vs 3.62 ms for
    the whole layer at 8 heads on A100, because the narrower accumulator keeps
    more programs resident instead of spilling.
    """
    if num_heads >= 32:
        return 32
    if num_heads >= 16:
        return 16
    return 8


def flash_mla_sparse_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    prefix_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, None, None]:
    """Drop-in SM80 stand-in for ``sgl_kernel.flash_mla.flash_mla_sparse_fwd``.

    Returns ``(out, None, None)`` to match the three-tuple contract of the
    FlashMLA entry point (the extra members are SM90 kernel metadata that has
    no meaning here).
    """
    assert q.dim() == 3, f"q must be [tokens, heads, dim], got {q.shape}"
    # kv may arrive as [tokens, dim] or [tokens, 1, dim] (the workspace is
    # allocated with the kernel's 1-head shape); squeeze the singleton.
    if kv.dim() == 3 and kv.shape[1] == 1:
        kv = kv.squeeze(1)
    assert kv.dim() == 2, f"kv must be [tokens, dim], got {kv.shape}"
    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()

    num_tokens, num_heads, head_dim = q.shape
    # indices arrives as [tokens, 1, topk]; flatten the trailing dims.
    flat_indices = indices.reshape(num_tokens, -1).contiguous()
    topk = flat_indices.shape[1]

    out = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
    )

    sink = (
        attn_sink.reshape(-1).to(torch.float32).contiguous()
        if attn_sink is not None
        else torch.empty(0, dtype=torch.float32, device=q.device)
    )
    lengths = (
        topk_length.to(torch.int32).contiguous()
        if topk_length is not None
        else torch.empty(0, dtype=torch.int32, device=q.device)
    )
    prefixes = (
        prefix_length.to(torch.int32).contiguous()
        if prefix_length is not None
        else torch.empty(0, dtype=torch.int32, device=q.device)
    )

    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(d_v)
    block_h = _pick_block_h(num_heads)
    _sparse_mla_prefill_kernel[(num_tokens, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        flat_indices,
        lengths,
        prefixes,
        out,
        sink,
        sm_scale,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        flat_indices.stride(0),
        out.stride(0),
        out.stride(1),
        H=num_heads,
        D=head_dim,
        DV=d_v,
        TOPK=topk,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_KV=_pick_block_kv(topk),
        BLOCK_H=block_h,
        HAS_TOPK_LEN=topk_length is not None,
        HAS_PREFIX_LEN=prefix_length is not None,
        HAS_SINK=attn_sink is not None,
    )
    return out, None, None
