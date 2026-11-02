"""SM80 (A100) fallback for the DSA k-pool indexer MQA-logits kernels.

DeepGEMM (and the TileLang FP8 kernel) have no SM80 support: A100 lacks FP8
tensor cores. These helpers replicate the DeepGEMM contract by dequantizing
the FP8 activations/keys to bf16 and running the per-head logits with cuBLAS:

    logits[b, j] = k_scale[j] * sum_h weights[b, h] * relu(dot(q[b, h], k[j]))

(the per-head relu follows the reference TileLang kernel in
sglang/kernels/ops/attention/dsa/tilelang_kernel.py::fp8_paged_mqa_logits).

Index-cache page layout (per kpool_fp8_index writers and the TileLang
reader): [page_size x 128B fp8 data | page_size x 4B fp32 scales] — the
(p, 64, 1, 132) view used by DeepGEMM is only a page-stride view, NOT
per-row interleaved.

Not fast, but correct; SM80 bring-up only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def sm80_fp8_mqa_logits(
    q_fp8: torch.Tensor,  # (m, h, d) float8_e4m3fn
    kv: Tuple[torch.Tensor, torch.Tensor],  # (k_fp8 (n, d) fp8, k_sf (n,) fp32)
    weights: torch.Tensor,  # (m, h) fp32
    ks: torch.Tensor,  # (m,) int32
    ke: torch.Tensor,  # (m,) int32
    clean_logits: bool = True,
    max_seqlen_k: int = 0,
) -> torch.Tensor:  # (m, max(ke - ks)) fp32; local j == kv[ks[m] + j], -inf outside
    k_fp8, k_sf = kv
    m = q_fp8.shape[0]
    n = k_fp8.shape[0]
    widths = (ke - ks).to(torch.int64)
    width = int(widths.max().item()) if m > 0 else 0
    width = max(width, 1)
    out = q_fp8.new_full((m, width), float("-inf"), dtype=torch.float32)
    if n == 0:
        return out
    device = q_fp8.device
    q = q_fp8.to(torch.bfloat16)
    k_bf = k_fp8.to(torch.bfloat16)
    col = torch.arange(width, device=device)
    # ponytail: per-head loop + small row chunks keep the (m, h, n) dots tile
    # off-memory at 1M ctx (n = 262k pooled rows); peak transient ~1 GiB.
    for row in range(0, m, 512):
        end = min(row + 512, m)
        acc = torch.zeros(end - row, n, dtype=torch.float32, device=device)
        for hi in range(q.shape[1]):
            dots = torch.matmul(q[row:end, hi], k_bf.t())  # (c, n) bf16
            dots = torch.relu(dots.float()) * weights[row:end, hi, None]
            acc += dots
        logits = acc * k_sf[None, :]
        gather_idx = (ks[row:end, None].to(torch.int64) + col[None, :]).clamp(max=n - 1)
        local = logits.gather(1, gather_idx)
        valid = col[None, :] < widths[row:end, None]
        out[row:end] = torch.where(valid, local, torch.full_like(local, float("-inf")))
    return out


@triton.jit
def _u8_to_bf16_e4m3(u):
    """Exact decode of finite float8_e4m3fn from uint8 bits via fp16.

    ponytail: SM80 triton cannot touch fp8e4nv pointers at all, so fp8
    tensors are viewed as uint8 before launch and decoded in-kernel.
    """
    h16 = ((u.to(tl.uint16) & 0x80) << 8) | ((u.to(tl.uint16) & 0x7F) << 7)
    return (tl.cast(h16, tl.float16, bitcast=True) * 256.0).to(tl.bfloat16)


@triton.jit
def _sm80_paged_mqa_logits_kernel(
    q_u8_ptr,  # (b, h, d) uint8 view of fp8 q
    w_ptr,  # (b, h) fp32
    buf_u8_ptr,  # index cache, uint8, page = [ps*d data | ps*4 scales]
    buf_f32_ptr,  # same buffer viewed fp32
    bt_ptr,  # (b, max_pages) pooled block table
    ctx_ptr,  # (b,) int32 pooled context lens
    out_ptr,  # (b, L) fp32
    stride_bt,
    L,
    total_blocks,  # b * JB_MAX, grid-stride loop bound
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    SCALE_OFF: tl.constexpr,  # PAGE * D bytes
    BLOCK_J: tl.constexpr,
    JB_MAX: tl.constexpr,  # cdiv(L, BLOCK_J)
):
    # ponytail: persistent grid — decode cuda graphs bake the capture-time
    # (max) context width; the grid-stride loop skips work beyond each row's
    # runtime ctx so replay cost scales with the real context.
    wid = tl.program_id(0)
    num_p = tl.num_programs(0)
    for it in range(wid, total_blocks, num_p):
        b = it // JB_MAX
        jb = it % JB_MAX
        offs_j = jb * BLOCK_J + tl.arange(0, BLOCK_J)
        offs_h = tl.arange(0, H)
        offs_d = tl.arange(0, D)

        ctx = tl.load(ctx_ptr + b)
        if jb * BLOCK_J < ctx:
            mask_j = offs_j < ctx
            L_mask = offs_j < L
            safe_j = tl.where(L_mask, offs_j, 0)
            pg = tl.load(bt_ptr + b * stride_bt + safe_j // PAGE).to(tl.int64)
            row = safe_j % PAGE

            q_u8 = tl.load(q_u8_ptr + b * H * D + offs_h[:, None] * D + offs_d[None, :])
            q = _u8_to_bf16_e4m3(q_u8)  # (H, D)
            k_u8 = tl.load(
                buf_u8_ptr
                + pg[:, None] * PAGE_BYTES
                + row[:, None] * D
                + offs_d[None, :],
                mask=L_mask[:, None],
                other=0,
            )  # (BJ, D)
            k = _u8_to_bf16_e4m3(k_u8)
            k_sf = tl.load(
                buf_f32_ptr + (pg * PAGE_BYTES + SCALE_OFF) // 4 + row,
                mask=L_mask,
                other=0.0,
            )

            dots = tl.dot(q, tl.trans(k))  # (H, BJ) fp32 acc
            dots = tl.where(dots > 0, dots, 0.0)
            w = tl.load(w_ptr + b * H + offs_h)
            logits = tl.sum(dots * w[:, None], axis=0) * k_sf
            tl.store(
                out_ptr + b * L + offs_j,
                tl.where(mask_j, logits, float("-inf")),
                mask=L_mask,
            )


def sm80_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # (b, 1, h, d) float8_e4m3fn
    kv_cache_fp8: torch.Tensor,  # (pages, page, 1, d+4) fp8/uint8 page-stride view
    weights: torch.Tensor,  # (b, h) fp32
    context_lens: torch.Tensor,  # (b,) int32 (pooled lengths)
    block_tables: torch.Tensor,  # (b, pages) int
    schedule_meta,  # unused
    max_context_len: int,
    clean_logits: bool = False,
    indices=None,
) -> (
    torch.Tensor
):  # (b, max_context_len) fp32; beyond ctx is garbage (masked downstream)
    b, _, h, d = q_fp8.shape
    page_size = kv_cache_fp8.shape[1]
    context_lens = context_lens.reshape(-1)  # deep_gemm passes (b, 1)
    block_tables = block_tables.reshape(block_tables.shape[0], -1)
    out = torch.empty((b, max_context_len), dtype=torch.float32, device=q_fp8.device)
    if max_context_len == 0 or b == 0:
        return out
    assert h == 32 and d == 128, "kernel assumes the GLM indexer geometry"
    buf_u8 = (
        kv_cache_fp8.view(torch.uint8)
        if kv_cache_fp8.dtype != torch.uint8
        else kv_cache_fp8
    ).view(kv_cache_fp8.shape[0], -1)
    page_bytes = buf_u8.shape[1]
    jb_max = triton.cdiv(max_context_len, 64)
    total = b * jb_max
    grid = (min(2 * 108, total),)
    _sm80_paged_mqa_logits_kernel[grid](
        q_fp8.reshape(b, h, d).view(torch.uint8),
        weights,
        buf_u8,
        buf_u8.view(torch.float32),
        block_tables,
        context_lens,
        out,
        block_tables.stride(0),
        max_context_len,
        total,
        H=h,
        D=d,
        PAGE=page_size,
        PAGE_BYTES=page_bytes,
        SCALE_OFF=page_size * d,
        BLOCK_J=64,
        JB_MAX=jb_max,
    )
    return out
