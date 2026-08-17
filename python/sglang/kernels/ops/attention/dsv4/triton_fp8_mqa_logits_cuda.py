# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fp8 MQA logits for the DeepSeek V4 lightning indexer on sm_80.

Replaces DeepGEMM's ``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits``.
Math (must match ``triton_fp8_mqa_logits.py``):
logits[m, n] = sum_h relu(dot(q[m, h], k[n]) * kv_scale[n]) * w[m, h];
the FP8 Q per-token scale is already folded into ``weights`` by the caller.
A100 has no fp8 tensor cores and Triton rejects fp8e4nv pointers on sm_80,
so fp8 bytes are loaded as uint8, decoded manually, and dotted as bf16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8e4m3_to_f32(x):
    # Decode e4m3fn bits: normal = (1 + m/8) * 2^(e-7), subnormal = m * 2^-9.
    # ponytail: NaN (e=15, m=7) decodes to 480; quantized fp8 has no NaNs.
    u = x.to(tl.uint32)
    e = (u >> 3) & 15
    m = u & 7
    norm = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    s = tl.where((u & 0x80) != 0, -1.0, 1.0)
    return tl.where(e == 0, sub, norm) * s


@triton.jit
def _mqa_logits_inner(q_bf16, kv_bf16, kv_scales, w, BLOCK_KV: tl.constexpr):
    # q_bf16 [H_PAD, D]; kv_bf16 [D, BLOCK_KV]; kv_scales [BLOCK_KV]; w [H_PAD]
    scores = tl.dot(q_bf16, kv_bf16)  # [H_PAD, BLOCK_KV] fp32
    scores = scores * kv_scales[None, :]
    scores = tl.maximum(scores, 0.0)
    scores = scores * w[:, None]
    return tl.sum(scores, 0)  # [BLOCK_KV]


@triton.jit
def _fp8_mqa_logits_cuda_kernel(
    Q_ptr,
    KV_ptr,
    Scales_ptr,
    W_ptr,
    Start_ptr,
    End_ptr,
    Logits_ptr,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    stride_q_m,
    stride_w_m,
    stride_logits_m,
    stride_kv_n,
    stride_kv_d,
    stride_scale_n,
):
    m = tl.program_id(0)
    offs_h = tl.arange(0, NUM_HEADS_PADDED)
    h_mask = offs_h < NUM_HEADS
    offs_d = tl.arange(0, HEAD_SIZE)
    q = _fp8e4m3_to_f32(
        tl.load(
            Q_ptr + m * stride_q_m + offs_h[:, None] * HEAD_SIZE + offs_d[None, :],
            mask=h_mask[:, None],
            other=0,
        )
    ).to(tl.bfloat16)
    w = tl.load(W_ptr + m * stride_w_m + offs_h, mask=h_mask, other=0.0)

    start = tl.maximum(tl.load(Start_ptr + m), 0)
    end = tl.minimum(tl.load(End_ptr + m), seq_len_kv)

    cols = start + tl.arange(0, BLOCK_KV)
    kv_ptrs = KV_ptr + cols[None, :] * stride_kv_n + offs_d[:, None] * stride_kv_d
    scale_ptrs = Scales_ptr + cols * stride_scale_n
    out_ptrs = Logits_ptr + m * stride_logits_m + cols
    for _ in tl.range(start, end, BLOCK_KV):
        mask = cols < end
        kv = _fp8e4m3_to_f32(
            tl.load(
                kv_ptrs,
                mask=mask[None, :],
                other=0,
            )
        ).to(tl.bfloat16)
        sc = tl.load(scale_ptrs, mask=mask, other=0.0)
        scores = _mqa_logits_inner(q, kv, sc, w, BLOCK_KV)
        tl.store(out_ptrs, scores, mask=mask)
        kv_ptrs += BLOCK_KV * stride_kv_n
        scale_ptrs += BLOCK_KV * stride_scale_n
        out_ptrs += BLOCK_KV
        cols += BLOCK_KV


def fp8_mqa_logits_cuda(q, k_fp8, kv_scales, weights, cu_starts, cu_ends):
    """Contiguous-KV fp8 MQA logits (prefill call site).

    Args:
        q: Queries ``[M, H, D]`` fp8e4m3 with head/token dims contiguous.
        k_fp8: Keys ``[N, D]`` fp8e4m3.
        kv_scales: Per-key scales ``[N]`` (or ``[N, 1]``) float32.
        weights: Per-head weights ``[M, H]`` float32 (Q scale folded in).
        cu_starts: Per-row window start (inclusive) ``[M]`` int32.
        cu_ends: Per-row window end (exclusive) ``[M]`` int32.

    Returns:
        Logits ``[M, N]`` float32; positions outside
        ``[cu_starts[m], cu_ends[m])`` are ``-inf``.
    """
    M, num_heads, head_size = q.shape
    N = k_fp8.shape[0]
    assert q.dtype == k_fp8.dtype == torch.float8_e4m3fn
    assert head_size >= 16 and head_size & (head_size - 1) == 0
    assert q.stride(1) == head_size and q.stride(2) == 1
    assert weights.stride(1) == 1
    assert cu_starts.stride(0) == 1 and cu_ends.stride(0) == 1
    kv_scales_1d = kv_scales.reshape(-1)
    logits = torch.full((M, N), -float("inf"), dtype=torch.float32, device=q.device)
    _fp8_mqa_logits_cuda_kernel[(M,)](
        q.view(torch.uint8),
        k_fp8.view(torch.uint8),
        kv_scales_1d,
        weights,
        cu_starts,
        cu_ends,
        logits,
        N,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        NUM_HEADS_PADDED=triton.next_power_of_2(max(num_heads, 16)),
        BLOCK_KV=64,
        stride_q_m=q.stride(0),
        stride_w_m=weights.stride(0),
        stride_logits_m=logits.stride(0),
        stride_kv_n=k_fp8.stride(0),
        stride_kv_d=k_fp8.stride(1),
        stride_scale_n=kv_scales_1d.stride(0),
        num_warps=4,
    )
    return logits


@triton.jit
def _paged_fp8_mqa_logits_cuda_kernel(
    Q_ptr,
    Cache_ptr,
    W_ptr,
    Ctx_ptr,
    BlockTable_ptr,
    Logits_ptr,
    max_ctx,
    block_size,
    next_n,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    stride_q_m,
    stride_w_m,
    stride_logits_m,
    stride_bt_b,
    stride_cache_b,  # bytes per block row (may include inter-block padding)
    stride_cache_n,  # bytes per token row within a block
):
    row = tl.program_id(0)
    b = row // next_n
    ctx = tl.load(Ctx_ptr + row)
    offs_h = tl.arange(0, NUM_HEADS_PADDED)
    h_mask = offs_h < NUM_HEADS
    offs_d = tl.arange(0, HEAD_SIZE)
    q = _fp8e4m3_to_f32(
        tl.load(
            Q_ptr + row * stride_q_m + offs_h[:, None] * HEAD_SIZE + offs_d[None, :],
            mask=h_mask[:, None],
            other=0,
        )
    ).to(tl.bfloat16)
    w = tl.load(W_ptr + row * stride_w_m + offs_h, mask=h_mask, other=0.0)

    for start_n in tl.range(0, ctx, BLOCK_KV):
        offs_n = start_n + tl.arange(0, BLOCK_KV)
        mask = offs_n < ctx
        blocks = tl.load(
            BlockTable_ptr + b * stride_bt_b + offs_n // block_size,
            mask=mask,
            other=0,
        ).to(tl.int64)
        row_bytes = blocks * stride_cache_b + (offs_n % block_size) * stride_cache_n
        kv = _fp8e4m3_to_f32(
            tl.load(
                Cache_ptr + row_bytes[None, :] + offs_d[:, None],
                mask=mask[None, :],
                other=0,
            )
        ).to(tl.bfloat16)
        scale_bytes = row_bytes + HEAD_SIZE
        sc = tl.load(
            Cache_ptr.to(tl.pointer_type(tl.float32)) + scale_bytes // 4,
            mask=mask,
            other=0.0,
        )
        scores = _mqa_logits_inner(q, kv, sc, w, BLOCK_KV)
        tl.store(Logits_ptr + row * stride_logits_m + offs_n, scores, mask=mask)


def paged_fp8_mqa_logits_cuda(
    q, kv_cache, weights, context_lens, block_tables, max_len
):
    """Paged fp8 MQA logits (decode call site).

    Args:
        q: Queries ``[B, next_n, H, D]`` fp8e4m3 (``next_n == 1`` only).
        kv_cache: ``[num_blocks, block_size, D + 4]`` uint8; per token, ``D``
            fp8 value bytes followed by a float32 scale (little-endian).
            Must be contiguous and 4-byte aligned (asserted).
        weights: Per-head weights ``[B * next_n, H]`` float32.
        context_lens: Context lengths ``[B, next_n]`` int32.
        block_tables: Block ids ``[B, max_blocks]`` int32.
        max_len: Static row width for the output (``max_model_len``); avoids
            a ``tensor.max()`` host sync per call.

    Returns:
        Logits ``[B * next_n, max_len]`` float32. Only ``[0, ctx)`` per row
        is written; the rest is undefined — the decode top-k consumer bounds
        its scan by ``seq_lens``.
    """
    B, next_n, num_heads, head_size = q.shape
    assert next_n == 1, "sm_80 port supports next_n=1 only (no MTP)"
    assert q.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    assert head_size >= 16 and head_size & (head_size - 1) == 0
    assert kv_cache.stride(2) == 1
    assert kv_cache.stride(1) == head_size + 4
    # The cache is a view into a packed pool: the block stride is the pool's
    # byte span (shared by several layer types), rows are contiguous within a
    # block. Addressing is block_id * stride(0) + off * stride(1).
    assert kv_cache.stride(0) >= kv_cache.shape[1] * kv_cache.stride(1), (
        "block stride smaller than logical block size"
    )
    assert kv_cache.stride(0) % 4 == 0 and kv_cache.data_ptr() % 4 == 0, (
        "fp32 scale load needs 4B alignment"
    )
    q_flat = q.reshape(B * next_n, num_heads, head_size)
    assert q_flat.stride(1) == head_size and q_flat.stride(2) == 1
    assert weights.stride(1) == 1 and block_tables.stride(1) == 1
    ctx_flat = context_lens.reshape(-1)
    logits = torch.empty((B * next_n, max_len), dtype=torch.float32, device=q.device)
    _paged_fp8_mqa_logits_cuda_kernel[(B * next_n,)](
        q_flat.view(torch.uint8),
        kv_cache,
        weights,
        ctx_flat,
        block_tables,
        logits,
        max_len,
        kv_cache.shape[1],
        next_n,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        NUM_HEADS_PADDED=triton.next_power_of_2(max(num_heads, 16)),
        BLOCK_KV=64,
        stride_q_m=q_flat.stride(0),
        stride_w_m=weights.stride(0),
        stride_logits_m=logits.stride(0),
        stride_bt_b=block_tables.stride(0),
        stride_cache_b=kv_cache.stride(0),
        stride_cache_n=kv_cache.stride(1),
        num_warps=4,
    )
    return logits


def sglang_paged_mqa_logits(
    q, kv_cache, weights, seq_lens, page_table, deep_gemm_metadata, max_seq_len, use_fp4=False
):
    """DeepGEMM `fp8_paged_mqa_logits` signature adapter for sm_80.

    q: [M, 1, H, D] fp8; kv_cache: [pages, 64, 1, 132] uint8 (scale in last
    4 bytes of each 132B row); weights: [M, H] f32; seq_lens: [M, 1] int;
    page_table: [M, max_blocks] int32 (block granularity 64 rows).
    deep_gemm_metadata is ignored (None on sm80). Writes logits [M, max];
    tail beyond each ctx is undefined — sglang's topk scans bounded by len.
    """
    assert not use_fp4
    kv = kv_cache.squeeze(2)
    assert kv.dim() == 3
    logits = paged_fp8_mqa_logits_cuda(
        q,
        kv,
        weights,
        seq_lens,
        page_table,
        max_len=int(max_seq_len),
    )
    return logits
