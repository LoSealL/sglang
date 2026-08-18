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
    # e=15, m=7 is the only NaN encoding; decode it to NaN (torch parity, the
    # dequant_k_cache self-check compares against torch on random bytes).
    u = x.to(tl.uint32)
    e = (u >> 3) & 15
    m = u & 7
    norm = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    s = tl.where((u & 0x80) != 0, -1.0, 1.0)
    out = tl.where(e == 0, sub, norm) * s
    return tl.where((e == 15) & (m == 7), float("nan"), out)


@triton.jit
def _f32_to_e4m3_bits(x):
    # Software fp32 -> e4m3fn encode (RNE, satfinite to +/-448, NaN -> 0x7F):
    # inverse of _fp8e4m3_to_f32 above (bias 7, 3 mantissa bits, subnormal
    # step 2^-9). Triton rejects fp8e4nv pointers on sm_80, so OCP fp8 values
    # are stored as bits via a uint8 view. Validated bit-exact vs torch.
    ux = x.to(tl.uint32, bitcast=True)
    sign = (ux >> 24).to(tl.int32) & 0x80
    e32 = ((ux >> 23) & 0xFF).to(tl.int32)
    frac = (ux & 0x7FFFFF).to(tl.int32)

    # Normals |x| >= 2^-6: e4m3 exp = fp32 exp - 120 (bias 127 vs 7); RNE the
    # top 3 mantissa bits, carrying into the exponent on overflow.
    m = frac >> 20
    rem = frac & 0xFFFFF
    round_up = (rem > 0x80000) | ((rem == 0x80000) & ((m & 1) != 0))
    m = m + round_up.to(tl.int32)
    carry = (m == 8).to(tl.int32)
    norm_bits = ((e32 - 120 + carry) << 3) | tl.where(carry != 0, 0, m)

    # Subnormals |x| < 2^-6: RNE of |x| * 2^9 to an integer (the 2^23 + 2^22
    # magic add rounds to nearest even); 8 carries into the first normal.
    f = tl.abs(x) * 512.0
    m_sub = ((f + 12582912.0) - 12582912.0).to(tl.int32)

    bits = tl.where(e32 < 121, m_sub, norm_bits)
    # Saturation matches torch/c10 (2.13): everything |x| >= 448-that-rounds-up
    # including inf saturates to max finite 0x7E; only NaN inputs map to 0x7F.
    sat = (e32 >= 136) | ((e32 == 135) & (frac >= 0x600000)) | (e32 == 255)
    nan_out = (e32 == 255) & (frac != 0)
    bits = tl.where(nan_out, 0x7F, tl.where(sat, 0x7E, bits))
    return (sign | bits).to(tl.uint8)


@triton.jit
def _halve(x, N: tl.constexpr, BKV: tl.constexpr):
    # x [N, BKV] -> [N//2, BKV]: fixed pairwise tree (row h + row h + N//2).
    # tl.sum's reduction order is layout-chosen, which breaks bitwise equality
    # between the per-row and BLOCK_M-tiled kernels; this tree is identical in
    # any kernel/warp-count context. Kernels launch with enable_fp_fusion=False
    # so the adds stay pure (no mul+add FMA contraction).
    y = tl.reshape(x, (2, N // 2, BKV))
    y = tl.permute(y, (1, 2, 0))
    u, v = tl.split(y)
    return u + v


@triton.jit
def _sum_h(x, H_PAD: tl.constexpr, BKV: tl.constexpr):
    if H_PAD > 64:
        x = _halve(x, H_PAD, BKV)
    if H_PAD > 32:
        x = _halve(x, 64, BKV)
    if H_PAD > 16:
        x = _halve(x, 32, BKV)
    if H_PAD > 8:
        x = _halve(x, 16, BKV)
    if H_PAD > 4:
        x = _halve(x, 8, BKV)
    if H_PAD > 2:
        x = _halve(x, 4, BKV)
    if H_PAD > 1:
        x = _halve(x, 2, BKV)
    return tl.reshape(x, (BKV,))


@triton.jit
def _mqa_logits_inner(
    q_bf16,
    kv_bf16,
    kv_scales,
    w,
    NUM_HEADS_PADDED: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # q_bf16 [H_PAD, D]; kv_bf16 [D, BLOCK_KV]; kv_scales [BLOCK_KV]; w [H_PAD]
    scores = tl.dot(q_bf16, kv_bf16)  # [H_PAD, BLOCK_KV] fp32
    scores = scores * kv_scales[None, :]
    scores = tl.maximum(scores, 0.0)
    scores = scores * w[:, None]
    return _sum_h(scores, NUM_HEADS_PADDED, BLOCK_KV)  # [BLOCK_KV]


@triton.jit
def _halve_mid(x, B: tl.constexpr, N: tl.constexpr, BKV: tl.constexpr):
    # x [B, N, BKV] -> [B, N//2, BKV]: same pairwise order as _halve (row
    # n + row n + N//2 on the head axis), batched over B query rows.
    y = tl.reshape(x, (B, 2, N // 2, BKV))
    y = tl.permute(y, (0, 2, 3, 1))
    u, v = tl.split(y)
    return u + v


@triton.jit
def _sum_h3(x, B: tl.constexpr, H_PAD: tl.constexpr, BKV: tl.constexpr):
    # x [B, H_PAD, BKV] -> [B, BKV]; batched _sum_h (identical pairing order,
    # so the tiled kernel stays bitwise-equal to the per-row kernel).
    if H_PAD > 64:
        x = _halve_mid(x, B, H_PAD, BKV)
    if H_PAD > 32:
        x = _halve_mid(x, B, 64, BKV)
    if H_PAD > 16:
        x = _halve_mid(x, B, 32, BKV)
    if H_PAD > 8:
        x = _halve_mid(x, B, 16, BKV)
    if H_PAD > 4:
        x = _halve_mid(x, B, 8, BKV)
    if H_PAD > 2:
        x = _halve_mid(x, B, 4, BKV)
    if H_PAD > 1:
        x = _halve_mid(x, B, 2, BKV)
    return tl.reshape(x, (B, BKV))


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
        scores = _mqa_logits_inner(q, kv, sc, w, NUM_HEADS_PADDED, BLOCK_KV)
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
        enable_fp_fusion=False,
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
    split_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    stride_q_m,
    stride_w_m,
    stride_logits_m,
    stride_bt_b,
    stride_cache_b,  # bytes per page: block layout [block_size*HEAD_SIZE values | block_size*4 scales]
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    b = row // next_n
    ctx = tl.load(Ctx_ptr + row)
    start = split * split_kv
    if start < ctx:
        end = tl.minimum(start + split_kv, ctx)
        offs_h = tl.arange(0, NUM_HEADS_PADDED)
        h_mask = offs_h < NUM_HEADS
        offs_d = tl.arange(0, HEAD_SIZE)
        q = _fp8e4m3_to_f32(
            tl.load(
                Q_ptr
                + row * stride_q_m
                + offs_h[:, None] * HEAD_SIZE
                + offs_d[None, :],
                mask=h_mask[:, None],
                other=0,
            )
        ).to(tl.bfloat16)
        w = tl.load(W_ptr + row * stride_w_m + offs_h, mask=h_mask, other=0.0)

        for start_n in tl.range(start, end, BLOCK_KV):
            offs_n = start_n + tl.arange(0, BLOCK_KV)
            mask = offs_n < end
            blocks = tl.load(
                BlockTable_ptr + b * stride_bt_b + offs_n // block_size,
                mask=mask,
                other=0,
            ).to(tl.int64)
            page_off = offs_n % block_size
            # sglang block layout (kernels/jit .../store.cuh fused_store_cache):
            # value row at page + off*HEAD_SIZE, scale at
            # page + block_size*HEAD_SIZE + off*4. All offsets 4B-aligned.
            row_bytes = blocks * stride_cache_b + page_off * HEAD_SIZE
            kv = _fp8e4m3_to_f32(
                tl.load(
                    Cache_ptr + row_bytes[None, :] + offs_d[:, None],
                    mask=mask[None, :],
                    other=0,
                )
            ).to(tl.bfloat16)
            scale_bytes = (
                blocks * stride_cache_b + block_size * HEAD_SIZE + page_off * 4
            )
            sc = tl.load(
                Cache_ptr.to(tl.pointer_type(tl.float32)) + scale_bytes // 4,
                mask=mask,
                other=0.0,
            )
            scores = _mqa_logits_inner(q, kv, sc, w, NUM_HEADS_PADDED, BLOCK_KV)
            tl.store(Logits_ptr + row * stride_logits_m + offs_n, scores, mask=mask)


@triton.jit
def _paged_fp8_mqa_logits_tiled_kernel(
    Q_ptr,
    Cache_ptr,
    W_ptr,
    Ctx_ptr,
    BlockTable_ptr,
    Logits_ptr,
    M,
    next_n,
    b_max,
    split_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_HEADS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    block_size: tl.constexpr,
    stride_q_m,
    stride_w_m,
    stride_logits_m,
    stride_bt_b,
    stride_cache_b,
):
    # Prefill variant of the kernel above: one CTA covers BLOCK_M query rows,
    # so each 64-row KV block is software-decoded once and the logits run as
    # one [BLOCK_M * H_PAD, D] x [D, BLOCK_KV] tensor-core GEMM per KV block
    # (the per-row kernel's [H_PAD, D] dot wastes ~90% of TC throughput and
    # re-decodes the same KV bytes per query row). BLOCK_KV == block_size and
    # split_kv % block_size == 0 (asserted in the wrapper) keep every KV step
    # inside one page, so the page id is the scalar start_n // block_size.
    # Per-row ctxs differ inside a tile: the loop bound is the tile max and
    # each row stores only its own [0, end) columns.
    pid_m = tl.program_id(0)
    split = tl.program_id(1)
    start = split * split_kv
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_m_c = tl.minimum(offs_m, M - 1)
    offs_n = tl.arange(0, BLOCK_KV)
    # flattened (row, head) index for the big dot
    offs_mh = tl.arange(0, BLOCK_M * NUM_HEADS_PADDED)
    mh_row = pid_m * BLOCK_M + offs_mh // NUM_HEADS_PADDED
    mh_h = offs_mh % NUM_HEADS_PADDED
    mh_mask = (mh_row < M) & (mh_h < NUM_HEADS)
    mh_row_c = tl.minimum(mh_row, M - 1)
    offs_d = tl.arange(0, HEAD_SIZE)

    ctx_vec = tl.load(Ctx_ptr + offs_m, mask=m_mask, other=0)
    end_vec = tl.minimum(start + split_kv, ctx_vec)
    block_end = tl.max(end_vec, 0)
    if start < block_end:
        b_vec = tl.minimum(offs_m // next_n, b_max)
        # q/w are loaded once per CTA and held in registers across the KV
        # loop (in-loop reloads of the strided (m, h)-flattened tile measure
        # 20-40x slower; fp8-byte-held + per-block re-decode makes the
        # compiler spill the whole tile).
        q2d = _fp8e4m3_to_f32(
            tl.load(
                Q_ptr
                + mh_row_c[:, None] * stride_q_m
                + mh_h[:, None] * HEAD_SIZE
                + offs_d[None, :],
                mask=mh_mask[:, None],
                other=0,
            )
        ).to(tl.bfloat16)
        w2d = tl.load(W_ptr + mh_row_c * stride_w_m + mh_h, mask=mh_mask, other=0.0)
        for start_n in tl.range(start, block_end, BLOCK_KV):
            col = start_n // block_size
            pids = tl.load(BlockTable_ptr + b_vec * stride_bt_b + col)
            act = end_vec > start_n
            lo = tl.min(tl.where(act, pids, 2147483647), 0)
            hi = tl.max(tl.where(act, pids, -2147483648), 0)
            if lo == hi:
                # All active rows read this KV block from the same page
                # (request-keyed block tables): decode once, one big GEMM
                # for the whole tile. q/w live in registers across the KV
                # loop (BLOCK_M * H_PAD * 2B; BLOCK_M is bounded to keep
                # this under the register budget).
                base = lo * stride_cache_b
                kv = _fp8e4m3_to_f32(
                    tl.load(
                        Cache_ptr
                        + base
                        + offs_n[None, :] * HEAD_SIZE
                        + offs_d[:, None],
                        mask=offs_n[None, :] < block_end,
                        other=0,
                    )
                ).to(tl.bfloat16)
                sc = tl.load(
                    Cache_ptr.to(tl.pointer_type(tl.float32))
                    + (base + block_size * HEAD_SIZE + offs_n * 4) // 4,
                    mask=offs_n < block_end,
                    other=0.0,
                )
                scores = tl.dot(q2d, kv)  # [BLOCK_M * H_PAD, BLOCK_KV] fp32
                scores = scores * sc[None, :]
                scores = tl.maximum(scores, 0.0)
                scores = scores * w2d[:, None]
                out = _sum_h3(
                    tl.reshape(scores, (BLOCK_M, NUM_HEADS_PADDED, BLOCK_KV)),
                    BLOCK_M,
                    NUM_HEADS_PADDED,
                    BLOCK_KV,
                )  # [BLOCK_M, BLOCK_KV]
                tl.store(
                    Logits_ptr
                    + offs_m_c[:, None] * stride_logits_m
                    + start_n
                    + offs_n[None, :],
                    out,
                    mask=(offs_n[None, :] < end_vec[:, None]) & m_mask[:, None],
                )
            else:
                # Request boundary inside the tile: rows disagree on the page,
                # fall back to per-row fetches (bitwise-equal, just slower).
                offs_h = tl.arange(0, NUM_HEADS_PADDED)
                h_mask = offs_h < NUM_HEADS
                for i in tl.range(0, BLOCK_M):
                    row_raw = pid_m * BLOCK_M + i
                    row = tl.minimum(row_raw, M - 1)
                    b_i = tl.minimum(row // next_n, b_max)
                    pid_i = tl.load(BlockTable_ptr + b_i * stride_bt_b + col)
                    base = pid_i * stride_cache_b
                    e = tl.minimum(start + split_kv, tl.load(Ctx_ptr + row))
                    e = tl.where(row_raw < M, e, 0)
                    mask_i = offs_n < e
                    kv = _fp8e4m3_to_f32(
                        tl.load(
                            Cache_ptr
                            + base
                            + offs_n[None, :] * HEAD_SIZE
                            + offs_d[:, None],
                            mask=mask_i[None, :],
                            other=0,
                        )
                    ).to(tl.bfloat16)
                    sc = tl.load(
                        Cache_ptr.to(tl.pointer_type(tl.float32))
                        + (base + block_size * HEAD_SIZE + offs_n * 4) // 4,
                        mask=mask_i,
                        other=0.0,
                    )
                    q = _fp8e4m3_to_f32(
                        tl.load(
                            Q_ptr
                            + row * stride_q_m
                            + offs_h[:, None] * HEAD_SIZE
                            + offs_d[None, :],
                            mask=h_mask[:, None],
                            other=0,
                        )
                    ).to(tl.bfloat16)
                    w = tl.load(
                        W_ptr + row * stride_w_m + offs_h, mask=h_mask, other=0.0
                    )
                    scores = _mqa_logits_inner(q, kv, sc, w, NUM_HEADS_PADDED, BLOCK_KV)
                    tl.store(
                        Logits_ptr + row * stride_logits_m + start_n + offs_n,
                        scores,
                        mask=mask_i,
                    )


# KV rows per CTA (grid axis 1 = ceil(max_len / _SPLIT_KV)); each CTA writes a
# disjoint logits slice, so any value is correct — measured fastest at decode
# ctx 8k/24.5k on idle A100-80GB (128: 0.079/0.075 ms; 256: 0.093/0.084 ms;
# 512: 0.116/0.117 ms; 1024: 0.179/0.180 ms): bs=1 decode needs ~200 CTAs to
# fill 108 SMs; the kernel is per-CTA-overhead bound, so more/smaller splits win.
_SPLIT_KV = 128

# Prefill tiles: M >= this routes to the BLOCK_M-tiled kernel (decode sizes
# keep the tuned per-row kernel). Tuned on idle A100-80GB at M=8192
# (c4 ctx 2k/6k/24.5k -> 5.4/16.0/63.7 ms vs the per-row kernel's
# 13.5/37.5/131.5 ms, ~2.1-2.6x): BM=4/nw=8/split=4096 compiles spill-free
# (250 regs); BM=2/nw=4 is 4% faster but spills (246 slots), so the robust
# config wins. The tiled split is far larger than decode's 128: each CTA's
# held q tile amortizes over many KV blocks.
_TILED_MIN_M = 32
_TILE_M = 4
_SPLIT_KV_TILED = 4096


def paged_fp8_mqa_logits_cuda(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    max_len,
    block_size=64,
    split_kv=None,
    force_row_kernel=False,
    force_tiled_kernel=False,
):
    """Paged fp8 MQA logits (decode call site).

    Args:
        q: Queries ``[B, next_n, H, D]`` fp8e4m3 (``next_n == 1`` only).
        kv_cache: Raw per-layer indexer pool buffer ``[num_blocks,
            block_size * (D + 4)]`` uint8 in sglang's block layout (the
            writer is ``kernels/jit/csrc/deepseek_v4/store.cuh``
            ``fused_store_cache(type="indexer")``): each page is
            ``[block_size x D fp8 value rows][block_size x 4B fp32 scales]``
            — value at ``page + off * D``, scale at
            ``page + block_size * D + off * 4``. This is NOT vLLM's
            interleaved ``D + 4``-byte row layout. Must be 4-byte aligned
            (asserted).
        weights: Per-head weights ``[B * next_n, H]`` float32.
        context_lens: Context lengths ``[B, next_n]`` int32.
        block_tables: Block ids ``[B, max_blocks]`` int32 (row granularity
            ``block_size``).
        max_len: Static row width for the output (``max_model_len``); avoids
            a ``tensor.max()`` host sync per call.
        block_size: Rows per cache page (indexer pool: 64).
        split_kv: KV rows per CTA along grid axis 1 (default: tuned
            ``_SPLIT_KV``, ``_SPLIT_KV_TILED`` for the tiled prefill path).
            Splits beyond a row's ctx exit immediately.
        force_row_kernel / force_tiled_kernel: test hooks to pin the kernel
            choice regardless of the ``M`` selector.

    Returns:
        Logits ``[B * next_n, max_len]`` float32. Only ``[0, ctx)`` per row
        is written; the rest is undefined — the decode top-k consumer bounds
        its scan by ``seq_lens``.
    """
    B, next_n, num_heads, head_size = q.shape
    assert next_n == 1, "sm_80 port supports next_n=1 only (no MTP)"
    assert q.dtype == torch.float8_e4m3fn
    assert kv_cache.dtype == torch.uint8
    assert kv_cache.dim() == 2
    assert head_size >= 16 and head_size & (head_size - 1) == 0
    assert kv_cache.stride(1) == 1
    assert kv_cache.shape[1] == block_size * (head_size + 4), (
        "raw block-layout pool buffer expected: [num_blocks, block_size * (D + 4)]"
    )
    assert kv_cache.stride(0) % 4 == 0 and kv_cache.data_ptr() % 4 == 0, (
        "fp32 scale load needs 4B alignment"
    )
    q_flat = q.reshape(B * next_n, num_heads, head_size)
    assert q_flat.stride(1) == head_size and q_flat.stride(2) == 1
    assert weights.stride(1) == 1 and block_tables.stride(1) == 1
    ctx_flat = context_lens.reshape(-1)
    logits = torch.empty((B * next_n, max_len), dtype=torch.float32, device=q.device)
    M = B * next_n
    tiled = (M >= _TILED_MIN_M or force_tiled_kernel) and not force_row_kernel
    if split_kv is None:
        split_kv = _SPLIT_KV_TILED if tiled else _SPLIT_KV
    if tiled:
        assert split_kv % block_size == 0 and block_size & (block_size - 1) == 0, (
            "tiled kernel needs one KV step per page (split_kv multiple of a"
            " power-of-two block_size)"
        )
        splits = (max_len + split_kv - 1) // split_kv
        _paged_fp8_mqa_logits_tiled_kernel[(triton.cdiv(M, _TILE_M), splits)](
            q_flat.view(torch.uint8),
            kv_cache,
            weights,
            ctx_flat,
            block_tables,
            logits,
            M,
            next_n,
            B - 1,
            split_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            NUM_HEADS_PADDED=triton.next_power_of_2(max(num_heads, 16)),
            BLOCK_M=_TILE_M,
            BLOCK_KV=block_size,
            block_size=block_size,
            stride_q_m=q_flat.stride(0),
            stride_w_m=weights.stride(0),
            stride_logits_m=logits.stride(0),
            stride_bt_b=block_tables.stride(0),
            stride_cache_b=kv_cache.stride(0),
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=False,
        )
        return logits
    splits = (max_len + split_kv - 1) // split_kv
    _paged_fp8_mqa_logits_cuda_kernel[(B * next_n, splits)](
        q_flat.view(torch.uint8),
        kv_cache,
        weights,
        ctx_flat,
        block_tables,
        logits,
        max_len,
        block_size,
        next_n,
        split_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        NUM_HEADS_PADDED=triton.next_power_of_2(max(num_heads, 16)),
        BLOCK_KV=64,
        stride_q_m=q_flat.stride(0),
        stride_w_m=weights.stride(0),
        stride_logits_m=logits.stride(0),
        stride_bt_b=block_tables.stride(0),
        stride_cache_b=kv_cache.stride(0),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return logits


def sglang_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    seq_lens,
    page_table,
    deep_gemm_metadata,
    max_seq_len,
    use_fp4=False,
):
    """DeepGEMM `fp8_paged_mqa_logits` signature adapter for sm_80.

    q: [M, 1, H, D] fp8; kv_cache: the RAW per-layer indexer pool buffer
    ``[pages, 64 * (D + 4)]`` uint8 (``get_index_k_with_scale_buffer``,
    BEFORE the ``[pages, 64, 1, D + 4]`` view) in sglang's block layout — per
    page ``[64 x D fp8 values | 64 x 4B fp32 scales]``. That view is
    nominal-shape only (DeepGEMM layout convention); its strides do not
    describe the real byte layout, so sm_80 takes the raw buffer instead.
    weights: [M, H] f32; seq_lens: [M, 1] int; page_table: [M, max_blocks]
    int32 (block granularity 64 rows). deep_gemm_metadata is ignored (None on
    sm80). Writes logits [M, max]; tail beyond each ctx is undefined —
    sglang's topk scans bounded by len.
    """
    assert not use_fp4
    assert kv_cache.dtype == torch.uint8 and kv_cache.dim() == 2
    return paged_fp8_mqa_logits_cuda(
        q,
        kv_cache,
        weights,
        seq_lens,
        page_table,
        max_len=int(max_seq_len),
    )
