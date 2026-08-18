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

# Block-size policy (measured on idle A100, T=8192/H=64 prefill shapes):
# BLOCK_H=32 fuses two head-tiles per CTA so each gathered KV row is
# dequantized/dotted once for 32 heads instead of 16 — c4 45.6->35.8 ms,
# c128@94k 61.9->40.7 ms; it LOSES below T~32 (decode: 215->277 us), so small
# batches keep BLOCK_H=16 (4 tiles = more CTAs). BLOCK_N=32 beats 64 for the
# short c4-style lists (<=640 rows/token: 31.9 vs 35.8 ms) while long c128
# lists prefer 64 (40.7 vs 43.4 ms).
_SPLIT_MIN_T_FOR_BH32 = 32
_SHORT_LIST_CAP = 640

# Split-K auto policy: measured NEVER faster than the single path on A100 at
# any list length (T=8192: c4 45.6 vs 48-56 ms, c128 61.9 vs 64-71, comp3000
# 136 vs 210 — partial write+read traffic dominates), so auto-enable only for
# extreme lists (> _SPLIT_MIN_LIST rows/token) as a safety valve; explicit
# num_splits works for any caller.
_SPLIT_MIN_LIST = 4096
_SPLIT_LIST_PER_SPLIT = 1024
_SPLIT_MAX = 8


@triton.jit
def _attend_packed(
    CacheU8_ptr,
    CacheBf16_ptr,
    idx_ptr,
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
    lo,
    hi,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Online-softmax attention over index rows [lo, hi) of token t's list
    (empty when lo >= hi); updates and returns (m, denom, acc)."""
    offs_d = tl.arange(0, HEAD_DIM)
    nope_mask = offs_d < _DIM_NOPE
    for start_n in tl.range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
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
        0,
        tl.load(SwaLens_ptr + t),
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
    )
    if HAS_COMP:
        m, denom, acc = _attend_packed(
            CompU8_ptr,
            CompBf16_ptr,
            CompIdx_ptr,
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
            0,
            tl.load(CompLens_ptr + t),
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


@triton.jit
def _sparse_mla_fwd_split_kernel(
    Q_ptr,
    PartO_ptr,
    PartM_ptr,
    PartL_ptr,
    SwaU8_ptr,
    SwaBf16_ptr,
    SwaIdx_ptr,
    SwaLens_ptr,
    CompU8_ptr,
    CompBf16_ptr,
    CompIdx_ptr,
    CompLens_ptr,
    sm_scale,
    stride_q_t,
    stride_q_h,
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
    NUM_SPLITS: tl.constexpr,
):
    """Split-K prefill: CTA (t, s) attends over split s of token t's
    FLATTENED [swa ++ comp] index list — boundaries at multiples of a
    BLOCK_N-aligned chunk, so one split may straddle the swa/comp seam (two
    sub-ranges) — and emits an UNNORMALIZED partial (m_s, l_s, acc_s). The
    sink is NOT applied here; the reducer owns it."""
    t = tl.program_id(0)
    s = tl.program_id(1)
    hb = tl.program_id(2)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_h = offs_h < H

    q = tl.load(
        Q_ptr + t * stride_q_t + offs_h[:, None] * stride_q_h + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    swa_len = tl.load(SwaLens_ptr + t)
    if HAS_COMP:
        comp_len = tl.load(CompLens_ptr + t)
    else:
        comp_len = 0
    total = swa_len + comp_len
    chunk = (total + NUM_SPLITS - 1) // NUM_SPLITS
    chunk = (chunk + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    lo = s * chunk
    hi = tl.minimum(lo + chunk, total)

    m = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)

    m, denom, acc = _attend_packed(
        SwaU8_ptr,
        SwaBf16_ptr,
        SwaIdx_ptr,
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
        lo,
        tl.minimum(hi, swa_len),
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
    )
    if HAS_COMP:
        m, denom, acc = _attend_packed(
            CompU8_ptr,
            CompBf16_ptr,
            CompIdx_ptr,
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
            tl.maximum(lo - swa_len, 0),
            tl.minimum(hi - swa_len, comp_len),
            HEAD_DIM=HEAD_DIM,
            BLOCK_N=BLOCK_N,
        )

    # Partials layout: [T, NUM_SPLITS, tiles, BLOCK_H, ...] contiguous —
    # indexed with TILE-LOCAL head offsets (the tile itself is in `base`).
    # int64: T*S*tiles*BLOCK_H*HEAD_DIM exceeds 2^31 at prefill sizes
    # (8192*8*4*16*512 == 2^31) — int32 arithmetic silently corrupts.
    base = (t * NUM_SPLITS + s) * tl.num_programs(2) + hb
    base64 = base.to(tl.int64) * (BLOCK_H * HEAD_DIM)
    offs_hl = tl.arange(0, BLOCK_H)
    tl.store(
        PartO_ptr + base64 + offs_hl[:, None] * HEAD_DIM + offs_d[None, :],
        acc,
        mask=mask_h[:, None],
    )
    tl.store(PartM_ptr + base * BLOCK_H + offs_hl, m, mask=mask_h)
    tl.store(PartL_ptr + base * BLOCK_H + offs_hl, denom, mask=mask_h)


@triton.jit
def _split_reduce_kernel(
    PartO_ptr,
    PartM_ptr,
    PartL_ptr,
    Sink_ptr,
    Out_ptr,
    stride_o_t,
    stride_o_h,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Merge the per-split partials with a logsumexp combine. The sink enters
    here as the softmax init (m=sink, l=sink>-inf?1:0, acc=0) — it contributes
    no value to the accumulator, matching the single-path semantics."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_h = offs_h < H
    tiles = tl.num_programs(1)

    sink = tl.load(Sink_ptr + offs_h, mask=mask_h, other=float("-inf"))
    m = sink
    l = tl.where(sink > float("-inf"), 1.0, 0.0)
    acc = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)

    for s in tl.range(0, NUM_SPLITS):
        base = (t * NUM_SPLITS + s) * tiles + hb
        base64 = base.to(tl.int64) * (BLOCK_H * HEAD_DIM)
        offs_hl = tl.arange(0, BLOCK_H)
        ms = tl.load(
            PartM_ptr + base * BLOCK_H + offs_hl, mask=mask_h, other=float("-inf")
        )
        ls = tl.load(PartL_ptr + base * BLOCK_H + offs_hl, mask=mask_h, other=0.0)
        a = tl.load(
            PartO_ptr + base64 + offs_hl[:, None] * HEAD_DIM + offs_d[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        m_new = tl.maximum(m, ms)
        # Guard the both--inf case (no sink, empty splits): exp(-inf - -inf)
        # is NaN; zero rescales keep l=0, acc=0 for that head.
        ro = tl.where(m > float("-inf"), tl.exp(m - m_new), 0.0)
        rn = tl.where(ms > float("-inf"), tl.exp(ms - m_new), 0.0)
        acc = acc * ro[:, None] + a * rn[:, None]
        l = l * ro + ls * rn
        m = m_new

    l = tl.maximum(l, 1e-20)
    out = (acc / l[:, None]).to(Out_ptr.dtype.element_ty)
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
    num_splits: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over index-selected rows of the packed paged KV.

    q/out: [T, H, 512] bf16. swa_cache_u8/comp_cache_u8: [num_pages,
    bytes_per_page_padded] uint8 pools in the DSV4 packed layout (see
    dequant_k_cache). swa_indices/comp_indices: [T, W] int32 with global slot
    ids, -1 padded; swa_lens/comp_lens: [T] int32 segment lengths.
    comp_cache_u8=None selects the SWA-only single-segment specialization.

    num_splits>1 routes to the split-K path (one CTA per (token, split,
    head-tile) plus a logsumexp reduce kernel); num_splits=None auto-enables
    it only for prefill-sized batches with long per-token index lists (c4/c128
    chunks) — decode keeps the single-CTA-per-token path. The split path
    allocates a fp32 partials workspace that grows as
    T·num_splits·head_tiles·16·512 (≈8.6GB at T=8192/S=8/H=64) — callers
    must budget it. Writes and returns ``out``.
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

    # Per-token flattened index-list capacity (sync-free upper bound on the
    # actual device-side lens) drives the block-size and split policies.
    cap = swa_indices.shape[-1] + (comp_indices.shape[-1] if has_comp else 0)
    if num_splits is None and cap > _SPLIT_MIN_LIST:
        num_splits = min(_SPLIT_MAX, -(-cap // _SPLIT_LIST_PER_SPLIT))
    if num_splits is None or num_splits <= 1:
        BLOCK_H = 32 if (H % 32 == 0 and T >= _SPLIT_MIN_T_FOR_BH32) else 16
        BLOCK_N = 32 if (BLOCK_H == 32 and cap <= _SHORT_LIST_CAP) else 64
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

    S = num_splits
    BLOCK_H, BLOCK_N = 16, 64
    tiles = triton.cdiv(H, BLOCK_H)
    # Unnormalized per-split partials; fp32 so the merge only reassociates.
    part_o = torch.empty(
        (T, S, tiles, BLOCK_H, head_dim), dtype=torch.float32, device=q.device
    )
    part_m = torch.empty((T, S, tiles, BLOCK_H), dtype=torch.float32, device=q.device)
    part_l = torch.empty_like(part_m)
    _sparse_mla_fwd_split_kernel[(T, S, tiles)](
        q,
        part_o,
        part_m,
        part_l,
        swa_cache_u8,
        swa_bf16,
        swa_indices,
        swa_lens,
        comp_cache_u8,
        comp_bf16,
        comp_indices,
        comp_lens,
        sm_scale,
        q.stride(0),
        q.stride(1),
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
        NUM_SPLITS=S,
        num_warps=8,
        num_stages=1,
    )
    _split_reduce_kernel[(T, tiles)](
        part_o,
        part_m,
        part_l,
        sink,
        out,
        out.stride(0),
        out.stride(1),
        H=H,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H,
        NUM_SPLITS=S,
        num_warps=8,
    )
    return out
