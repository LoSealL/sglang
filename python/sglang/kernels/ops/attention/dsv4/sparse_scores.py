"""Fused relu-MQA scoring for the DeepSeek V4.1 sparse index layers.

``dsv41_sparse.scores`` computes ``sum_h w[t, h] * relu(q[t, h] . k[n])`` as
``einsum("bhd,nd->bhn")`` followed by three tensor-wide passes (relu, weight,
head sum). That materialises ``[T, H, N]``, which at a 4k prefill with 32 heads
and 2002 candidate rows is 0.48 GiB and costs ~1.46 GiB of traffic per call,
against 0.061 GiB for the [T, N] result the caller actually wants.

This fuses the head reduction into the GEMM: one ``tl.dot`` per head, with the
relu and weight applied to the fp32 accumulator before it joins the row sum, so
the [T, H, N] intermediate never exists. Shapes are static in T and N (masked),
which also makes the launch capture-legal for a prefill CUDA graph.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Heads per dot are a loop here, so a row block is what a program owns; the
# candidate block only has to be wide enough to amortise the k tile load.
_BLOCK_T = 64
_BLOCK_N = 128
_NUM_WARPS = 4
_NUM_STAGES = 3


@triton.jit
def _sparse_scores_kernel(
    Q,
    K,
    W,
    OUT,
    T,
    N,
    stride_qt,
    stride_qh,
    stride_kn,
    stride_wt,
    stride_ot,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    t_mask = offs_t < T
    n_mask = offs_n < N
    d_mask = offs_d < D

    # The candidate table is shared by every query row, so it is loaded once per
    # program and reused for all H heads.
    k = tl.load(
        K + offs_n[:, None] * stride_kn + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    k_t = tl.trans(k)

    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for h in tl.static_range(H):
        q = tl.load(
            Q + offs_t[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        dot = tl.dot(q, k_t, out_dtype=tl.float32)
        w = tl.load(W + offs_t * stride_wt + h, mask=t_mask, other=0.0)
        acc += tl.maximum(dot, 0.0) * w.to(tl.float32)[:, None]

    tl.store(
        OUT + offs_t[:, None] * stride_ot + offs_n[None, :],
        acc,
        mask=t_mask[:, None] & n_mask[None, :],
    )


def sparse_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``q [T, H, D]``, ``k [N, D]``, ``weights [T, H]`` -> ``[T, N]`` float32.

    ``weights`` is applied to the per-head score *after* the relu, matching the
    einsum form: a negative head contributes nothing, not a negative term.
    """
    t, h, d = q.shape
    n = k.shape[0]
    assert k.shape == (n, d), f"{k.shape=} against q {tuple(q.shape)}"
    assert weights.shape == (t, h), f"{weights.shape=}"
    q = q.contiguous()
    k = k.contiguous()
    if out is None:
        out = torch.empty((t, n), dtype=torch.float32, device=q.device)
    if t == 0 or n == 0:
        return out
    _sparse_scores_kernel[(triton.cdiv(t, _BLOCK_T), triton.cdiv(n, _BLOCK_N))](
        q,
        k,
        weights,
        out,
        t,
        n,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        out.stride(0),
        H=h,
        D=d,
        BLOCK_T=_BLOCK_T,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out
