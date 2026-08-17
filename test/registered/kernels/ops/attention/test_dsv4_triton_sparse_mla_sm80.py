# test/registered/kernels/ops/attention/test_dsv4_triton_sparse_mla_sm80.py
"""sm80 DSV4 sparse MLA over the packed paged KV vs torch-softmax reference
(needs 1 GPU). Run: pytest -q <this file>."""

import math
import types

import torch
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import SetKAndS
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.kernels.ops.attention.dsv4.triton_sparse_mla import triton_sparse_mla_fwd


def _make_pool(rows, page, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(rows, 512, dtype=torch.bfloat16, device="cuda")
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k)
    raw = page * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bpp = -(-raw // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES  # ceil to a 576 multiple
    buf = torch.zeros(rows // page, bpp, dtype=torch.uint8, device="cuda")
    locs = torch.arange(rows, dtype=torch.int32, device="cuda")
    # Brief bug: SetKAndS dereferences pool.page_size, so pool=None crashes;
    # a shim carries the page size.
    SetKAndS.execute(types.SimpleNamespace(page_size=page), buf, locs, pack)
    # The ref must attend over the dequantized (quantization round-tripped)
    # rows — the kernel reads the packed pool, not the original bf16 k.
    deq = dequantize_k_cache_paged(buf, locs, page).squeeze(1)
    return deq, buf


def _ref_attn(q, k_rows, sm_scale, sink, idx, length):
    T, H, _ = q.shape
    out = torch.empty_like(q)
    for t in range(T):
        ids = idx[t][: int(length[t])].long()
        kv = k_rows[ids].float()  # [L, 512]
        logits = (q[t].float() @ kv.T) * sm_scale  # [H, L]
        for h in range(H):
            s = float(sink[h])
            if s > float("-inf"):
                # Gemma-style sink: one extra zero-value logit in the softmax
                # (brief bug fixed: the sink's exp(s - m) belongs in the
                # denominator; a hardcoded 1.0 left the ref unnormalized).
                m = max(s, logits[h].max().item())
                p = torch.exp(logits[h] - m)
                den = p.sum() + math.exp(s - m)
            else:
                m = logits[h].max().item()
                p = torch.exp(logits[h] - m)
                den = p.sum()
            out[t, h] = (p @ kv / max(den, 1e-20)).bfloat16()
    return out


def test_sink_and_segments():
    T, H = 9, 64
    sm = 512**-0.5
    swa_k, swa_buf = _make_pool(256, 128, seed=1)
    comp_k, comp_buf = _make_pool(512, 64, seed=2)
    q = torch.randn(T, H, 512, dtype=torch.bfloat16, device="cuda")
    sink = torch.full((H,), 1.3, device="cuda")
    sink[3] = float("-inf")
    W = 128
    swa_idx = torch.randint(0, 256, (T, W), device="cuda", dtype=torch.int32)
    swa_len = torch.randint(1, W + 1, (T,), device="cuda", dtype=torch.int32)
    for t in range(T):  # -1 pad tails
        swa_idx[t, int(swa_len[t]) :] = -1
    K = 512
    comp_idx = torch.randint(0, 512, (T, K), device="cuda", dtype=torch.int32)
    comp_len = torch.randint(0, K + 1, (T,), device="cuda", dtype=torch.int32)
    for t in range(T):
        comp_idx[t, int(comp_len[t]) :] = -1
    out = torch.empty_like(q)
    triton_sparse_mla_fwd(
        q,
        out,
        swa_buf,
        swa_idx,
        swa_len,
        comp_buf,
        comp_idx,
        comp_len,
        sm,
        sink,
        swa_page=128,
        comp_page=64,
    )
    kv = torch.cat([swa_k, comp_k])
    ref = torch.empty_like(q)
    for t in range(T):
        ids = torch.cat(
            [
                swa_idx[t][: int(swa_len[t])].long(),
                256 + comp_idx[t][: int(comp_len[t])].long(),
            ]
        )
        lens = ids.new_tensor([ids.numel()])
        ref[t] = _ref_attn(q[t : t + 1], kv, sm, sink, ids[None], lens)[0]
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


def test_swa_only():
    T, H = 5, 64
    sm = 512**-0.5
    swa_k, swa_buf = _make_pool(128, 128, seed=3)
    q = torch.randn(T, H, 512, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(H, device="cuda")
    idx = torch.randint(0, 128, (T, 128), device="cuda", dtype=torch.int32)
    ln = torch.full((T,), 64, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    triton_sparse_mla_fwd(
        q, out, swa_buf, idx, ln, None, None, None, sm, sink, swa_page=128
    )
    ref = _ref_attn(q, swa_k, sm, sink, idx, ln)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


def test_pool_roundtrip_bit_exact():
    """Quantize -> SetKAndS pool -> Triton dequant is bit-equal to the pure-torch
    ref: pins cross-arch bit-exactness without needing Hopper."""
    torch.manual_seed(7)
    rows, page = 384, 128
    k = torch.randn(rows, 512, dtype=torch.bfloat16, device="cuda")
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k)
    raw = page * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bpp = -(-raw // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES  # ceil to a 576 multiple
    buf = torch.zeros(rows // page, bpp, dtype=torch.uint8, device="cuda")
    locs = torch.arange(rows, dtype=torch.int32, device="cuda")
    SetKAndS.execute(types.SimpleNamespace(page_size=page), buf, locs, pack)
    out = dequantize_k_cache_paged(buf, locs, page)
    ref = dequantize_k_cache_paged_ref(buf, locs, page)
    assert torch.equal(out, ref)
