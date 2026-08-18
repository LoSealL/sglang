# test/registered/kernels/ops/attention/test_dsv4_triton_fp8_mqa_logits_sm80.py
"""sm80 DSV4 indexer fp8 MQA logits vs torch reference. Run: pytest -q <this file> (needs 1 GPU)."""
import torch
from sglang.kernels.ops.attention.dsv4.triton_fp8_mqa_logits_cuda import (
    fp8_mqa_logits_cuda,
    paged_fp8_mqa_logits_cuda,
    sglang_paged_mqa_logits,
)

H, D = 64, 128


def _ref(q, k, kv_scales, weights, starts, ends, N):
    qf = q.float(); kf = k.float()
    logits = torch.full((q.shape[0], N), -float("inf"), device=q.device)
    for m in range(q.shape[0]):
        s, e = int(starts[m]), min(int(ends[m]), N)
        if e > s:
            dots = (torch.relu((qf[m] @ kf[s:e].T) * kv_scales[s:e]) * weights[m][:, None]).sum(0)
            logits[m, s:e] = dots
    return logits


def test_contiguous():
    torch.manual_seed(0)
    M, N = 37, 500
    q = (torch.randn(M, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    k = (torch.randn(N, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    sc = torch.rand(N, device="cuda") + 0.5
    w = torch.randn(M, H, device="cuda")
    starts = torch.randint(0, 5, (M,), device="cuda", dtype=torch.int32)
    ends = starts + torch.randint(0, 200, (M,), device="cuda", dtype=torch.int32)
    out = fp8_mqa_logits_cuda(q, k, sc, w, starts, ends)
    # fp8->bf16 dot vs fp32 ref: rel tolerance is loose, layout correctness is the point
    torch.testing.assert_close(
        out[:, : int(ends.min())], _ref(q, k, sc, w, starts, ends, N)[:, : int(ends.min())],
        atol=0.5, rtol=0.05,
    )
    assert torch.isinf(out[0, int(ends[0]):]).all()


def test_paged_garbage_tail():
    """Logits beyond ctx are unwritten; garbage tail must not crash; [0,ctx) must match ref."""
    torch.manual_seed(0)
    B, block, pages, ctx = 5, 64, 9, 400
    q = (torch.randn(B, 1, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    # Byte 0x7F is NaN in e4m3fn; garbage values must stay finite for the ref
    # decode, garbage scales sane so magnitudes stay comparable. Block layout
    # per page: [block x D value bytes | block x 4B fp32 scales].
    buf = torch.randint(0, 254, (pages, block * (D + 4)), dtype=torch.uint8, device="cuda")
    buf[buf == 127] = 126
    buf[:, block * D :] = (torch.rand(pages, block, device="cuda") + 0.5).view(torch.uint8)
    w = torch.randn(B, H, device="cuda")
    lens = torch.tensor([[ctx], [7], [64], [65], [399]], device="cuda", dtype=torch.int32)
    bt = torch.randint(0, pages, (B, 16), device="cuda", dtype=torch.int32)
    out = paged_fp8_mqa_logits_cuda(q, buf, w, lens, bt, max_len=512)
    assert out.shape == (B, 512)
    # decode against a torch decode of the same u8 cache for row 0
    u8 = buf.view(-1)
    page_bytes = block * (D + 4)

    def dec_row(b, n):
        base = bt[b, n // block].item() * page_bytes
        vals = u8[base + (n % block) * D : base + (n % block) * D + D]
        scale = u8[base + block * D + (n % block) * 4 : base + block * D + (n % block) * 4 + 4]
        return vals.view(torch.float8_e4m3fn).float() * scale.view(torch.float32).float().item()

    k0 = torch.stack([dec_row(0, n) for n in range(ctx)])
    ref0 = (torch.relu(q[0, 0].float() @ k0.T) * w[0][:, None]).sum(0)
    torch.testing.assert_close(out[0, :ctx], ref0, atol=0.5, rtol=0.05)


def test_adapter_sglang_paged_mqa_logits():
    """DeepGEMM-signature adapter: same raw pool buffer, same kernel → bitwise-equal logits."""
    torch.manual_seed(0)
    B, block, pages = 5, 64, 9
    q = (torch.randn(B, 1, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    # Pool buffer like get_index_k_with_scale_buffer: raw [pages, 64*(D+4)]
    # uint8 in sglang block layout (values block, then scales block, per page).
    buf = torch.randint(0, 254, (pages, block * (D + 4)), dtype=torch.uint8, device="cuda")
    buf[buf == 127] = 126
    buf[:, block * D :] = (torch.rand(pages, block, device="cuda") + 0.5).view(torch.uint8)
    w = torch.randn(B, H, device="cuda")
    lens = torch.tensor([[400], [7], [64], [65], [399]], device="cuda", dtype=torch.int32)
    bt = torch.randint(0, pages, (B, 16), device="cuda", dtype=torch.int32)
    out = sglang_paged_mqa_logits(q, buf, w, lens, bt, None, 512)
    assert out.shape == (B, 512) and out.dtype == torch.float32
    ref = paged_fp8_mqa_logits_cuda(q, buf, w, lens, bt, max_len=512)
    # Tail beyond ctx is undefined (torch.empty); written prefix must be exact.
    for m in range(B):
        assert torch.equal(out[m, : int(lens[m, 0])], ref[m, : int(lens[m, 0])])


def test_paged_sglang_block_layout():
    """Kernel must read sglang's BLOCK-layout indexer cache (store.cuh): each
    64-row page = [64 x 128 fp8 value rows | 64 x 4B fp32 scales] (8448B), NOT
    vLLM-style interleaved 132B rows. NaN scales just past each ctx boundary
    must not leak into [0, ctx)."""
    torch.manual_seed(0)
    B, block, pages = 5, 64, 40
    ctxs = [1, 63, 64, 65, 400]

    # fp8-quantize random rows with per-row scales, written in block layout
    x = torch.randn(pages, block, D, device="cuda")
    sc = (x.abs().amax(-1) / 448).clamp(min=1e-4)
    vals = (x / sc[..., None]).to(torch.float8_e4m3fn)
    buf = torch.zeros(pages, block * (D + 4), dtype=torch.uint8, device="cuda")
    buf[:, : block * D] = vals.view(torch.uint8).reshape(pages, -1)
    buf[:, block * D :] = sc.contiguous().view(torch.uint8).reshape(pages, -1)

    # Per-row dedicated page ranges; NaN fp32 scales just past each ctx.
    bt = (
        torch.arange(B, device="cuda", dtype=torch.int32)[:, None] * 8
        + torch.arange(8, device="cuda", dtype=torch.int32)[None, :]
    )
    nan = torch.tensor([0x00, 0x00, 0xC0, 0x7F], dtype=torch.uint8, device="cuda")
    for b, ctx in enumerate(ctxs):
        for off in (ctx, ctx + 1):
            page = bt[b, off // block].item()
            s0 = block * D + (off % block) * 4
            buf[page, s0 : s0 + 4] = nan

    # The DeepGEMM-style [pages, 64, 1, 132] view is nominal-shape only: its
    # implied 132B value rows (stride(1)==132) do not match the real 128B rows.
    assert buf.view(pages, block, 1, D + 4).squeeze(2).stride(1) == D + 4 != D

    q = (torch.randn(B, 1, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    w = torch.randn(B, H, device="cuda")
    lens = torch.tensor([[c] for c in ctxs], device="cuda", dtype=torch.int32)

    vals_dec = buf[:, : block * D].reshape(pages, block, D).view(torch.float8_e4m3fn).float()
    sc_dec = buf[:, block * D :].view(torch.float32)
    outs = [
        paged_fp8_mqa_logits_cuda(q, buf, w, lens, bt, max_len=512),
        sglang_paged_mqa_logits(q, buf, w, lens, bt, None, 512),
    ]
    for out in outs:
        for b, ctx in enumerate(ctxs):
            n = torch.arange(ctx, device="cuda")
            k = vals_dec[bt[b, n // block], n % block]
            s = sc_dec[bt[b, n // block], n % block]
            # kernel decodes fp8 -> bf16 before the dot; mirror that in the ref
            qf = q[b, 0].float().to(torch.bfloat16).float()
            kb = k.to(torch.bfloat16).float()
            ref = (torch.relu(torch.einsum("hd,nd->hn", qf, kb)) * s * w[b][:, None]).sum(0)
            torch.testing.assert_close(out[b, :ctx], ref, rtol=2e-3, atol=2e-3)


def _block_layout_pool(pages, block):
    """fp8 values + scales written in sglang block layout (store.cuh)."""
    x = torch.randn(pages, block, D, device="cuda")
    sc = (x.abs().amax(-1) / 448).clamp(min=1e-4)
    vals = (x / sc[..., None]).to(torch.float8_e4m3fn)
    buf = torch.zeros(pages, block * (D + 4), dtype=torch.uint8, device="cuda")
    buf[:, : block * D] = vals.view(torch.uint8).reshape(pages, -1)
    buf[:, block * D :] = sc.contiguous().view(torch.uint8).reshape(pages, -1)
    return buf


def _block_layout_ref(buf, bt, b, ctx, q, w, block):
    vals_dec = buf[:, : block * D].reshape(-1, block, D).view(torch.float8_e4m3fn).float()
    sc_dec = buf[:, block * D :].view(torch.float32)
    n = torch.arange(ctx, device="cuda")
    k = vals_dec[bt[b, n // block], n % block]
    s = sc_dec[bt[b, n // block], n % block]
    qf = q[b, 0].float().to(torch.bfloat16).float()
    kb = k.to(torch.bfloat16).float()
    return (torch.relu(torch.einsum("hd,nd->hn", qf, kb)) * s * w[b][:, None]).sum(0)


def test_paged_splitkv_parallel():
    """2D grid (rows, KV splits): multi-CTA execution must be bitwise-identical
    to single-split (sequential) execution and match the fp32 ref. Covers split
    boundaries on/off SPLIT_KV and block_size multiples (255/256/257) and empty
    splits when ctx < max_len."""
    torch.manual_seed(0)
    block, B = 64, 5
    ctxs = [1, 255, 256, 257, 1000]
    pages_per_row, max_len = 16, 1024
    buf = _block_layout_pool(B * pages_per_row, block)
    bt = (
        torch.arange(B, device="cuda", dtype=torch.int32)[:, None] * pages_per_row
        + torch.arange(pages_per_row, device="cuda", dtype=torch.int32)[None, :]
    )
    q = (torch.randn(B, 1, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    w = torch.randn(B, H, device="cuda")
    lens = torch.tensor([[c] for c in ctxs], device="cuda", dtype=torch.int32)

    seq = paged_fp8_mqa_logits_cuda(q, buf, w, lens, bt, max_len, split_kv=max_len)
    par = paged_fp8_mqa_logits_cuda(q, buf, w, lens, bt, max_len, split_kv=256)
    for b, ctx in enumerate(ctxs):
        assert torch.equal(seq[b, :ctx], par[b, :ctx])
        torch.testing.assert_close(
            par[b, :ctx],
            _block_layout_ref(buf, bt, b, ctx, q, w, block),
            rtol=2e-3,
            atol=2e-3,
        )

    # Large ctx, minimal pool: 40000 rows not divisible by SPLIT_KV=512.
    pages, big_ctx, big_max = 640, 40000, 40960
    buf2 = _block_layout_pool(pages, block)
    bt2 = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
    q2 = (torch.randn(1, 1, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    w2 = torch.randn(1, H, device="cuda")
    lens2 = torch.tensor([[big_ctx]], device="cuda", dtype=torch.int32)
    seq2 = paged_fp8_mqa_logits_cuda(
        q2, buf2, w2, lens2, bt2, big_max, split_kv=big_max
    )
    par2 = paged_fp8_mqa_logits_cuda(q2, buf2, w2, lens2, bt2, big_max, split_kv=512)
    assert torch.equal(seq2[0, :big_ctx], par2[0, :big_ctx])

    # Perf sanity at decode-scale ctx: < 1.5 ms (single-CTA baseline is 3.12 ms).
    bench_ctx = 24576
    lens3 = torch.tensor([[bench_ctx]], device="cuda", dtype=torch.int32)
    start_e, end_e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        paged_fp8_mqa_logits_cuda(q2, buf2, w2, lens3, bt2, bench_ctx)
    times = []
    for _ in range(50):
        start_e.record()
        paged_fp8_mqa_logits_cuda(q2, buf2, w2, lens3, bt2, bench_ctx)
        end_e.record()
        torch.cuda.synchronize()
        times.append(start_e.elapsed_time(end_e))
    assert sorted(times)[25] < 1.5, f"kernel too slow: {sorted(times)[25]:.3f} ms"


def test_sm80_metadata_gating_and_kernel_importable(monkeypatch):
    """sm80: Triton adapter is the dispatch target; metadata avoids deep_gemm/topk_v2."""
    from sglang.srt.layers.attention.dsv4 import indexer as indexer_mod
    from sglang.srt.layers.attention.dsv4 import metadata as metadata_mod

    monkeypatch.setattr(indexer_mod, "is_sm80_supported", lambda: True)
    monkeypatch.setattr(metadata_mod, "is_sm80_supported", lambda: True)

    # Same import path the sm80 dispatch branch in forward_c4_indexer uses.
    from sglang.kernels.ops.attention.dsv4.triton_fp8_mqa_logits_cuda import (
        sglang_paged_mqa_logits,
    )

    assert callable(sglang_paged_mqa_logits)

    m = metadata_mod.PagedIndexerMetadata(
        page_size=256,
        page_table=torch.zeros((1, 16), dtype=torch.int32),
        c4_seq_lens=torch.ones((1, 1), dtype=torch.int32),
    )
    assert m.deep_gemm_metadata is None
    assert m.topk_metadata.numel() == 0
