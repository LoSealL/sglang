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
    # decode, garbage scales sane so magnitudes stay comparable.
    kv = torch.randint(0, 254, (pages, block, D + 4), dtype=torch.uint8, device="cuda")
    kv[kv == 127] = 126
    kv[..., D:] = (torch.rand(pages, block, 1, device="cuda") + 0.5).view(torch.uint8)
    w = torch.randn(B, H, device="cuda")
    lens = torch.tensor([[ctx], [7], [64], [65], [399]], device="cuda", dtype=torch.int32)
    bt = torch.randint(0, pages, (B, 16), device="cuda", dtype=torch.int32)
    out = paged_fp8_mqa_logits_cuda(q, kv, w, lens, bt, max_len=512)
    assert out.shape == (B, 512)
    # decode against a torch decode of the same u8 cache for row 0
    u8 = kv.view(-1)
    def dec_row(b, n):
        blk = bt[b, n // block].item()
        off = (blk * block + n % block) * (D + 4)
        vals = u8[off : off + D].view(torch.float8_e4m3fn).float()
        scale = u8[off + D : off + D + 4].view(torch.float32).float().item()
        return vals * scale
    k0 = torch.stack([dec_row(0, n) for n in range(ctx)])
    ref0 = (torch.relu(q[0, 0].float() @ k0.T) * w[0][:, None]).sum(0)
    torch.testing.assert_close(out[0, :ctx], ref0, atol=0.5, rtol=0.05)


def test_adapter_sglang_paged_mqa_logits():
    """DeepGEMM-signature adapter: same pool, same kernel → bitwise-equal logits."""
    torch.manual_seed(0)
    B, block, pages = 5, 64, 9
    q = (torch.randn(B, 1, H, D, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    # Pool laid out like indexer.py: flat [pages, 64*132], viewed [pages, 64, 1, 132].
    buf = torch.randint(0, 254, (pages, block * (D + 4)), dtype=torch.uint8, device="cuda")
    buf[buf == 127] = 126
    buf.view(pages, block, D + 4)[..., D:] = (
        torch.rand(pages, block, 1, device="cuda") + 0.5
    ).view(torch.uint8)
    w = torch.randn(B, H, device="cuda")
    lens = torch.tensor([[400], [7], [64], [65], [399]], device="cuda", dtype=torch.int32)
    bt = torch.randint(0, pages, (B, 16), device="cuda", dtype=torch.int32)
    out = sglang_paged_mqa_logits(
        q, buf.view(pages, block, 1, D + 4), w, lens, bt, None, 512
    )
    assert out.shape == (B, 512) and out.dtype == torch.float32
    ref = paged_fp8_mqa_logits_cuda(
        q, buf.view(pages, block, D + 4), w, lens, bt, max_len=512
    )
    # Tail beyond ctx is undefined (torch.empty); written prefix must be exact.
    for m in range(B):
        assert torch.equal(out[m, : int(lens[m, 0])], ref[m, : int(lens[m, 0])])


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
