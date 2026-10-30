"""Unit tests for the portable Triton sparse-MLA attention path.

``flash_mla_sparse_fwd`` / ``flash_mla_with_kvcache`` ship sm_90a and sm_100
images only, so pre-SM90 GPUs (an A100, for instance) have no FlashMLA at all.
``sglang.kernels.ops.attention.sparse_mla_prefill_sm80`` provides a plain
``tl.dot``-free implementation that runs on any Triton target; these tests pin
the contract it has to satisfy:

1. ``test_matches_exact_softmax``: agreement with an fp32 reference on dense
   index lists.
2. ``test_negative_entries_are_dropped``: ``-1`` marks an unused slot and must
   not leak a KV row into the softmax.
3. ``test_all_invalid_row_is_zero``: a row with no live slot contributes zeros
   rather than NaN.
4. ``test_attention_sink``: the sink only moves the normalizer.
5. ``test_topk_length_bounds_the_prefix``: entries past ``topk_length`` are
   padding even when they hold a valid slot id.
6. ``test_decode_dequant_reads_the_paged_view``: feeding the ``(pages,
   page_size, 1, bytes)`` view the attention caller passes straight to the
   dequantizer silently produces zeros, because its page stride is read from
   ``shape[-1]``; the flattening helper that decode applies must recover the
   real page layout.
"""

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sparse MLA Triton kernel requires CUDA"
)

DEVICE = "cuda"
HEAD_DIM = 512
LOG2E = 1.4426950408889634


def _kernel():
    from sglang.kernels.ops.attention.sparse_mla_prefill_sm80 import (
        flash_mla_sparse_prefill_triton,
    )

    return flash_mla_sparse_prefill_triton


def _reference(q, kv, indices, sm_scale, d_v, attn_sink):
    """Per-row fp32 softmax over the live slots, sink folded into the normalizer."""
    num_q, _, _ = q.shape
    flat = indices.reshape(num_q, -1)
    live = flat >= 0
    rows = flat.clamp(min=0).long()
    k = kv[rows].float()

    scores = torch.einsum("qhd,qtd->qht", q.float(), k) * sm_scale
    scores = scores.masked_fill(~live[:, None, :], float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    if attn_sink is not None:
        sink = attn_sink.reshape(1, -1).float()
        lse = torch.logsumexp(torch.stack([lse, sink.expand_as(lse)], dim=0), dim=0)
    # A row with no live slot has no finite logsumexp; it must yield zeros.
    lonely = ~torch.isfinite(lse)
    lse = torch.where(lonely, torch.zeros_like(lse), lse)

    weights = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("qht,qtd->qhd", weights, k[..., :d_v])
    return torch.where(lonely.unsqueeze(-1), torch.zeros_like(out), out)


def _relative_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _make_inputs(num_q=8, num_heads=8, num_kv=256, topk=128, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(num_q, num_heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(num_kv, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    indices = torch.randint(
        0, num_kv, (num_q, 1, topk), device=DEVICE, dtype=torch.int32
    )
    return q, kv, indices


def test_matches_exact_softmax():
    kernel = _kernel()
    q, kv, indices = _make_inputs(topk=128)
    sink = torch.randn(8, device=DEVICE, dtype=torch.float32)
    sm_scale = HEAD_DIM**-0.5

    out, _, _ = kernel(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=HEAD_DIM,
        attn_sink=sink,
        topk_length=None,
    )

    ref = _reference(q, kv, indices, sm_scale, HEAD_DIM, sink)
    assert torch.isfinite(out).all()
    # bf16 accumulation over a 512-wide dot product; a fraction of a percent is
    # the expected agreement with the fp32 reference.
    assert _relative_error(out, ref) < 2e-2


def test_negative_entries_are_dropped():
    kernel = _kernel()
    q, kv, indices = _make_inputs(num_q=4, num_heads=4, topk=64)
    # Keep 10 real slots per row and blank the rest. The kernel must agree with
    # a reference built from the live slots alone, so a leaked row 0 -- what
    # folding -1 onto slot 0 would produce -- shows up as a large error.
    keep = 10
    live = torch.randint(0, 256, (4, 1, keep), device=DEVICE, dtype=torch.int32)
    pad = torch.full_like(indices, -1)
    pad[:, :, :keep] = live
    indices = pad

    sink = torch.zeros(4, device=DEVICE, dtype=torch.float32)
    sm_scale = HEAD_DIM**-0.5
    out, _, _ = kernel(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=HEAD_DIM,
        attn_sink=sink,
        topk_length=None,
    )

    ref = _reference(q, kv, indices, sm_scale, HEAD_DIM, sink)
    assert _relative_error(out, ref) < 2e-2


def test_all_invalid_row_is_zero():
    kernel = _kernel()
    q, kv, indices = _make_inputs(num_q=4, num_heads=4, topk=32)
    indices[0] = -1

    out, _, _ = kernel(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=HEAD_DIM**-0.5,
        d_v=HEAD_DIM,
        attn_sink=torch.zeros(4, device=DEVICE, dtype=torch.float32),
        topk_length=None,
    )

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[0]) == 0
    assert torch.count_nonzero(out[1:]) > 0


def test_attention_sink():
    kernel = _kernel()
    q, kv, indices = _make_inputs(num_q=2, num_heads=4, topk=64)
    sm_scale = HEAD_DIM**-0.5

    no_sink, _, _ = kernel(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=HEAD_DIM,
        attn_sink=None,
        topk_length=None,
    )
    # A sink below every score only renormalizes, so the output shrinks toward
    # zero without changing direction. Keep it inside the score range: an
    # extreme sink underflows every weight to zero, which is the sink's own
    # arithmetic rather than a kernel property worth asserting on.
    big_sink = torch.full((4,), -4.0, device=DEVICE, dtype=torch.float32)
    with_sink, _, _ = kernel(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=HEAD_DIM,
        attn_sink=big_sink,
        topk_length=None,
    )

    ref = _reference(q, kv, indices, sm_scale, HEAD_DIM, big_sink)
    assert _relative_error(with_sink, ref) < 2e-2
    # The sink only adds normalizer mass, so it must change the result...
    assert not torch.allclose(with_sink, no_sink)
    # ...always downward, never growing the attention output.
    assert with_sink.norm() <= no_sink.norm() * 1.01


def test_topk_length_bounds_the_prefix():
    kernel = _kernel()
    q, kv, indices = _make_inputs(num_q=4, num_heads=4, topk=64)
    sm_scale = HEAD_DIM**-0.5
    sink = torch.zeros(4, device=DEVICE, dtype=torch.float32)

    # Row r keeps r + 1 slots live but carries valid ids in every lane, so only
    # topk_length can exclude the tail. Truncating the ids to -1 past the same
    # point gives the reference for "those lanes do not exist".
    lengths = torch.arange(1, 5, device=DEVICE, dtype=torch.int32)
    truncated = indices.clone()
    for r in range(4):
        truncated[r, :, int(lengths[r]) :] = -1

    bounded, _, _ = kernel(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=HEAD_DIM,
        attn_sink=sink,
        topk_length=lengths,
    )
    ref = _reference(q, kv, truncated, sm_scale, HEAD_DIM, sink)

    assert _relative_error(bounded, ref) < 2e-2


def _paged_kv_bytes(pages, page_size, layout):
    """Fill a page buffer shaped like the real pool: padded stride, live content.

    ``create_buffer`` allocates ``layout.page_bytes(page_size)`` bytes per page,
    which for V4 pads the live 584-byte rows up to a 576-byte multiple. The
    padding is dead space, so leaving it out would hide every bug that depends
    on the page stride.
    """
    page_bytes = layout.page_bytes(page_size)
    assert page_bytes >= page_size * layout.bytes_per_token
    # Per-page RNG: identical pages would hide a page-stride mistake, because
    # reading the wrong page would still return the right bytes.
    gen = torch.Generator().manual_seed(1234 + pages * 7919 + page_size)
    raw = torch.zeros((pages, page_bytes), dtype=torch.uint8)
    data = page_size * layout.data_bytes
    live = page_size * layout.bytes_per_token
    raw[:, : page_size * 448] = torch.randint(
        0, 0x38, (pages, page_size * 448), dtype=torch.uint8, generator=gen
    )
    raw[:, page_size * 448 : data] = torch.randint(
        0, 0x3B, (pages, data - page_size * 448), dtype=torch.uint8, generator=gen
    )
    # One e8m0 byte per scale row; 127 is 2^0.
    scale_row = live - data
    n_rows = scale_row // 8
    raw[:, data : data + n_rows * 8] = 127
    return raw


def test_decode_dequant_reads_the_paged_view():
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        KVLayout,
        dequantize_k_cache_paged_dsv4,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend import _paged_byte_rows

    layout = KVLayout.V4
    pages, page_size = 4, 256
    raw = _paged_kv_bytes(pages, page_size, layout).cuda()
    # Slot ids must cross page boundaries: an id list confined to page 0 reads
    # the same page under either stride and hides the mistake.
    ids = torch.arange(0, pages * page_size, device=DEVICE, dtype=torch.int32)

    # What attention hands over: the per-token view of the same storage.
    view = raw.view(torch.float8_e4m3fn)[:, : page_size * layout.bytes_per_token]
    view = view.view(pages, page_size, 1, layout.bytes_per_token)
    # The pool pads a V4 page to a 576-byte multiple, so the per-token view is
    # strided rather than contiguous -- that gap is the page stride.
    assert view.stride(0) == layout.page_bytes(page_size)
    assert view.stride(0) > page_size * layout.bytes_per_token

    reference = dequantize_k_cache_paged_dsv4(
        raw.view(torch.float8_e4m3fn), ids, page_size=page_size
    )
    assert reference.abs().max() > 0

    # Read through the view directly: shape[-1] is the token width, so the
    # kernel steps page_index * 584 and lands in unrelated bytes. The
    # dequantizer demands a contiguous cache, and a fresh copy of the view is
    # contiguous while keeping the misleading trailing dimension -- that is
    # exactly the shape the decode caller used to pass.
    naive = dequantize_k_cache_paged_dsv4(view.contiguous(), ids, page_size=page_size)

    rows = _paged_byte_rows(view)
    fixed = dequantize_k_cache_paged_dsv4(rows, ids, page_size=page_size)

    # The row view keeps the pool's page stride, which is the stride the
    # FlashMLA paths use and the one PD moves pages by, so it reproduces the
    # reference exactly without repacking the pages.
    assert fixed.shape == reference.shape
    assert torch.equal(fixed, reference)
    assert rows.data_ptr() == raw.data_ptr(), "the row view must not copy"

    # Reading the view directly treats shape[-1] (the token width) as the page
    # stride, so it addresses page_index * 584 instead of page_index *
    # page_size * 584 and returns unrelated bytes.
    assert naive.shape == reference.shape
    assert not torch.equal(naive, reference)
