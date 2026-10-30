"""Pin the DeepSeek-V4 KV byte layout that PD disaggregation depends on.

A PD pair splits one logical KV cache across two hosts: the prefill node writes
pages, transfers them as opaque bytes, and the decode node addresses those pages
with its own attention kernel. Nothing in the transfer interprets the bytes, so
both roles must agree on the exact page geometry. These tests pin the two halves
of that contract.

1. ``test_store_kernel_writes_the_flashmla_row_layout``: the store kernel puts a
   token's fp8 NoPE part, bf16 RoPE part and ue8m0 scales at the byte offsets
   ``sgl_kernel/deepseek_v4/kv_layout.cuh`` declares (``PagedKV::data_offset`` /
   ``scale_offset``), and pads the page to a 576-byte multiple. This is what a
   FlashMLA decode peer reads, so it is the layout the prefill side must emit.
2. ``test_reader_matches_the_independent_decode``: the portable reader used on
   GPUs without FlashMLA recovers bit-identical values from those same offsets.
3. ``test_pd_layout_pins_page_size_and_stride``: the PD handshake descriptor
   carries the page size and per-page stride, not just the layout name.
4. ``test_pd_layout_separates_the_sm120_direct_swa_pages`` /
   ``test_pd_layout_separates_differing_page_sizes``: configurations that keep
   the layout name but change the page geometry produce a different descriptor,
   so the handshake rejects the pair instead of decoding page N out of page M.
"""

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the KV store kernel needs CUDA"
)

PAGE_SIZE = 256
DIM_NOPE = 448
DIM_ROPE = 64
HEAD_DIM = DIM_NOPE + DIM_ROPE
NOPE_ROPE_BYTES = 576  # 448 fp8 + 64 bf16
SCALES_PER_TOKEN = 8  # 7 ue8m0 tiles + 1 pad byte


def _e4m3_byte_to_float(byte: int) -> float:
    """Decode one e4m3 byte: 1 sign, 4 exponent (bias 7), 3 mantissa bits.

    The format is e4m3fn, so only the all-ones mantissa at the maximum exponent
    is NaN; ``0x7E`` (448) is the largest finite value and is reachable here
    because the store scales into ``|x| <= 448``.
    """
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return float("nan")
    if exponent == 0:
        value = mantissa / 8 * 2.0**-6
    else:
        value = (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return -value if byte & 0x80 else value


def _decode_page_token(raw, page: int, slot: int, page_bytes: int) -> torch.Tensor:
    """Read one token straight from the documented offsets, in plain Python.

    Deliberately shares no code with the kernels under test: it is the
    independent statement of where the bytes are supposed to be.
    """
    base = page * page_bytes + slot * NOPE_ROPE_BYTES
    nope = raw[base : base + DIM_NOPE]
    rope = raw[base + DIM_NOPE : base + NOPE_ROPE_BYTES]
    scale_base = (
        page * page_bytes + PAGE_SIZE * NOPE_ROPE_BYTES + slot * SCALES_PER_TOKEN
    )
    scales = raw[scale_base : scale_base + SCALES_PER_TOKEN]

    out = torch.zeros(HEAD_DIM, dtype=torch.float32)
    tile = 64  # KVLayout.V4.tile_size: 448 / 64 == 7 tiles
    for t in range(DIM_NOPE // tile):
        scale = 2.0 ** (int(scales[t]) - 127)
        for j in range(tile):
            out[t * tile + j] = _e4m3_byte_to_float(int(nope[t * tile + j])) * scale
    out[DIM_NOPE:] = torch.frombuffer(bytearray(rope), dtype=torch.bfloat16).float()
    return out, scales


@requires_cuda
def test_store_kernel_writes_the_flashmla_row_layout():
    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    layout = KVLayout.V4
    pages = 4
    page_bytes = layout.page_bytes(PAGE_SIZE)
    # V4 is the only layout whose tile rows do not already land on page_align.
    assert page_bytes == -(-(PAGE_SIZE * layout.bytes_per_token) // 576) * 576
    assert page_bytes > PAGE_SIZE * layout.bytes_per_token

    cache = torch.zeros((pages, page_bytes), dtype=torch.uint8, device="cuda")
    # Locations cross page boundaries on purpose: a page-0-confined list agrees
    # under any page stride.
    locs = torch.tensor(
        [0, 7, PAGE_SIZE, PAGE_SIZE + 44, 2 * PAGE_SIZE - 1, 700, 4 * PAGE_SIZE - 1],
        dtype=torch.int64,
        device="cuda",
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        locs.numel(), HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=gen
    )

    fused_store_cache(
        input=x,
        cache=cache,
        indices=locs,
        page_size=PAGE_SIZE,
        type="flashmla",
        layout=layout,
    )
    torch.cuda.synchronize()
    raw = cache.cpu().numpy().reshape(-1)

    for i, loc in enumerate(locs.tolist()):
        page, slot = divmod(loc, PAGE_SIZE)
        decoded, scales = _decode_page_token(raw, page, slot, page_bytes)
        expected = x[i].float().cpu()

        # The RoPE half is stored as bf16, so it is exact.
        assert torch.equal(decoded[DIM_NOPE:], expected[DIM_NOPE:]), (
            f"loc={loc}: RoPE is not the bf16 tail at page byte {slot * NOPE_ROPE_BYTES + DIM_NOPE}"
        )

        # The NoPE half is fp8 with one power-of-two scale per 64 values, so it
        # can only be checked against the quantization step it lands in.
        nope_err = (decoded[:DIM_NOPE] - expected[:DIM_NOPE]).abs().max().item()
        amax = expected[:DIM_NOPE].abs().max().item()
        assert nope_err <= 0.07 * amax, (
            f"loc={loc}: NoPE error {nope_err} for amax {amax}"
        )

        # Seven live scale bytes, then the padding byte, which stays zero.
        assert int(scales[7]) == 0, f"loc={loc}: scale padding byte is not zero"


@requires_cuda
def test_reader_matches_the_independent_decode():
    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged_dsv4,
    )
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.layers.attention.deepseek_v4_backend import _paged_byte_rows

    layout = KVLayout.V4
    pages = 4
    page_bytes = layout.page_bytes(PAGE_SIZE)
    cache = torch.zeros((pages, page_bytes), dtype=torch.uint8, device="cuda")
    locs = torch.arange(0, pages * PAGE_SIZE, dtype=torch.int64, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(
        locs.numel(), HEAD_DIM, dtype=torch.bfloat16, device="cuda", generator=gen
    )
    fused_store_cache(
        input=x,
        cache=cache,
        indices=locs,
        page_size=PAGE_SIZE,
        type="flashmla",
        layout=layout,
    )
    torch.cuda.synchronize()
    raw = cache.cpu().numpy().reshape(-1)

    # The attention caller hands over the per-token view; the reader must take
    # the page stride from the tensor rather than from shape[-1].
    view = cache.view(torch.float8_e4m3fn)[:, : PAGE_SIZE * layout.bytes_per_token]
    view = view.view(pages, PAGE_SIZE, 1, layout.bytes_per_token)
    assert view.stride(0) == page_bytes, (
        "the pool pads the page, so the view is strided"
    )

    rows = _paged_byte_rows(view)
    assert rows.data_ptr() == cache.data_ptr(), "the row view must not repack"
    got = dequantize_k_cache_paged_dsv4(
        rows, locs.to(torch.int32), page_size=PAGE_SIZE, layout=layout
    ).squeeze(1)

    ref = torch.stack(
        [
            _decode_page_token(raw, *divmod(int(loc), PAGE_SIZE), page_bytes)[0]
            for loc in locs.tolist()
        ]
    ).to("cuda")
    assert torch.equal(got.float(), ref), "reader disagrees with the documented offsets"


def _fake_pool(layout_name, page_size, bytes_per_page, ratio_to_pool=None):
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    layout = KVLayout.parse(layout_name)
    pool = SimpleNamespace(
        kv_layout=layout, page_size=page_size, bytes_per_page_padded=bytes_per_page
    )
    return SimpleNamespace(
        uniform_fp8=False,
        kv_layout=layout,
        swa_kv_pool=pool,
        unified_kv_pool=None,
        compression_ratios=[0, 4, 128],
        kv_pools=ratio_to_pool or {},
    )


def test_pd_layout_pins_page_size_and_stride():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    pool = _fake_pool("v4", 256, 149760)
    descriptor = DeepSeekV4TokenToKVPool.pd_kv_layout(pool)
    assert descriptor.startswith("v4;"), descriptor
    assert "swa=v4@256:149760" in descriptor, descriptor

    # The same layout name with a different page stride must not compare equal:
    # every page after the first would be read from the wrong offset.
    other = _fake_pool("v4", 256, 149504)
    assert DeepSeekV4TokenToKVPool.pd_kv_layout(other) != descriptor


def test_pd_layout_separates_the_sm120_direct_swa_pages():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    # SGLANG_OPT_SM120_DIRECT_SWA_KV stores the 256-token logical page as
    # 64-token physical rows. Both nodes report layout "v4"; only the page
    # geometry distinguishes them.
    logical = _fake_pool("v4", 256, 149760)
    physical = _fake_pool("v4", 64, 37440)
    assert 37440 == -(-(64 * 584) // 576) * 576
    assert DeepSeekV4TokenToKVPool.pd_kv_layout(
        logical
    ) != DeepSeekV4TokenToKVPool.pd_kv_layout(physical)


def test_pd_layout_separates_differing_page_sizes():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    a = _fake_pool("v4", 128, 74880)
    b = _fake_pool("v4", 256, 149760)
    assert DeepSeekV4TokenToKVPool.pd_kv_layout(
        a
    ) != DeepSeekV4TokenToKVPool.pd_kv_layout(b)


def test_pd_layout_covers_the_compressed_pools():
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    # ``_make_kv_pool`` gives a compressed pool ``page_size // ratio`` rows, so
    # on the 256-token logical page these hold 64 and 2 tokens. Their padded
    # strides do not follow from the main pool's, which is why the descriptor
    # is per component rather than one number for the whole cache.
    c4 = SimpleNamespace(
        kv_layout=KVLayout.V4,
        page_size=64,
        bytes_per_page_padded=KVLayout.V4.page_bytes(64),
    )
    c128 = SimpleNamespace(
        kv_layout=KVLayout.V4,
        page_size=2,
        bytes_per_page_padded=KVLayout.V4.page_bytes(2),
    )
    pool = _fake_pool("v4", 256, 149760, ratio_to_pool={4: c4, 128: c128})
    descriptor = DeepSeekV4TokenToKVPool.pd_kv_layout(pool)
    assert "c4=v4@64:37440" in descriptor, descriptor
    assert "c128=v4@2:1728" in descriptor, descriptor
