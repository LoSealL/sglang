"""Tests for the fused relu-MQA scorer used by the DeepSeek V4.1 sparse index layers.

The reference is the einsum form in ``dsv41_sparse.scores``, but it accumulates
in bf16: ``einsum("bhd,nd->bhn")`` returns bf16, and the relu, the weight and
the 32-term head sum all run in bf16 as well. So the contract pinned here is
agreement with an **fp32** evaluation, and the torch form is only checked to
land within its own rounding error. ``test_beats_the_bf16_reference`` records
which of the two is closer to the truth, because the fused kernel changes the
arithmetic of a selection that feeds a top-k.
"""

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.kernels.ops.attention.dsv4.sparse_scores import sparse_scores


def _reference_fp32(q, k, weights):
    s = torch.einsum("bhd,nd->bhn", q.float(), k.float())
    return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=1)


def _reference_bf16(q, k, weights):
    s = torch.einsum("bhd,nd->bhn", q, k)
    return (s.relu() * weights.unsqueeze(-1)).sum(dim=1).float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [(64, 32, 128, 200), (130, 8, 128, 517), (1, 4, 64, 3)]
)
def test_matches_fp32_reference(dtype, shape):
    torch.manual_seed(0)
    t, h, d, n = shape
    q = torch.randn(t, h, d, dtype=dtype, device="cuda")
    k = torch.randn(n, d, dtype=dtype, device="cuda")
    w = torch.randn(t, h, dtype=dtype, device="cuda")

    want = _reference_fp32(q, k, w)
    got = sparse_scores(q, k, w)

    assert got.shape == (t, n)
    assert got.dtype == torch.float32
    assert got.device == q.device
    atol = 1e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(got, want, atol=atol, rtol=1e-3)


def test_beats_the_bf16_reference():
    """The kernel accumulates the head sum in fp32, so it is the *torch* form
    that moves: this is the reason the two disagree at all."""
    torch.manual_seed(0)
    t, h, d, n = 512, 32, 128, 1024
    q = torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(t, h, dtype=torch.bfloat16, device="cuda")

    truth = _reference_fp32(q, k, w)
    scale = truth.abs().mean()
    fused_err = (sparse_scores(q, k, w) - truth).abs().max() / scale
    torch_err = (_reference_bf16(q, k, w) - truth).abs().max() / scale
    assert fused_err < 1e-3, fused_err
    assert fused_err < torch_err / 1000, (fused_err, torch_err)


def test_head_weight_applies_after_relu():
    """A negative head must contribute nothing, not a negative term."""
    t, h, d, n = 4, 2, 32, 5
    q = torch.zeros(t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(n, d, dtype=torch.bfloat16, device="cuda")
    # head 0 scores positive, head 1 negative, at every row and column.
    q[..., 0] = 1.0
    k[:, 0] = 1.0
    q[:, 1, 0] = -1.0
    w = torch.ones(t, h, dtype=torch.bfloat16, device="cuda")
    got = sparse_scores(q, k, w)
    torch.testing.assert_close(got, torch.ones(t, n, device="cuda"))


def test_empty_shapes():
    q = torch.empty(0, 4, 32, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(7, 32, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(0, 4, dtype=torch.bfloat16, device="cuda")
    assert sparse_scores(q, k, w).shape == (0, 7)

    q = torch.randn(3, 4, 32, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(0, 32, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(3, 4, dtype=torch.bfloat16, device="cuda")
    assert sparse_scores(q, k, w).shape == (3, 0)


def test_out_argument_is_written_in_place():
    torch.manual_seed(0)
    t, h, d, n = 33, 4, 64, 71
    q = torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(t, h, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(t, n, dtype=torch.float32, device="cuda")
    got = sparse_scores(q, k, w, out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got, _reference_fp32(q, k, w), atol=1e-2, rtol=1e-3)
