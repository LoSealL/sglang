"""Benchmark the fused relu-MQA scorer (``sparse_scores``) against the einsum form.

``dsv41_sparse.scores`` materialises ``[T, H, N]`` and walks it three times; the
fused kernel keeps the head sum in fp32 registers and writes only ``[T, N]``.
The line axis is which of the two runs, so the ratio is the point of the run.

Shape axes follow a 4k prefill: ``tokens`` is the query rows of one source
layer, ``heads`` the index heads per rank (32 at TP8 on the deployment config),
``head_dim`` 128 and ``candidates`` the candidate rows the sparse table holds.

Traffic, both counted as the minimum each form must move:

  - fused: ``q`` + ``k`` + ``weights`` + the ``[T, N]`` fp32 result
  - einsum: the ``[T, H, N]`` bf16 intermediate written and read by the relu,
    the weight multiply and the head sum, plus the same result
"""

import torch
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.attention.dsv4.sparse_scores import sparse_scores
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEAD_DIM = 128


def _einsum_scores(q, k, weights):
    s = torch.einsum("bhd,nd->bhn", q, k)
    return (s.relu() * weights.unsqueeze(-1)).sum(dim=1).float()


@marker.parametrize("candidates", [512, 2002], [2002])
@marker.parametrize("heads", [8, 32], [32])
@marker.parametrize("tokens", [1024, 2048, 4096], [4096])
@marker.benchmark("impl", ["fused", "einsum"])
def benchmark(tokens: int, heads: int, candidates: int, impl: str):
    torch.manual_seed(0)
    q = create_random(tokens, heads, HEAD_DIM, dtype=torch.bfloat16)
    k = create_random(candidates, HEAD_DIM, dtype=torch.bfloat16)
    weights = create_random(tokens, heads, dtype=torch.bfloat16)
    # Both forms allocate their result, so the line compares the same work.
    fn = sparse_scores if impl == "fused" else _einsum_scores

    footprint = q.nbytes + k.nbytes + tokens * candidates * 4
    if impl == "einsum":
        # The intermediate is written once, then read and written by each of the
        # three passes over it before the head sum reads it out.
        footprint += tokens * heads * candidates * 2 * 5

    return marker.do_bench(
        fn,
        input_args=(q, k, weights),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=footprint,
    )


if __name__ == "__main__":
    benchmark.run()
