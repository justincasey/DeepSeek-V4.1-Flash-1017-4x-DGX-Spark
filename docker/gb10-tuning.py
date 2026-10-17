# SPDX-License-Identifier: Apache-2.0
"""GB10 serving knobs and fixes that vLLM does not expose. Runs once at image build; fails closed on drift.

DSV41_MXFP8_BACKEND: the FlashInfer backend behind vLLM's FlashInferCutlassMxfp8LinearKernel, the kernel vLLM
selects for this checkpoint's FP8 (32x32 ue8m0, MXFP8) dense layers on sm_121. Stock vLLM hard-codes "cutlass".
FlashInfer 0.6.18 also ships "b12x", a warp-level MMA kernel with 16/32-row tiles for SM12x decode, which needs
K % 128 == 0; at TP4 some sharded layers have K = 576 (2304 / 4). Values:
  cutlass  stock behaviour (default)
  b12x     b12x for every GEMM whose K is a multiple of 128, cutlass for the rest
  auto     FlashInfer's own heuristic
Keeping the same vLLM kernel class keeps vLLM's fused MXFP8 query-quant path. Measured neutral (README).
The b12x-for-decode finding is MiaAI-Lab's (DeepSeek-v4.1-Flash-DGX-Sparks, SGLang); no code is taken from it.

DSV41_TOPK_SORT (default 1): put the sparse-attention indexer's top-k indices in ascending order (-1 padding last)
after the top-k kernels, in decode and prefill. The per-row top-k kernels emit indices in an order set by atomics,
so attention accumulated in a different order on every run and identical requests diverged; vLLM's eager
short-context shortcut already emits ascending order, which is why eager decode was reproducible and CUDA-graph
decode (which always runs the real top-k) was not. Sorting restores one fixed order everywhere.
"""
import sys

MXFP8 = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
INDEXER = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py"


def patch(path, mark, pairs, append=""):
    s = open(path).read()
    if mark in s:
        print(f"{mark}: already installed")
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{path}: anchor found {s.count(old)} times: {old!r}")
        s = s.replace(old, new)
    s += append
    compile(s, path, "exec")
    open(path, "w").write(s)
    print(f"{mark}: installed")


MXFP8_HELPER = '''
import os as _os

_DSV41_MXFP8 = _os.environ.get("DSV41_MXFP8_BACKEND", "cutlass")  # dsv41-1017: cutlass | b12x | auto
if _DSV41_MXFP8 not in ("cutlass", "b12x", "auto"):
    raise ValueError(f"DSV41_MXFP8_BACKEND must be cutlass, b12x or auto, got {_DSV41_MXFP8!r}")


def _dsv41_mxfp8_backend(x):
    # b12x needs the contraction dim K to be a multiple of 128; TP-sharded K=576 layers stay on cutlass.
    if _DSV41_MXFP8 == "b12x" and x.shape[-1] % 128:
        return "cutlass"
    return _DSV41_MXFP8
'''
mxfp8_import = "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n"
patch(MXFP8, "_dsv41_mxfp8_backend", [
    ('            backend="cutlass",\n', "            backend=_dsv41_mxfp8_backend(input_mxfp8),\n"),
    (mxfp8_import, mxfp8_import + MXFP8_HELPER),
])

TOPK_HELPER = '''

# dsv41-1017: one fixed order for the indexer's top-k indices (see docker/gb10-tuning.py).
import os as _dsv41_os

_DSV41_TOPK_SORT = _dsv41_os.environ.get("DSV41_TOPK_SORT", "1") == "1"


def _dsv41_sort_topk(idx: torch.Tensor) -> None:
    big = torch.iinfo(idx.dtype).max
    s = torch.where(idx < 0, big, idx).sort(dim=-1).values
    idx.copy_(torch.where(s == big, -1, s))
'''
patch(INDEXER, "_dsv41_sort_topk", [
    # prefill: after ops.top_k_per_row_prefill(...)
    ("                    topk_tokens,\n                )\n\n            _merge_dcp_topk_global(\n",
     "                    topk_tokens,\n                )\n"
     "                if _DSV41_TOPK_SORT:\n                    _dsv41_sort_topk(topk_indices)\n\n"
     "            _merge_dcp_topk_global(\n"),
    # decode: after the cooperative / persistent / per-row top-k branch
    ("                topk_tokens,\n            )\n\n        if decode_metadata.global_seq_lens is not None:\n",
     "                topk_tokens,\n            )\n"
     "        if _DSV41_TOPK_SORT:\n            _dsv41_sort_topk(topk_indices)\n\n"
     "        if decode_metadata.global_seq_lens is not None:\n"),
], append=TOPK_HELPER)
