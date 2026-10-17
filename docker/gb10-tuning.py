# SPDX-License-Identifier: Apache-2.0
"""GB10 serving knobs that vLLM does not expose. Runs once at image build; fails closed on drift.

DSV41_MXFP8_BACKEND: the FlashInfer backend behind vLLM's FlashInferCutlassMxfp8LinearKernel, the kernel vLLM
selects for this checkpoint's FP8 (32x32 ue8m0, MXFP8) dense layers on sm_121. Stock vLLM hard-codes "cutlass".
FlashInfer 0.6.18 also ships "b12x", a warp-level MMA kernel with 16/32-row tiles for SM12x decode, which needs
K % 128 == 0; at TP4 some sharded layers have K = 576 (2304 / 4). Values:
  cutlass  stock behaviour (default)
  b12x     b12x for every GEMM whose K is a multiple of 128, cutlass for the rest
  auto     FlashInfer's own heuristic
Keeping the same vLLM kernel class keeps vLLM's fused MXFP8 query-quant path.
The b12x-for-decode finding is MiaAI-Lab's (DeepSeek-v4.1-Flash-DGX-Sparks, SGLang); no code is taken from it.
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
MARK = "_dsv41_mxfp8_backend"

HELPER = f'''
import os as _os

_DSV41_MXFP8 = _os.environ.get("DSV41_MXFP8_BACKEND", "cutlass")  # dsv41-1017: cutlass | b12x | auto
if _DSV41_MXFP8 not in ("cutlass", "b12x", "auto"):
    raise ValueError(f"DSV41_MXFP8_BACKEND must be cutlass, b12x or auto, got {{_DSV41_MXFP8!r}}")


def {MARK}(x):
    # b12x needs the contraction dim K to be a multiple of 128; TP-sharded K=576 layers stay on cutlass.
    if _DSV41_MXFP8 == "b12x" and x.shape[-1] % 128:
        return "cutlass"
    return _DSV41_MXFP8
'''

s = open(PATH).read()
if MARK not in s:
    anchor_import = "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n"
    for old, new in (
        ('            backend="cutlass",\n', f"            backend={MARK}(input_mxfp8),\n"),
        (anchor_import, anchor_import + HELPER),
    ):
        if s.count(old) != 1:
            sys.exit(f"{PATH}: anchor found {s.count(old)} times: {old!r}")
        s = s.replace(old, new)
    compile(s, PATH, "exec")
    open(PATH, "w").write(s)
print("DSV41_MXFP8_BACKEND selector installed")
