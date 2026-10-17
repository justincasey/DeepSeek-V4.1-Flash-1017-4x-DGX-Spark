# SPDX-License-Identifier: Apache-2.0
"""GB10 serving knobs that vLLM does not expose. Runs once at image build; fails closed on drift.

DSV41_MXFP8_BACKEND: the FlashInfer backend behind vLLM's FlashInferCutlassMxfp8LinearKernel, the kernel vLLM
selects for this checkpoint's FP8 (32x32 ue8m0, MXFP8) dense layers on sm_121. Stock vLLM hard-codes "cutlass".
FlashInfer 0.6.18 also ships "b12x", a warp-level MMA kernel with 16/32-row tiles for SM12x decode, and "auto",
which prefers b12x on SM12x and falls back to cutlass for shapes b12x cannot take (K % 128 != 0). Keeping the
same vLLM kernel class keeps vLLM's fused MXFP8 query-quant path. Default "cutlass" = stock behaviour.
The b12x-for-decode finding is MiaAI-Lab's (DeepSeek-v4.1-Flash-DGX-Sparks, SGLang); no code is taken from it.
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
MARK = "_DSV41_MXFP8_BACKEND"

s = open(PATH).read()
if MARK not in s:
    for old, new in (
        ('            backend="cutlass",\n', f"            backend={MARK},\n"),
        ("from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n",
         "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n\n"
         "import os as _os\n"
         f'{MARK} = _os.environ.get("DSV41_MXFP8_BACKEND", "cutlass")  # dsv41-1017: cutlass | b12x | auto\n'),
    ):
        if s.count(old) != 1:
            sys.exit(f"{PATH}: anchor found {s.count(old)} times: {old!r}")
        s = s.replace(old, new)
    compile(s, PATH, "exec")
    open(PATH, "w").write(s)
print("DSV41_MXFP8_BACKEND selector installed")
