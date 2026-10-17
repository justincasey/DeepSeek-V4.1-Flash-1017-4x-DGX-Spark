#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Apply the pinned upstream vLLM runtime fixes that DeepSeek-V4.1 needs on GB10 (sm_121).
# Runs once at image build. Every step fails closed: a missing anchor or file stops the build,
# so base-image drift shows up here instead of as wrong tokens at serve time.
#
# The fixes themselves are not ours. They come from vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe
# (AGPL-3.0-only), fetched at a pinned commit and applied unmodified:
#   overlays/disk-engram       disk-backed Engram (Blackwellboy, PR #2; hardened by vcruz305)
#   overlays/sm120-sparse-fix  SM120 sparse-MLA contract fixes 01-08 (fattchris)
#   tools/dspark/fixswa.py     Fix 1 only: DSpark non-causal SWA passthrough on SM120 (fattchris). Its Fix 2
#                              amends mkdq8's _DQ8 loader, which this image does not install.
#   tools/engram/*.py          decode hoist, cold I/O, draft EP filter, SM12x persistent top-k (fattchris)
# Not applied: tools/engram/engram_key_fix.py repairs a registry-key collision in an opaque split-op Engram
# implementation; neither this base's engram.py nor the overlay has that registry, so the defect cannot occur.
# Not applied: tools/dspark/dq.py and mkdq8.py (the draft-quant override and the FP8 draft dequant that pairs
# with it exist for EXL3 packs, whose config hides the drafter's native format; mkdq8's anchor is not in the
# stock base), and everything vllm-exl3 / ExLlamaV3 (this image serves the native checkpoint).
set -euo pipefail

UP=${1:?usage: apply-upstream.sh /path/to/upstream-recipe}
VLLM=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.realpath(vllm.__file__)))')
# The upstream patch scripts hard-code the stock site-packages layout.
[ "$VLLM" = /usr/local/lib/python3.12/dist-packages/vllm ] || { echo "vLLM at $VLLM, upstream patches expect /usr/local/lib/python3.12/dist-packages/vllm" >&2; exit 1; }

echo "== disk-backed Engram overlay"
for rel in config/engram.py models/deepseek_v4_1/common/engram.py \
           models/deepseek_v4_1/common/engram_disk.py model_executor/model_loader/weight_utils.py; do
  if [ "$rel" != models/deepseek_v4_1/common/engram_disk.py ] && [ ! -f "$VLLM/$rel" ]; then
    echo "base image is missing $VLLM/$rel" >&2; exit 1
  fi
  install -D -m 0644 "$UP/overlays/disk-engram/vllm/$rel" "$VLLM/$rel"
  echo "overlay $rel"
done
# Upstream overlay carries one typo in an eager-strategy branch (vcruz305 fixes it the same way at build).
sed -i 's/_adviseadvise_dontneed_path/_fadvise_dontneed_path/' "$VLLM/model_executor/model_loader/weight_utils.py"
if grep -q _adviseadvise_dontneed_path "$VLLM/model_executor/model_loader/weight_utils.py"; then
  echo "fadvise typo still present" >&2; exit 1
fi

echo "== SM120 sparse-MLA fixes"
bash "$UP/overlays/sm120-sparse-fix/apply.sh"
echo "== DSpark and Engram fixes (upstream order: fixswa anchors on the sm120 decode-narrow patch)"
# fixswa.py holds two independent fixes; run only the section before its "# Fix 2" marker.
python3 - "$UP/tools/dspark/fixswa.py" <<'PY'
import sys
src = open(sys.argv[1]).read()
start, cut = src.find("# Fix 1:"), src.find("# Fix 2:")
if start < 0 or cut < start or "_DSPARK_NONCAUSAL_PASSTHROUGH" not in src[start:cut]:
    sys.exit("fixswa.py layout changed: Fix 1 / Fix 2 markers not found")
exec(compile(src[:cut], sys.argv[1] + " (Fix 1)", "exec"))
PY
for s in tools/engram/engram_hoist.py tools/engram/engram_io_fix.py \
         tools/engram/draft_ep_filter_fix.py tools/engram/persistent_topk_sm12x_fix.py; do
  echo "-- $s"
  python3 "$UP/$s"
done

echo "== compile and import checks"
python3 - <<'PY'
import compileall, os, sys
import vllm
root = os.path.dirname(vllm.__file__)
ok = True
for d in ("models/deepseek_v4_1", "v1/worker/gpu/spec_decode", "model_executor/model_loader",
          "model_executor/layers", "v1/attention/backends/mla", "config"):
    ok &= compileall.compile_dir(os.path.join(root, d), quiet=1)
from vllm.config.engram import EngramConfig
assert hasattr(EngramConfig, "disk_backed"), "EngramConfig.disk_backed missing"
os.environ["VLLM_ENGRAM_DISK_BACKED"] = "1"
from vllm.models.deepseek_v4_1.common.engram_disk import should_skip_engram_embed_tensor
assert should_skip_engram_embed_tensor("layers.1.engram.embed.weight")
assert should_skip_engram_embed_tensor("layers.14.engram.embed.scale")
print("vLLM", vllm.__version__, "patched tree compiles:", ok)
sys.exit(0 if ok else 1)
PY
