#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Static contract test: no GPU, no ssh, no network. Generates every rank's docker run + vllm serve command
# with pinned ring values and checks the invariants a deployment depends on.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
fails=0
ok()   { printf 'PASS %s\n' "$1"; }
bad()  { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }
has()  { grep -qF -- "$2" <<<"$1"; }

bash -n dsv41.sh docker/apply-upstream.sh tests/test-recipe.sh && ok "shell syntax" || bad "shell syntax"
python3 -m py_compile tests/api_conformance.py tests/test_encoding_parity.py && ok "python syntax" || bad "python syntax"

# Pinned ring, explicit values for every key the commands depend on (caller env wins over .env).
export NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4" MODELS_DIR=/m MODEL_NAME=M IMAGE=img:t CONTAINER=c \
  SWITCHLESS_NCCL=/nccl CACHE_DIR=/cache SERVED_NAME=s HOST=0.0.0.0 PORT=8000 MASTER_PORT=29511 API_KEY= \
  MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 GPU_MEMORY_UTILIZATION=0.8 KV_CACHE_MEMORY_BYTES= \
  MOE_MODE=ep DSPARK=1 NUM_SPECULATIVE_TOKENS=3 EAGER=0 COMPILATION_CONFIG= TEXT_ONLY=0 PREFIX_CACHING=1 \
  EXTRA_VLLM_ARGS= EXTRA_DOCKER_ARGS= \
  RING_PREFIX=24 RING_HCA_0=hA,hB RING_MGMT_0=m0 RING_HCA_1=hB,hA RING_MGMT_1=m1 \
  RING_HCA_2=hB,hA RING_MGMT_2=m2 RING_HCA_3=hB,hA RING_MGMT_3=m3
serve() { ./dsv41.sh print-serve "$1"; }

for r in 0 1 2 3; do
  c=$(serve "$r")
  # switchless NCCL runtime contract (ranks 0-2 and 1-3 share no cable)
  for want in "LD_PRELOAD=/opt/switchless-nccl/libnccl.so.2" "VLLM_NCCL_SO_PATH=/opt/switchless-nccl/libnccl.so.2" \
              "/nccl:/opt/switchless-nccl:ro" "NCCL_SWITCHLESS_RING_ONLY=1" "NCCL_SKIP_TREE_CONNECT=1" "NCCL_ALGO=Ring" \
              "NCCL_IB_SUBNET_AWARE_ROUTING=1" "NCCL_IB_SUBNET_PREFIX_LEN=24" "NCCL_IB_MERGE_NICS=0" \
              "NCCL_SOCKET_IFNAME=m$r" "GLOO_SOCKET_IFNAME=m$r" "--device /dev/infiniband" "--ulimit core=0"; do
    has "$c" "$want" || bad "rank $r: missing $want"
  done
  hca=$(grep -o 'NCCL_IB_HCA=[^ ]*' <<<"$c" | cut -d= -f2 | tr -d '\\')
  [[ $(tr ',' '\n' <<<"$hca" | grep -c .) == 2 ]] || bad "rank $r: NCCL_IB_HCA must name exactly two rails, got '$hca'"
  # multi-node rendezvous: every rank points at rank 0, only workers are headless
  has "$c" "--node-rank $r" && has "$c" "--master-addr 10.0.0.1" && has "$c" "--nnodes 4" || bad "rank $r: rendezvous args"
  if (( r == 0 )); then has "$c" "--headless" && bad "rank 0 must serve the API (no --headless)"
  else has "$c" "--headless" || bad "rank $r must run --headless"; fi
  # disk-backed Engram (resident Engram does not fit 121 GiB per rank)
  has "$c" "VLLM_ENGRAM_DISK_BACKED=1" && has "$c" "VLLM_ENGRAM_MODEL_DIR=/m/M" && has "$c" 'disk_backed' \
    || bad "rank $r: disk-backed Engram not configured"
  # full API surface: V4.1 tokenizer mode, reasoning and tool parsers, image input
  for want in "--tokenizer-mode deepseek_v41" "--reasoning-parser deepseek_v41" "--tool-call-parser deepseek_v41" \
              "--enable-auto-tool-choice" "--mm-encoder-tp-mode data"; do
    has "$c" "$want" || bad "rank $r: missing $want"
  done
  # EP dispatch over the ring must be collectives (allgather/reduce-scatter), never point-to-point all-to-all
  has "$c" "--all2all-backend allgather_reducescatter" || bad "rank $r: EP without the ring-safe all2all backend"
  has "$c" "--language-model-only" && bad "rank $r: image input disabled by default"
done
ok "per-rank invariants checked (4 ranks)"

c=$(serve 0)
spec=$(grep -o "\-\-speculative-config [^ ]*" <<<"$c" | cut -d' ' -f2- | sed 's/\\//g')
python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["method"]=="dspark" and d["num_speculative_tokens"]==3' "$spec" \
  && ok "DSpark speculative config is valid JSON" || bad "DSpark speculative config: $spec"
DSPARK=0 serve 0 | grep -q -- --speculative-config && bad "DSPARK=0 still passes --speculative-config" || ok "DSPARK=0 drops speculation"
c=$(KV_CACHE_MEMORY_BYTES=12345 serve 0)
has "$c" "--kv-cache-memory-bytes 12345" && ! has "$c" "--gpu-memory-utilization" && ok "KV byte budget replaces utilization" \
  || bad "KV_CACHE_MEMORY_BYTES handling"
API_KEY=k serve 0 | grep -q -- "--api-key k" && ok "API_KEY enforced" || bad "API_KEY not passed"
MOE_MODE=tp serve 0 | grep -q -- --enable-expert-parallel && bad "MOE_MODE=tp still enables EP" || ok "MOE_MODE=tp is EP1"
TEXT_ONLY=1 serve 0 | grep -q -- --language-model-only && ok "TEXT_ONLY=1 drops the vision encoder" || bad "TEXT_ONLY"

# pinned checkpoint manifest: every shard of the index plus config, tokenizer and encoder
m=config/DeepSeek-V4.1-Flash-1017.sha256
if awk 'NF != 2 || length($1) != 64 {exit 1}' "$m" && [[ $(grep -c '\.safetensors$' "$m") == 48 ]] \
   && grep -qE ' (config\.json|tokenizer\.json|model\.safetensors\.index\.json|encoding/encoding\.py)$' "$m"; then
  ok "checkpoint manifest pins 48 shards and the serving files"
else bad "checkpoint manifest $m"; fi

(( fails == 0 )) && echo "OK" || { echo "FAILED: $fails"; exit 1; }
