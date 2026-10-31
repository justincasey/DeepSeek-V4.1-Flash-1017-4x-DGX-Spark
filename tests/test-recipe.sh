#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Static contract test: no GPU, no ssh, no network. Generates every rank's memory-guard arm + docker run + vllm
# serve command with pinned fabric values (ring, then switched) and checks the invariants a deployment depends on
# (spark-ds41f r6w at tp4-ring, config/cluster.json @ 2dc95329; sparknet's "switched" profile).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
fails=0
ok()   { printf 'PASS %s\n' "$1"; }
bad()  { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }
has()  { grep -qF -- "$2" <<<"$1"; }
envval() { tr ' ' '\n' <<<"$1" | sed -n "s/^$2=//p" | head -1 | sed 's/\\//g'; }   # unquoted value of -e KEY=...

# bash -n checks only its first operand, so one call per file.
shopt -s nullglob
for f in dsv41.sh tests/*.sh docker/*.sh; do
  bash -n "$f" && ok "shell syntax $f" || bad "shell syntax $f"
done
python3 -m py_compile tests/*.py bench/bench.py docker/*.py && ok "python syntax" || bad "python syntax"
shopt -u nullglob

# Pinned ring, explicit values for every key the commands depend on (caller env wins over .env). The head's
# HCAs face its neighbours in the opposite order to the workers' (this site's unlike-port cabling), and the GID
# indices differ per node (5/3/3/5, as on the external 4 KiB site).
export NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4" MODELS_DIR=/m MODEL_NAME=M IMAGE=img:t CONTAINER=c \
  CACHE_DIR=/cache-host SERVED_NAME=s HOST=0.0.0.0 PORT=8000 MASTER_PORT=9999 API_KEY= \
  MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 GPU_MEMORY_UTILIZATION=0.82 \
  KV_CACHE_MEMORY_BYTES=11274289152 PREFIX_CACHING=1 PREFIX_CACHE_RETENTION_INTERVAL=8192 KERNEL_BACKEND=tilelang \
  DSPARK=1 NUM_SPECULATIVE_TOKENS=5 DRAFT_SAMPLE_METHOD=greedy ADAPTIVE_VERIFICATION_ROW_COST=0.02 \
  EAGER=0 COMPILATION_CONFIG= GENERATION_DEFAULTS='{"temperature":1.0,"top_p":1.0}' TEXT_ONLY=0 MAX_IMAGES=4 \
  DISPLAY_CARVEOUT=0 DRI_CARD=/dev/dri/by-path/pci-000f:01:00.0-card CPUSET=5-9,15-19 CONTAINER_MEMORY=112g \
  STARTUP_MEMGUARD_GIB=5 STARTUP_MEMGUARD_INTERVAL=0.25 STARTUP_MEMGUARD_WAIT=120 MEMGUARD_GIB=3 MEMGUARD_INTERVAL=2 \
  EXTRA_VLLM_ARGS= EXTRA_DOCKER_ARGS= \
  FABRIC=ring ROCE_TC=106 SWITCH_PREFIX= \
  RING_PREFIX=24 \
  RING_HCA_PREV_0=hA RING_HCA_NEXT_0=hB RING_GID_0=5 RING_MGMT_0=m0 \
  RING_HCA_PREV_1=hB RING_HCA_NEXT_1=hA RING_GID_1=3 RING_MGMT_1=m1 \
  RING_HCA_PREV_2=hB RING_HCA_NEXT_2=hA RING_GID_2=3 RING_MGMT_2=m2 \
  RING_HCA_PREV_3=hB,hB2 RING_HCA_NEXT_3=hA,hA2 RING_GID_3=5 RING_MGMT_3=m3
serve() { ./dsv41.sh print-serve "$1"; }
run() { ./dsv41.sh print-serve "$1" | sed -n 2p; }   # the docker line
guard() { ./dsv41.sh print-serve "$1" | sed -n 1p; }  # the memory-guard arm line

declare -A gid=([0]=5 [1]=3 [2]=3 [3]=5)
declare -A peers=([0]='{"3":["hA"],"1":["hB"]}' [1]='{"0":["hB"],"2":["hA"]}' [2]='{"1":["hB"],"3":["hA"]}'
                  [3]='{"2":["hB","hB2"],"0":["hA","hA2"]}')
declare -A nccl=([0]='=hA,hB' [1]='=hA,hB' [2]='=hA,hB' [3]='=hA,hA2,hB,hB2')
for r in 0 1 2 3; do
  c=$(run "$r") || { bad "rank $r: print-serve failed"; continue; }
  # the image's entrypoint is `vllm serve`: the argv after the image starts with the model path
  has "$c" "--entrypoint" && bad "rank $r: must keep the image's vllm serve entrypoint"
  has "$c" "img:t /models " || bad "rank $r: image must run 'vllm serve /models'"
  # spark-ds41f container.docker_run_args
  for want in "--network=host" "--ipc=host" "--gpus=all" "--device=/dev/infiniband:/dev/infiniband:rwm" \
              "--cap-add=IPC_LOCK" "--ulimit=memlock=-1:-1" "--ulimit=stack=67108864:67108864" "--ulimit=core=0" \
              "--security-opt=seccomp=unconfined" "--security-opt=label=disable" "--shm-size=16g" \
              "--memory=112g" "--memory-swap=112g" "--cpuset-cpus=5-9\,15-19" \
              "--mount=type=bind\,source=/dev/dri/by-path/pci-000f:01:00.0-card\,target=/dev/dri/card0" \
              "--device-cgroup-rule=c\ 226:\*\ rw" "-v /cache-host:/cache:rw" "-v /m/M:/models:ro"; do
    has "$c" "$want" || bad "rank $r: missing $want"
  done
  # the image carries the patched NCCL: nothing from the host is preloaded or mounted over it
  for gone in LD_PRELOAD VLLM_NCCL_SO_PATH switchless-nccl NCCL_SWITCHLESS_RING_ONLY; do
    has "$c" "$gone" && bad "rank $r: host NCCL bundle leftover $gone"
  done
  # sparknet's tp4-ring NCCL profile
  for want in "NCCL_ALGO=Ring" "NCCL_SWITCHLESS_BIDIRECTIONAL=2" "NCCL_IB_MERGE_NICS=1" "NCCL_P2P_DISABLE=1" \
              "NCCL_SHM_DISABLE=1" "NCCL_RUNTIME_CONNECT=1" "NCCL_MIN_NCHANNELS=4" "NCCL_MAX_NCHANNELS=4" \
              "NCCL_BUFFSIZE=4194304" "NCCL_PROTO=\^LL128" "NCCL_IB_SUBNET_AWARE_ROUTING=1" "NCCL_IB_SUBNET_PREFIX_LEN=24" \
              "SPARKNET_ROCE_TOPOLOGY=ring4" "SPARK3_DISPLAY_CARVEOUT_WEIGHTS=0" "VLLM_DS41_KERNEL_BACKEND=tilelang" \
              "VLLM_L2_PREFETCH_KERNELS=tilelang" "TILELANG_CACHE_DIR=/cache/kkref/jit/tilelang" "HF_HUB_OFFLINE=1" \
              "NCCL_SOCKET_IFNAME=m$r" "GLOO_SOCKET_IFNAME=m$r" "TP_SOCKET_IFNAME=m$r" "VLLM_HOST_IP=10.0.0.$((r + 1))"; do
    has "$c" "$want" || bad "rank $r: missing $want"
  done
  has "$c" "NCCL_THREAD_THRESHOLDS=-2\ -2\ -2\ 1\ 1\ 1" || bad "rank $r: NCCL_THREAD_THRESHOLDS"
  # per-node RoCE mapping: exact-match HCA set for NCCL, neighbour rank -> HCAs (lane order) for sparknet
  [[ $(envval "$c" NCCL_IB_HCA) == "${nccl[$r]}" ]] || bad "rank $r: NCCL_IB_HCA=$(envval "$c" NCCL_IB_HCA), want ${nccl[$r]}"
  [[ $(envval "$c" SPARKNET_ROCE_PEER_HCAS) == "${peers[$r]}" ]] \
    || bad "rank $r: SPARKNET_ROCE_PEER_HCAS=$(envval "$c" SPARKNET_ROCE_PEER_HCAS), want ${peers[$r]}"
  python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(envval "$c" SPARKNET_ROCE_PEER_HCAS)" || bad "rank $r: peer map is not JSON"
  [[ $(envval "$c" NCCL_IB_GID_INDEX) == "${gid[$r]}" && $(envval "$c" SPARKNET_ROCE_GID_INDEX) == "${gid[$r]}" ]] \
    || bad "rank $r: GID index must be ${gid[$r]} for NCCL and sparknet"
  # multi-node rendezvous: every rank points at rank 0, only workers are headless
  has "$c" "--node-rank $r" && has "$c" "--master-addr 10.0.0.1" && has "$c" "--master-port 9999" && has "$c" "--nnodes 4" \
    || bad "rank $r: rendezvous args"
  if (( r == 0 )); then has "$c" "--headless" && bad "rank 0 must serve the API (no --headless)"
  else has "$c" "--headless" || bad "rank $r must run --headless"; fi
  # r6v serve arguments: TileLang kernels, B12X loader, disk Engram, full API surface incl. images
  for want in "--tensor-parallel-size 4" "--load-format b12x" "--block-size 256" "--dtype bfloat16" \
              "--attention-backend TILELANG" "--linear-backend tilelang" "--moe-backend tilelang" \
              "--max-num-seqs 16" "--max-num-batched-tokens 8192" "--kv-cache-memory-bytes 11274289152" \
              "--gpu-memory-utilization 0.82" "--max-parallel-prefills 1" "--enable-chunked-prefill" \
              "--enable-prefix-caching" "--prefix-cache-retention-interval 8192" "--async-scheduling" \
              "--generation-config vllm" "--jit-monitor-mode warn" "--distributed-executor-backend mp" \
              "--no-scheduler-reserve-full-isl" "--no-enable-flashinfer-autotune" "--mm-processor-cache-gb 0" \
              "--tokenizer-mode deepseek_v41" "--reasoning-parser deepseek_v41" "--tool-call-parser deepseek_v41" \
              "--enable-auto-tool-choice" '--limit-mm-per-prompt \{\"image\":4\}'; do
    has "$c" "$want" || bad "rank $r: missing $want"
  done
  has "$c" 'table_memory\":\"disk' || bad "rank $r: Engram tables must be disk-backed"
  # old-engine knobs are gone
  for gone in DSV41_MXFP8_BACKEND DSV41_TOPK_SORT VLLM_ENGRAM_DISK_BACKED "--moe-backend marlin" --all2all-backend \
              --enable-expert-parallel --mm-encoder-tp-mode --language-model-only; do
    has "$c" "$gone" && bad "rank $r: obsolete or wrong default $gone"
  done
  # fail-closed memory guard armed (prelaunch) before the rank's docker run
  g=$(guard "$r")
  [[ $g == '"$HOME/.local/lib/c-memguard/scripts/memguard-start.sh" c startup 5 0.25 120 prelaunch' ]] \
    || bad "rank $r: startup memory guard line: $g"
done
ok "per-rank invariants checked (4 ranks)"

# Switched fabric: sparknet's "switched" profile (transport oneshot-switched). Rails in subnet order; rank 3 has a
# third rail (NCCL takes it, sparknet stripes over two at most); GID indices differ per node.
declare -A sw_hcas=([0]=rA,rB [1]=rA,rB [2]=rA,rB [3]=rA,rB,rC) sw_gid=([0]=3 [1]=3 [2]=5 [3]=3)
sw() {
  FABRIC=switched RING_PREFIX= SWITCH_PREFIX=24 \
    SWITCH_HCAS_0=${sw_hcas[0]} SWITCH_GID_0=${sw_gid[0]} SWITCH_MGMT_0=m0 \
    SWITCH_HCAS_1=${sw_hcas[1]} SWITCH_GID_1=${sw_gid[1]} SWITCH_MGMT_1=m1 \
    SWITCH_HCAS_2=${sw_hcas[2]} SWITCH_GID_2=${sw_gid[2]} SWITCH_MGMT_2=m2 \
    SWITCH_HCAS_3=${sw_hcas[3]} SWITCH_GID_3=${sw_gid[3]} SWITCH_MGMT_3=m3 "$@"
}
swfail=0
for r in 0 1 2 3; do
  c=$(sw ./dsv41.sh print-serve "$r" | sed -n 2p) || { bad "switched rank $r: print-serve failed"; swfail=1; continue; }
  for want in "SPARKNET_ROCE_TOPOLOGY=direct" "NCCL_MAX_NCHANNELS=8" "NCCL_BUFFSIZE=1048576" "NCCL_CUMEM_ENABLE=0" \
              "SPARKNET_ROCE_ALLGATHER_MAX_BYTES=4194304" "SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES=2097152" \
              "NCCL_IB_TC=106" "SPARKNET_ROCE_TRAFFIC_CLASS=106" "NCCL_IB_SUBNET_PREFIX_LEN=24" "NCCL_PROTO=\^LL128" \
              "NCCL_IB_SUBNET_AWARE_ROUTING=1" "NCCL_SOCKET_IFNAME=m$r" "VLLM_HOST_IP=10.0.0.$((r + 1))"; do
    has "$c" "$want" || { bad "switched rank $r: missing $want"; swfail=1; }
  done
  # nothing of the neighbour ring: no forced ring algorithm, no switchless-ring controls, no neighbour route map
  for gone in NCCL_ALGO NCCL_SWITCHLESS_BIDIRECTIONAL NCCL_RUNTIME_CONNECT NCCL_MIN_NCHANNELS NCCL_THREAD_THRESHOLDS \
              NCCL_MIN_TRAFFIC_PER_CHANNEL SPARKNET_ROCE_PEER_HCAS SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES ring4; do
    has "$c" "$gone" && { bad "switched rank $r: ring setting $gone"; swfail=1; }
  done
  want_sn=$(cut -d, -f1-2 <<<"${sw_hcas[$r]}")
  [[ $(envval "$c" NCCL_IB_HCA) == "=${sw_hcas[$r]}" && $(envval "$c" SPARKNET_ROCE_HCA) == "$want_sn" ]] \
    || { bad "switched rank $r: NCCL_IB_HCA=$(envval "$c" NCCL_IB_HCA) SPARKNET_ROCE_HCA=$(envval "$c" SPARKNET_ROCE_HCA)"; swfail=1; }
  [[ $(envval "$c" NCCL_IB_GID_INDEX) == "${sw_gid[$r]}" && $(envval "$c" SPARKNET_ROCE_GID_INDEX) == "${sw_gid[$r]}" ]] \
    || { bad "switched rank $r: GID index must be ${sw_gid[$r]}"; swfail=1; }
done
(( swfail )) || ok "switched: sparknet switched profile, every rail to NCCL, two stripes to sparknet, ROCE_TC on both (4 ranks)"
c=$(sw env ROCE_TC= ./dsv41.sh print-serve 1 | sed -n 2p)
! has "$c" NCCL_IB_TC && ! has "$c" SPARKNET_ROCE_TRAFFIC_CLASS && ok "switched: ROCE_TC= leaves RoCE unmarked" || bad "ROCE_TC= still marks"
c=$(sw env ROCE_TC=040 ./dsv41.sh print-serve 1 | sed -n 2p)
has "$c" "NCCL_IB_TC=40" && has "$c" "SPARKNET_ROCE_TRAFFIC_CLASS=40" && ok "switched: ROCE_TC=040 passed as 40 (sparknet parses int(x, 0))" \
  || bad "ROCE_TC=040: $(envval "$c" SPARKNET_ROCE_TRAFFIC_CLASS)"
for bogus in "FABRIC=mesh" "ROCE_TC=256" "ROCE_TC=x" "SWITCH_GID_0=x" "SWITCH_MGMT_1=" "RING_PREFIX=24"; do
  if c=$(sw env $bogus ./dsv41.sh print-serve 0 2>/dev/null); then bad "switched $bogus print-serve succeeded"
  elif [[ -n $c ]]; then bad "switched $bogus printed a partial command"
  else ok "switched $bogus refused"; fi
done
if c=$(SWITCH_PREFIX=24 ./dsv41.sh print-serve 0 2>/dev/null); then bad "FABRIC=ring with SWITCH_PREFIX set print-serve succeeded"
else ok "FABRIC=ring refuses switch pins"; fi

c=$(run 0) || bad "print-serve 0 failed"
gen=$(envval "$(grep -o -- '--override-generation-config [^ ]*' <<<"$c" | tr ' ' '=' | sed 's/^--//')" override-generation-config)
[[ $gen == '{"temperature":1.0,"top_p":1.0}' ]] && ok "generation defaults temperature 1.0 top_p 1.0" || bad "generation defaults: $gen"
c=$(GENERATION_DEFAULTS= run 0) || bad "GENERATION_DEFAULTS= print-serve failed"
has "$c" --override-generation-config && bad "GENERATION_DEFAULTS= still overrides" || ok "GENERATION_DEFAULTS= keeps vLLM's sampling"

c=$(run 0)
spec=$(grep -o -- "--speculative-config [^ ]*" <<<"$c" | cut -d' ' -f2- | sed 's/\\//g')
python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert d=={"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":4,"attention_backend":"TILELANG","draft_sample_method":"greedy","rejection_sample_method":"block","enable_adaptive_verification":True,"adaptive_verification_row_cost":0.02}, d' "$spec" \
  && ok "DSpark speculative config (k=5, adaptive verification at row cost 0.02)" || bad "DSpark speculative config: $spec"
c=$(ADAPTIVE_VERIFICATION_ROW_COST= run 0)
has "$c" adaptive_verification && bad "ADAPTIVE_VERIFICATION_ROW_COST= still adaptive" || ok "ADAPTIVE_VERIFICATION_ROW_COST= turns adaptive verification off"
c=$(DSPARK=0 run 0) || bad "DSPARK=0 print-serve failed"
has "$c" --speculative-config && bad "DSPARK=0 still passes --speculative-config" || ok "DSPARK=0 drops speculation"

# CUDA graphs: FULL_AND_PIECEWISE on spark-ds41f's ladder up to MAX_NUM_SEQS x (k+1) rows; long prefill leaves them
ladder='1\,2\,3\,4\,6\,8\,12\,16\,20\,24\,28\,32\,40\,48\,56\,64\,72\,80\,88\,96'
c=$(run 0)
has "$c" "FULL_AND_PIECEWISE" && has "$c" "cudagraph_capture_sizes\\\":\\[$ladder\\]" && has "$c" "--max-cudagraph-capture-size 96" \
  && has "$c" "--long-prefill-token-threshold 8096" && has "$c" 'custom_ops\":\[\"all\"\]' \
  && ok "graph ladder to 96 rows, long-prefill threshold 8096" || bad "derived compilation config: $(grep -o -- '--compilation-config [^ ]*' <<<"$c")"
c=$(MAX_NUM_SEQS=8 run 0)
has "$c" "--max-cudagraph-capture-size 48" && has "$c" "1\,2\,3\,4\,6\,8\,12\,16\,20\,24\,28\,32\,40\,48\\]" \
  && has "$c" "--long-prefill-token-threshold 8144" && ok "MAX_NUM_SEQS=8: ladder to 48" || bad "MAX_NUM_SEQS=8 ladder"
c=$(DSPARK=0 run 0)
has "$c" "--max-cudagraph-capture-size 16" && has "$c" "1\,2\,3\,4\,6\,8\,12\,16\\]" && has "$c" "--long-prefill-token-threshold 8176" \
  && ok "DSPARK=0: one row per sequence" || bad "DSPARK=0 ladder"
c=$(EAGER=1 run 0) || bad "EAGER=1 print-serve failed"
has "$c" --enforce-eager && ! has "$c" --compilation-config && ! has "$c" --max-cudagraph-capture-size \
  && ok "EAGER=1 drops CUDA graphs" || bad "EAGER handling"

c=$(KV_CACHE_MEMORY_BYTES= run 0) || bad "KV_CACHE_MEMORY_BYTES= print-serve failed"
! has "$c" --kv-cache-memory-bytes && has "$c" "--gpu-memory-utilization 0.82" \
  && ok "empty KV budget profiles, utilization gate kept" || bad "KV_CACHE_MEMORY_BYTES handling"
c=$(PREFIX_CACHING=0 run 0)
has "$c" --no-enable-prefix-caching && ! has "$c" --prefix-cache-retention-interval && ok "PREFIX_CACHING=0" || bad "PREFIX_CACHING=0"
c=$(TEXT_ONLY=1 run 0) || bad "TEXT_ONLY=1 print-serve failed"
has "$c" --language-model-only && ! has "$c" --limit-mm-per-prompt && ok "TEXT_ONLY=1 drops the vision encoder" || bad "TEXT_ONLY"
c=$(KERNEL_BACKEND=b12x run 0)
has "$c" "--attention-backend B12X" && has "$c" "--linear-backend b12x" && has "$c" "--moe-backend b12x" \
  && has "$c" 'attention_backend\":\"B12X' && has "$c" VLLM_DS41_KERNEL_BACKEND=b12x && ! has "$c" VLLM_L2_PREFETCH_KERNELS \
  && ok "KERNEL_BACKEND=b12x switches every kernel flag" || bad "KERNEL_BACKEND=b12x"
c=$(CPUSET= CONTAINER_MEMORY= DRI_CARD= run 0)
! has "$c" --cpuset-cpus && ! has "$c" --memory && ! has "$c" /dev/dri && ok "CPUSET, CONTAINER_MEMORY, DRI_CARD empty drop their flags" \
  || bad "empty host-resource keys"
c=$(DISPLAY_CARVEOUT=1 run 0)
has "$c" SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1 && ok "DISPLAY_CARVEOUT=1" || bad "DISPLAY_CARVEOUT=1"

# The key reaches vLLM as VLLM_API_KEY copied from the remote shell env: named on rank 0 only, never valued in argv.
keyok=1
for r in 0 1 2 3; do
  c=$(API_KEY=k serve "$r") || { bad "API_KEY=k print-serve $r failed"; keyok=0; continue; }
  if (( r == 0 )); then has "$c" "-e VLLM_API_KEY " || keyok=0; else has "$c" VLLM_API_KEY && keyok=0; fi
  has "$c" --api-key && keyok=0
  tr ' ' '\n' <<<"$c" | grep -qxE 'k|[A-Z_]+=k' && keyok=0
done
(( keyok )) && ok "API_KEY enforced via env on rank 0, never in argv" || bad "API_KEY handling"

# memory guard values flow into the arm line; the steady guard must stay below the startup guard
g=$(STARTUP_MEMGUARD_GIB=6 STARTUP_MEMGUARD_INTERVAL=0.5 STARTUP_MEMGUARD_WAIT=90 guard 2)
[[ $g == *" c startup 6 0.5 90 prelaunch" ]] && ok "memory guard keys reach memguard-start.sh" || bad "memory guard keys: $g"

# invalid config must fail without printing a partial command
for bogus in KERNEL_BACKEND=bogus DRAFT_SAMPLE_METHOD=bogus ADAPTIVE_VERIFICATION_ROW_COST=1.5 NUM_SPECULATIVE_TOKENS=0 \
             MAX_NUM_BATCHED_TOKENS=96 RING_GID_0=x RING_MGMT_2= MEMGUARD_GIB=5 STARTUP_MEMGUARD_INTERVAL=0 \
             "DISPLAY_CARVEOUT=1 DRI_CARD=" CONTAINER=bad/name; do
  if c=$(env $bogus ./dsv41.sh print-serve 0 2>/dev/null); then bad "$bogus print-serve succeeded"
  elif [[ -n $c ]]; then bad "$bogus printed a partial command"
  else ok "$bogus rejected"; fi
done

# precedence: caller env > .env > dsv41.sh defaults (a scratch copy, so the real .env is not involved)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp dsv41.sh "$tmp/"; cp -R docker "$tmp/docker"
{ echo 'PORT=1111'; echo 'MAX_NUM_SEQS=7   # from .env'; echo 'GENERATION_DEFAULTS={"temperature":0.6}'
  echo 'RING_PREFIX=24'
  for r in 0 1 2 3; do echo "RING_HCA_PREV_$r=p$r"; echo "RING_HCA_NEXT_$r=n$r"; echo "RING_GID_$r=3"; echo "RING_MGMT_$r=e$r"; done
} >"$tmp/.env"
c=$(cd "$tmp" && env -i PATH="$PATH" HOME="$HOME" PORT=2222 ./dsv41.sh print-serve 1 | sed -n 2p) || bad "precedence print-serve failed"
has "$c" "--port 2222" && has "$c" "--max-num-seqs 7" && has "$c" 'temperature\":0.6' && has "$c" "--max-model-len 1048576" \
  && has "$c" "dsv41-1017-ds41f:r6w /models " && has "$c" 'SPARKNET_ROCE_PEER_HCAS=\{\"0\":\[\"p1\"\]\,\"2\":\[\"n1\"\]\}' \
  && ok "precedence: env > .env > defaults (ring pins from .env)" || bad "precedence: $c"

# the shipped site configuration renders for every rank without ssh
for r in 0 1 2 3; do
  c=$(cd "$tmp" && cp "$ROOT/.env.example" .env && env -i PATH="$PATH" HOME="$HOME" ./dsv41.sh print-serve "$r") \
    || { bad ".env.example print-serve $r failed"; continue; }
  [[ $(wc -l <<<"$c" | tr -d ' ') == 2 ]] || bad ".env.example rank $r: expected guard + docker lines"
done
c=$(cd "$tmp" && env -i PATH="$PATH" HOME="$HOME" ./dsv41.sh print-serve 0 | sed -n 2p)
has "$c" "NCCL_IB_HCA==rocep1s0f1\,roceP2p1s0f1" && has "$c" "SPARKNET_ROCE_HCA=rocep1s0f1\,roceP2p1s0f1" \
  && has "$c" "SPARKNET_ROCE_TOPOLOGY=direct" && has "$c" "NCCL_IB_TC=106" && has "$c" "NCCL_SOCKET_IFNAME=enP7s7" \
  && ok ".env.example renders the site's switch without ssh" || bad ".env.example switch: $c"

# weight snapshots: every rank must get the same switch (the fail-closed restore all-reduces across ranks; a rank
# that disagrees leaves the others waiting at boot), on by default and off with WEIGHT_SNAPSHOT=0
for want in 1 0; do
  vals=""
  for r in 0 1 2 3; do
    c=$(cd "$tmp" && env -i PATH="$PATH" HOME="$HOME" $([[ $want == 0 ]] && echo WEIGHT_SNAPSHOT=0) ./dsv41.sh print-serve "$r" | sed -n 2p)
    vals+="$(envval "$c" DSV41_WEIGHT_SNAPSHOT) "
  done
  [[ $vals == "$want $want $want $want " ]] && ok "DSV41_WEIGHT_SNAPSHOT=$want on every rank" || bad "DSV41_WEIGHT_SNAPSHOT per rank: $vals(want $want)"
done

# KV snapshots: every rank must get the same switch (the engine core installs the restored prefix table only if
# every rank restored its buffer, and saves through every rank), on by default and off with KV_PERSIST=0
for want in 0 1; do
  vals=""
  for r in 0 1 2 3; do
    c=$(cd "$tmp" && env -i PATH="$PATH" HOME="$HOME" $([[ $want == 0 ]] && echo KV_PERSIST=0) ./dsv41.sh print-serve "$r" | sed -n 2p)
    vals+="$(envval "$c" DSV41_KV_PERSIST) "
  done
  [[ $vals == "$want $want $want $want " ]] && ok "DSV41_KV_PERSIST=$want on every rank" || bad "DSV41_KV_PERSIST per rank: $vals(want $want)"
done

# pinned memory-guard scripts: the copy start installs must be spark-ds41f's own at the pinned commit
grep -q '^GUARD_REV=2dc95329610188451a5c5f5af98e5c392144809c$' dsv41.sh && ok "memory guard pinned to spark-ds41f@2dc95329" \
  || bad "GUARD_REV pin"

# pinned checkpoint manifest: every shard of the index plus config, tokenizer and encoder
m=config/DeepSeek-V4.1-Flash-1017.sha256
if awk 'NF != 2 || length($1) != 64 {exit 1}' "$m" && [[ $(grep -c '\.safetensors$' "$m") == 48 ]] \
   && grep -qE ' (config\.json|tokenizer\.json|model\.safetensors\.index\.json|encoding/encoding\.py)$' "$m"; then
  ok "checkpoint manifest $m"
else bad "checkpoint manifest $m"; fi

(( fails == 0 )) && echo "OK" || { echo "FAILED: $fails"; exit 1; }
