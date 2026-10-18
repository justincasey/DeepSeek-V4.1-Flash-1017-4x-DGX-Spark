#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# dsv41.sh: DeepSeek-V4.1-Flash-1017 on four DGX Sparks, TP4 over the switchless ring, on Christopher Owen's
# spark-ds41f engine (r6w: vLLM + TileLang kernels + sparknet one-shot collectives + patched NCCL, docker/Dockerfile).
# Run from any control machine with key-based ssh to the four nodes. Nothing runs on the nodes outside docker,
# except the Hugging Face client for `fetch` and the memory guard (a systemd unit per node, see start).
#
# Usage: ./dsv41.sh <command> [args]
#   doctor               read-only: ring (HCAs, GID indices, MTU), ssh, docker, image, GPU idle, memory, page size,
#                        sysctls, fast cores, DRM card, sudo/systemd-run, checkpoint, Engram shard layout
#   fetch                download the pinned checkpoint on rank 0, fan it out over the ring, sha256-verify every node
#   verify               sha256-verify the checkpoint on every node (no download)
#   build                build the serving image on rank 0, ship it over the ring, verify the image ID everywhere
#   start                arm a memory guard and start each rank (workers first), wait for the API, switch the
#                        guards to steady, run smoke
#   stop | restart | status | smoke
#   logs [rank] [lines]  tail one rank's engine log (default rank 0, 200 lines)
#   api-test [args]      full API conformance suite against the live endpoint (tests/api_conformance.py)
#   parity               server prompt rendering == publisher encoding.py (tests/test_encoding_parity.py)
#   bench STEP [args]    decode | prefill | needle | spec | pin | check | all (bench/bench.py); results in bench/runs/
#   stop-other           run STOP_OTHER: take down the stack that normally owns these GPUs
#   restore-other        stop this stack, then run RESTORE_OTHER
#   print-serve RANK     print the exact memory-guard arm + docker run + vllm serve commands start runs on one rank
#                        (no side effects; no ssh when the RING_* pins are set)
#   help
#
# Configuration: environment > .env > defaults below. Every key is documented in .env.example.
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 )); then echo "dsv41.sh needs bash >= 4 (macOS: brew install bash)" >&2; exit 1; fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ── configuration: caller env > .env > defaults ────────────────────────────────────────────────
declare -A _CALLER=()
while IFS= read -r _k; do _CALLER[$_k]=1; done < <(compgen -e)
if [[ -f $ROOT/.env ]]; then
  while IFS= read -r _line || [[ -n $_line ]]; do
    [[ $_line =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _k=${BASH_REMATCH[1]} _v=${BASH_REMATCH[2]}
    [[ -n ${_CALLER[$_k]:-} ]] && continue
    _v=${_v#"${_v%%[![:space:]]*}"}
    if [[ $_v =~ ^\"(.*)\"[[:space:]]*(#.*)?$ || $_v =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then _v=${BASH_REMATCH[1]}
    else   # '#' opens a comment only at the start of the value or after whitespace: API_KEY=abc#123 keeps it
      _v=${_v%%[[:space:]]#*}; [[ $_v == \#* ]] && _v=
      _v=${_v%"${_v##*[![:space:]]}"}
    fi
    printf -v "$_k" '%s' "$_v"; export "${_k?}"
  done < "$ROOT/.env"
fi
unset _CALLER _k _v _line

# Defaults are spark-ds41f r6w at shape tp4-ring (config/cluster.json @ 2dc95329, unchanged since cbf47749), as its
# site receipt experiments/2026-10-17-site-tp4-4k-1017 ran r6v on a stock 4 KiB kernel.
: "${NODES:=192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18}"   # management addresses, ring cable order, rank 0 first
: "${SSH_USER:=spark}"
: "${MODELS_DIR:=/home/spark/models}"
: "${MODEL_REPO:=deepseek-ai/DeepSeek-V4.1-Flash-1017}"
: "${MODEL_REVISION:=1bae0e1088993b9b82de08f59de5be5a34e42170}"
: "${MODEL_NAME:=DeepSeek-V4.1-Flash-1017}"
: "${CHECKSUMS:=config/DeepSeek-V4.1-Flash-1017.sha256}"
: "${HF_BIN:=/home/spark/venvs/hf/bin/hf}"
: "${IMAGE:=dsv41-1017-ds41f:r6w}"
: "${BUILD_DIR:=/home/spark/build/dsv41-1017-ds41f}"
: "${CACHE_DIR:=/home/spark/.cache/dsv41-ds41f}"   # mounted at /cache: JIT, TileLang, DSpark cost curves
: "${CONTAINER:=dsv41-1017}"
: "${SERVED_NAME:=deepseek-v4.1-flash}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${API_KEY:=}"
: "${MASTER_PORT:=9999}"
: "${MAX_MODEL_LEN:=1048576}"
: "${MAX_NUM_SEQS:=16}"
: "${MAX_NUM_BATCHED_TOKENS:=24576}"
: "${KV_CACHE_MEMORY_BYTES=11274289152}"   # 10.5 GiB per rank; set empty to profile
: "${GPU_MEMORY_UTILIZATION:=0.82}"
: "${PREFIX_CACHING:=1}"
: "${PREFIX_CACHE_RETENTION_INTERVAL:=8192}"
: "${KERNEL_BACKEND:=tilelang}"          # tilelang (spark-ds41f r6 onward) | b12x
: "${DSPARK:=1}"
: "${NUM_SPECULATIVE_TOKENS:=6}"
: "${DRAFT_SAMPLE_METHOD:=greedy}"
: "${ADAPTIVE_VERIFICATION_COST_SCALE=2.0}"   # empty = no adaptive verification
: "${DSPARK_COST_DIR:=}"                 # empty = one per build under /cache | off | absolute path under /cache
: "${EAGER:=0}"
: "${COMPILATION_CONFIG:=}"
[[ -v GENERATION_DEFAULTS ]] || GENERATION_DEFAULTS='{"temperature":1.0,"top_p":1.0}'   # empty = vLLM's own
: "${TEXT_ONLY:=0}"
: "${MAX_IMAGES:=4}"
: "${DISPLAY_CARVEOUT:=0}"
: "${DRI_CARD=/dev/dri/by-path/pci-000f:01:00.0-card}"
: "${CPUSET=5-9,15-19}"
: "${CONTAINER_MEMORY=112g}"
: "${GPU_CLOCKS:=2200}"                  # site policy: lock every node's GPU at this MHz at start (nvidia-smi -lgc N,N) | keep
: "${WEIGHT_SNAPSHOT:=1}"                # 1 = per-rank loaded-weight snapshots under CACHE_DIR (docker/0002): later boots
                                         #     read one file at NVMe speed instead of the checkpoint | 0 = always the checkpoint
: "${STARTUP_MEMGUARD_GIB:=5}"
: "${STARTUP_MEMGUARD_INTERVAL:=0.25}"
: "${STARTUP_MEMGUARD_WAIT:=120}"
: "${MEMGUARD_GIB:=3}"
: "${MEMGUARD_INTERVAL:=2}"
: "${EXTRA_VLLM_ARGS:=}"
: "${EXTRA_DOCKER_ARGS:=}"
: "${WAIT_TIMEOUT:=3600}"
: "${STOP_GRACE:=5}"                     # seconds docker stop waits for vLLM's graceful exit before SIGKILL
: "${STOP_OTHER:=}"
: "${RESTORE_OTHER:=}"
: "${ALLOWED_CONTAINERS:=}"            # other containers that may keep running alongside the ranks (space separated)

read -r -a NODE <<<"$NODES"
(( ${#NODE[@]} == 4 )) || { echo "NODES must list exactly four management addresses (got ${#NODE[@]})" >&2; exit 1; }
MASTER=${NODE[0]}
MODEL_DIR=$MODELS_DIR/$MODEL_NAME
URL="http://$MASTER:$PORT"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
# The Engram tables: the io_uring row reader needs them as regular files on local disk on every node.
ENGRAM_SHARDS="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"
# The memory guard is spark-ds41f's own scripts/memguard-start.sh + scripts/memguard.sh, which the image carries
# unmodified from the pinned commit; start copies them out of the image and refuses any other content.
GUARD_SRC=/opt/spark-ds41f/scripts
GUARD_REV=2dc95329610188451a5c5f5af98e5c392144809c
GUARD_SUMS="c70a379efec2e48858e86c77ab8eb90cc748700f702afe663480d92fa5c1ab09  memguard.sh
a1c0cb8e3d78eb3b22c457bb7fba8087f79e9ac31342dc38106f7cde65bd5f15  memguard-start.sh"
GUARD_UNIT=$CONTAINER-memguard.service
DSPARK_COST_ROOT=/cache/kkref/dspark-costs

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }
is_true() { case ${1:-} in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi | cut -d' ' -f1; }

# remote RANK SCRIPT [VAR=value ...]: run SCRIPT under `bash -s` on one rank with the variables exported.
# The script travels on ssh's stdin, so it runs as one brace group with stdin from /dev/null: bash parses the
# whole group before running it, and no child that reads stdin (an inner ssh, docker) can eat the rest of it.
remote() {
  local r=$1 script=$2 kv prelude="set -euo pipefail"$'\n'; shift 2
  for kv in "$@"; do prelude+="export ${kv%%=*}=$(printf '%q' "${kv#*=}")"$'\n'; done
  ssh "${SSH_OPTS[@]}" "$SSH_USER@${NODE[$r]}" bash -s <<<"$prelude{"$'\n'"$script"$'\n'"} </dev/null"
}
# every SCRIPT [VAR=value ...]: run on all ranks in parallel; output printed per rank in order, prefixed [rN];
# fails if any rank fails.
every() {
  local r rc=0 tmp; tmp=$(mktemp -d)
  local -a ep=()
  # `&& || echo $?` so the rc file is written even though set -e would end the subshell at a failed remote.
  for r in 0 1 2 3; do ( remote "$r" "$@" >"$tmp/$r" 2>&1 && echo 0 >"$tmp/$r.rc" || echo $? >"$tmp/$r.rc" ) & ep+=($!); done
  wait "${ep[@]}" || true
  for r in 0 1 2 3; do sed "s/^/[r$r] /" "$tmp/$r"; [[ $(<"$tmp/$r.rc") == 0 ]] || rc=1; done
  rm -rf "$tmp"
  return $rc
}

# ── the ring ────────────────────────────────────────────────────────────────────────────────────
# Per node: every up RoCE netdev with exactly one IPv4 address, as "subnet/prefix hca netdev addr gid mtu
# active_mtu" (gid: index of the netdev's IPv4 RoCE v2 GID, or none), plus the management netdev that routes to
# rank 0. Devices are matched by subnet, never by name, so any port-to-neighbour cabling works.
SCAN_SCRIPT='
for d in /sys/class/net/*; do
  n=${d##*/}; [ -d "$d/device/infiniband" ] || continue
  [ "$(cat "$d/operstate" 2>/dev/null)" = up ] || continue
  hca=$(ls "$d/device/infiniband" | head -1)
  mapfile -t a < <(ip -4 -o addr show dev "$n" | awk "{print \$4}")
  [ ${#a[@]} -eq 1 ] || { echo "@bad $n ${#a[@]}"; continue; }
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "${a[0]}")
  ip4=${a[0]%/*}
  want=$(printf "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" ${ip4//./ })
  p=/sys/class/infiniband/$hca/ports/1 gid=none
  for g in "$p"/gids/*; do
    [ "$(cat "$g" 2>/dev/null)" = "$want" ] || continue
    [ "$(cat "$p/gid_attrs/types/${g##*/}" 2>/dev/null | tr -d " ")" = RoCEv2 ] && { gid=${g##*/}; break; }
  done
  amtu=$(ibv_devinfo -d "$hca" -i 1 2>/dev/null | awk "/active_mtu/ {print \$2; exit}" || true)
  echo "$net $hca $n $ip4 $gid $(cat "$d/mtu") ${amtu:-?}"
done
mg=$(ip -4 -o addr show | awk -v ip="$MASTER" "index(\$4, ip \"/\") == 1 {print \$2; exit}")
[ -n "$mg" ] || mg=$(ip -4 -o route get "$MASTER" | sed -n "s/.* dev \([^ ]*\).*/\1/p")
echo "@mgmt $mg"'
# Per rank r: NETS (the subnets of the cable(s) to rank r+1, sorted, so both ends list lanes in the same order),
# LINK (the first of them), HCA_PREV/HCA_NEXT (local HCAs towards rank r-1/r+1 in lane order), GID (the one
# IPv4 RoCE v2 GID index of those HCAs), MGMT (bootstrap netdev), PEER_IP (rank r+1's address on LINK).
declare -a SCAN=() MGMT=() NETS=() LINK=() PEER_IP=() HCA_PREV=() HCA_NEXT=() GID=()
PREFIX=
shared_nets() {
  comm -12 <(awk '$1 !~ /^@/ {print $1}' <<<"${SCAN[$1]}" | LC_ALL=C sort) <(awk '$1 !~ /^@/ {print $1}' <<<"${SCAN[$2]}" | LC_ALL=C sort) \
    | paste -sd' ' -
}
hcas_on() { local r=$1 net; for net in $2; do awk -v net="$net" '$1 == net {print $2; exit}' <<<"${SCAN[$r]}"; done | paste -sd, -; }
detect_ring() {
  local r n p net g bad=0
  for r in 0 1 2 3; do
    SCAN[$r]=$(remote "$r" "$SCAN_SCRIPT" MASTER="$MASTER") || die "ssh to ${NODE[$r]} failed"
    MGMT[$r]=$(awk '$1 == "@mgmt" {print $2}' <<<"${SCAN[$r]}")
    grep -q '^@bad' <<<"${SCAN[$r]}" && { echo "rank $r: RoCE port with !=1 IPv4 address: $(grep '^@bad' <<<"${SCAN[$r]}" | tr '\n' ' ')"; bad=1; }
  done
  for r in 0 1 2 3; do n=$(( (r + 1) % 4 ))
    NETS[$r]=$(shared_nets "$r" "$n"); LINK[$r]=${NETS[$r]%% *}
    [[ -n ${LINK[$r]} ]] || { echo "no cable between rank $r (${NODE[$r]}) and rank $n (${NODE[$n]}): NODES must be in cable order"; bad=1; }
  done
  for r in 0 1; do
    [[ -z $(shared_nets "$r" $((r + 2))) ]] || { echo "rank $r and rank $((r + 2)) share a subnet: this is not a switchless ring"; bad=1; }
  done
  (( bad == 0 )) || return 1
  PREFIX=${LINK[0]#*/}
  for r in 0 1 2 3; do n=$(( (r + 1) % 4 )); p=$(( (r + 3) % 4 ))
    for net in ${NETS[$r]}; do [[ ${net#*/} == "$PREFIX" ]] || { echo "cable subnets differ in prefix length ($net)"; bad=1; }; done
    HCA_PREV[$r]=$(hcas_on "$r" "${NETS[$p]}") HCA_NEXT[$r]=$(hcas_on "$r" "${NETS[$r]}")
    PEER_IP[$r]=$(awk -v net="${LINK[$r]}" '$1 == net {print $4; exit}' <<<"${SCAN[$n]}")   # rank n's address on cable r->n
    # NCCL and sparknet take one GID index per process: every ring HCA of the node must hold its IPv4 RoCE v2
    # address at the same index.
    g=$(for net in ${NETS[$p]} ${NETS[$r]}; do awk -v net="$net" '$1 == net {print $5; exit}' <<<"${SCAN[$r]}"; done | sort -u | paste -sd' ' -)
    if [[ $g == none || $g == *' '* ]]; then echo "rank $r: ring HCAs have no common IPv4 RoCE v2 GID index (got: $g)"; bad=1; fi
    GID[$r]=$g
    [[ -n ${MGMT[$r]} && ${MGMT[$r]} != lo ]] || { echo "rank $r: no management route to $MASTER"; bad=1; }
  done
  return $bad
}
# ring_config: RoCE mapping and bootstrap interface for every rank, from RING_* pins in .env when RING_PREFIX is
# set (no ssh), else from detect_ring. Copies over the cables (fetch, build) always scan.
ring_config() {
  if [[ -n ${RING_PREFIX:-} ]]; then
    local r k
    PREFIX=$RING_PREFIX
    for r in 0 1 2 3; do
      for k in RING_HCA_PREV_$r RING_HCA_NEXT_$r RING_GID_$r RING_MGMT_$r; do
        [[ -n ${!k:-} ]] || die "RING_PREFIX is set, so $k must be set too"
      done
      k=RING_HCA_PREV_$r; HCA_PREV[$r]=${!k}
      k=RING_HCA_NEXT_$r; HCA_NEXT[$r]=${!k}
      k=RING_GID_$r; GID[$r]=${!k}
      k=RING_MGMT_$r; MGMT[$r]=${!k}
    done
    return 0
  fi
  detect_ring
}
# ring_verify: after ring_config loaded RING_* pins, scan the live ring and print every pin that differs.
# Returns 0 if all match (the arrays then hold the live values), 1 on a difference, 2 if the scan failed.
ring_verify() {
  local -a prev=("${HCA_PREV[@]}") next=("${HCA_NEXT[@]}") gid=("${GID[@]}") mgmt=("${MGMT[@]}")
  local prefix=$PREFIX r bad=0
  detect_ring || return 2
  [[ $PREFIX == "$prefix" ]] || { echo "RING_PREFIX=$prefix, live $PREFIX"; bad=1; }
  for r in 0 1 2 3; do
    [[ ${HCA_PREV[$r]} == "${prev[$r]}" ]] || { echo "RING_HCA_PREV_$r=${prev[$r]}, live ${HCA_PREV[$r]}"; bad=1; }
    [[ ${HCA_NEXT[$r]} == "${next[$r]}" ]] || { echo "RING_HCA_NEXT_$r=${next[$r]}, live ${HCA_NEXT[$r]}"; bad=1; }
    [[ ${GID[$r]} == "${gid[$r]}" ]] || { echo "RING_GID_$r=${gid[$r]}, live ${GID[$r]}"; bad=1; }
    [[ ${MGMT[$r]} == "${mgmt[$r]}" ]] || { echo "RING_MGMT_$r=${mgmt[$r]}, live ${MGMT[$r]}"; bad=1; }
  done
  return $bad
}
# ring_env RANK: the rank's NCCL and sparknet fabric settings (spark-ds41f scripts/topology.py node_environment):
# NCCL gets the node's ring HCAs as an exact-match set, sparknet each neighbour rank's HCAs in lane order.
ring_env() {
  local r=$1 p n hcas peers
  p=$(( (r + 3) % 4 )) n=$(( (r + 1) % 4 ))
  [[ ${GID[$r]} =~ ^[0-9]+$ ]] || die "rank $r: RoCE GID index '${GID[$r]}' is not a number"
  hcas=$(tr ',' '\n' <<<"${HCA_PREV[$r]},${HCA_NEXT[$r]}" | grep . | LC_ALL=C sort -u | paste -sd, -)
  peers=$(printf '{"%s":[%s],"%s":[%s]}' "$p" "$(sed 's/[^,][^,]*/"&"/g' <<<"${HCA_PREV[$r]}")" \
    "$n" "$(sed 's/[^,][^,]*/"&"/g' <<<"${HCA_NEXT[$r]}")")
  printf '%s\n' -e "NCCL_IB_HCA==$hcas" -e "NCCL_IB_GID_INDEX=${GID[$r]}" -e "NCCL_IB_SUBNET_PREFIX_LEN=$PREFIX" \
    -e "SPARKNET_ROCE_PEER_HCAS=$peers" -e "SPARKNET_ROCE_GID_INDEX=${GID[$r]}" \
    -e "NCCL_SOCKET_IFNAME=${MGMT[$r]}" -e "GLOO_SOCKET_IFNAME=${MGMT[$r]}" -e "TP_SOCKET_IFNAME=${MGMT[$r]}" \
    -e "VLLM_HOST_IP=${NODE[$r]}"
}

# ── serve command ───────────────────────────────────────────────────────────────────────────────
# spark-ds41f scripts/recipe.py: decode verifies k+1 rows per sequence with DSpark, so CUDA graphs are captured up
# to MAX_NUM_SEQS x (k+1) rows on his ladder (1-4, 6, 8, every 4th to 32, then every 8th), and the long-prefill
# threshold leaves those rows of the batch budget to decode.
graph_rows() { if is_true "$DSPARK"; then echo $(( MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1) )); else echo "$MAX_NUM_SEQS"; fi; }
capture_sizes() {
  local max=$1 s out=""
  for s in 1 2 3 4 6 8 12 16 20 24 28 32; do (( s <= max )) && out+="${out:+,}$s"; done
  for (( s = 40; s <= max; s += 8 )); do out+=",$s"; done
  echo "$out"
}
kernel_backend() {   # the three serve flags and the attention backend name of KERNEL_BACKEND (scripts/kernel_backend.py)
  case $KERNEL_BACKEND in
    tilelang) echo "TILELANG tilelang" ;;
    b12x) echo "B12X b12x" ;;
    *) die "KERNEL_BACKEND must be tilelang or b12x (got '$KERNEL_BACKEND')" ;;
  esac
}
serve_common() {  # the rank-independent `vllm serve` argv (the image entrypoint is `vllm serve`)
  local rows attn lin spec kb
  kb=$(kernel_backend) || exit 1
  read -r attn lin <<<"$kb"
  [[ $NUM_SPECULATIVE_TOKENS =~ ^[1-9][0-9]*$ ]] || die "NUM_SPECULATIVE_TOKENS must be a positive integer"
  [[ $MAX_NUM_SEQS =~ ^[1-9][0-9]*$ && $MAX_NUM_BATCHED_TOKENS =~ ^[1-9][0-9]*$ ]] || die "MAX_NUM_SEQS and MAX_NUM_BATCHED_TOKENS must be positive integers"
  rows=$(graph_rows)
  (( MAX_NUM_BATCHED_TOKENS > rows )) || die "MAX_NUM_BATCHED_TOKENS must exceed the decode rows ($rows)"
  printf '%s\n' /models \
    --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT" \
    --tokenizer-mode deepseek_v41 --dtype bfloat16 --tensor-parallel-size 4 --load-format b12x --block-size 256
  if is_true "$PREFIX_CACHING"; then echo --enable-prefix-caching; else echo --no-enable-prefix-caching; fi
  printf '%s\n' --enable-chunked-prefill \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-parallel-prefills 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  # The utilization fraction also gates vLLM's startup free-memory check, so it is always passed; a fixed KV
  # budget replaces profiling-based KV sizing on top of it.
  [[ -n $KV_CACHE_MEMORY_BYTES ]] && printf '%s\n' --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  is_true "$EAGER" || printf '%s\n' --max-cudagraph-capture-size "$rows"
  # Requests that set no sampling get GENERATION_DEFAULTS, not the checkpoint's generation_config.json.
  echo --generation-config; echo vllm
  [[ -n $GENERATION_DEFAULTS ]] && printf '%s\n' --override-generation-config "$GENERATION_DEFAULTS"
  # Engram tables read from disk (io_uring row reader), projections sharded over TP.
  printf '%s\n' --engram-config '{"table_memory":"disk","disk_resident_scales":false,"projection_tp":true}' \
    --attention-backend "$attn" --linear-backend "$lin" --moe-backend "$lin"
  if is_true "$DSPARK"; then
    case $DRAFT_SAMPLE_METHOD in greedy|probabilistic) ;; *) die "DRAFT_SAMPLE_METHOD must be greedy or probabilistic" ;; esac
    spec=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_tensor_parallel_size":4,"attention_backend":"%s","draft_sample_method":"%s","rejection_sample_method":"block"' \
      "$NUM_SPECULATIVE_TOKENS" "$attn" "$DRAFT_SAMPLE_METHOD")
    if [[ -n $ADAPTIVE_VERIFICATION_COST_SCALE ]]; then
      spec+=$(printf ',"enable_adaptive_verification":true,"adaptive_verification_cost_scale":%s' "$ADAPTIVE_VERIFICATION_COST_SCALE")
    fi
    printf '%s\n' --speculative-config "$spec}"
  fi
  printf '%s\n' --jit-monitor-mode warn
  if is_true "$EAGER"; then
    echo --enforce-eager
  elif [[ -n $COMPILATION_CONFIG ]]; then
    printf '%s\n' --compilation-config "$COMPILATION_CONFIG"
  else
    printf '%s\n' --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[$(capture_sizes "$rows")]}"
  fi
  is_true "$PREFIX_CACHING" && printf '%s\n' --prefix-cache-retention-interval "$PREFIX_CACHE_RETENTION_INTERVAL"
  printf '%s\n' --long-prefill-token-threshold $(( MAX_NUM_BATCHED_TOKENS - rows )) --async-scheduling \
    --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --distributed-executor-backend mp --nnodes 4 --no-scheduler-reserve-full-isl --no-enable-flashinfer-autotune
  if is_true "$TEXT_ONLY"; then echo --language-model-only
  else printf '%s\n' --limit-mm-per-prompt "{\"image\":$MAX_IMAGES}"; fi
  printf '%s\n' --mm-processor-cache-gb 0   # vLLM's 4 GiB default host cache would come out of the guard margin
  if [[ -n $EXTRA_VLLM_ARGS ]]; then local -a extra; read -r -a extra <<<"$EXTRA_VLLM_ARGS"; printf '%s\n' "${extra[@]}"; fi
  return 0
}
serve_args() {   # serve_args RANK: the container argv after the image name
  local r=$1
  serve_common
  printf '%s\n' --node-rank "$r" --master-addr "$MASTER" --master-port "$MASTER_PORT"
  (( r > 0 )) && echo --headless
  return 0
}

# profile_env: the rank-independent engine environment, spark-ds41f config/cluster.json at shape tp4-ring (the
# shared "environment" plus the shape's NCCL ring profile). NCCL is the image's patched 2.30.7 (bidirectional
# switchless rings); nothing on the host is mounted over it.
profile_env() {
  local e
  kernel_backend >/dev/null || exit 1
  local -a env=(
    # B12X (checkpoint loader, and kernels when KERNEL_BACKEND=b12x)
    B12X_AUTOTUNE=0 B12X_BIND_COMPILE_WORKERS=2 B12X_STATE_COMPILE_WORKERS=2 B12X_WEIGHTS_COMPILE_WORKERS=4
    B12X_COMPILE_CACHE_DIR=/cache/kkref/jit/b12x/compile B12X_CUTE_COMPILE_CACHE_DIR=/cache/kkref/jit/b12x/cute
    B12X_DENSE_SPLITK_TURBO=1 B12X_W4A8_TINY_DECODE=0
    # CUDA / compiler caches, all under the persistent /cache mount
    CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH=/cache/kkref/jit/nv-compute CUDA_DEVICE_MAX_CONNECTIONS=32
    CUTE_DSL_ARCH=sm_121a CUTE_DSL_CACHE_DIR=/cache/kkref/jit/cute-dsl FLASHINFER_WORKSPACE_BASE=/cache/kkref/flashinfer
    TILELANG_CACHE_DIR=/cache/kkref/jit/tilelang TORCHINDUCTOR_CACHE_DIR=/cache/kkref/jit/torchinductor
    TRITON_CACHE_DIR=/cache/kkref/jit/triton XDG_CACHE_HOME=/cache/kkref/jit
    VLLM_CACHE_DIR=/cache/kkref/jit/vllm-r5o VLLM_CACHE_ROOT=/cache/kkref/jit/vllm-r5o
    HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    MAX_JOBS=2 OMP_NUM_THREADS=8
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 SAFETENSORS_FAST_GPU=1
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # NCCL: sparknet's tp4-ring profile (neighbour ring only: no tree/PAT/NVLS, runtime connect)
    NCCL_ALGO=Ring NCCL_BUFFSIZE=4194304 NCCL_COLLNET_ENABLE=0 NCCL_CROSS_NIC=1 NCCL_CUMEM_ENABLE=1
    NCCL_CUMEM_HOST_ENABLE=0 NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=INIT NCCL_GIN_ENABLE=0 NCCL_IB_ADDR_FAMILY=AF_INET
    NCCL_IB_DISABLE=0 NCCL_IB_MERGE_NICS=1 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_SUBNET_AWARE_ROUTING=1
    NCCL_IGNORE_CPU_AFFINITY=1 NCCL_LL128_BUFFSIZE=262144 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4
    NCCL_MIN_TRAFFIC_PER_CHANNEL=512 NCCL_MNNVL_ENABLE=0 NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_NVLS_ENABLE=0
    NCCL_P2P_DISABLE=1 NCCL_PAT_ENABLE=0 'NCCL_PROTO=^LL128' NCCL_RMA_DISABLE=1 NCCL_RUNTIME_CONNECT=1
    NCCL_SHM_DISABLE=1 NCCL_SWITCHLESS_BIDIRECTIONAL=2 'NCCL_THREAD_THRESHOLDS=-2 -2 -2 1 1 1'
    # sparknet one-shot collectives over the relay ring (small all-reduce / all-gather)
    SPARKNET_ROCE_ALLGATHER_MAX_BYTES=2097152 SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES=2097152
    SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=2097152 SPARKNET_ROCE_PROXY_CPU=big SPARKNET_ROCE_TIMEOUT_MS=10000
    SPARKNET_ROCE_TOPOLOGY=ring4
    # DSpark drafter
    SPARK3_DSPARK_DEAD_ROWS_TAU=0.2 SPARK3_DSPARK_MAIN_PROJ_TP=1 SPARK3_DSPARK_PROFILE_TOKENS=random
    SPARK3_DSPARK_VOCAB_PARALLEL=1
    "SPARK3_DISPLAY_CARVEOUT_WEIGHTS=$(is_true "$DISPLAY_CARVEOUT" && echo 1 || echo 0)"
    "DSV41_WEIGHT_SNAPSHOT=$(is_true "$WEIGHT_SNAPSHOT" && echo 1 || echo 0)"
    # vLLM
    # Drafter heads in NVFP4 (draft vocab head and Markov head): only the proposals change, every committed token
    # is still the target's; measured +1.3% c=1 decode on this ring (spark-ds41f's TP4 run: c1 level, c8 +3.5%).
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_DISABLE_PYNCCL=0 VLLM_DS41_ATTENTION_COMPUTE=auto
    VLLM_DS41_DRAFT_NVFP4_HEAD=1 VLLM_DS41_HEAD_DTYPE=float32 "VLLM_DS41_KERNEL_BACKEND=$KERNEL_BACKEND"
    VLLM_DS41_MARKOV_NVFP4=1 VLLM_ENABLE_PCIE_ALLREDUCE=0
    VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 VLLM_L2_PREFETCH_GRID=2
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024 VLLM_PLUGINS=b12x_loader
    VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_USE_MEGA_AOT_ARTIFACT=1
    VLLM_USE_STANDALONE_COMPILE=1 VLLM_USE_V2_MODEL_RUNNER=1
  )
  # Side-stream L2 prefetch of TileLang decode projections; the B12X profiles never set it. The packed BF16 vocabulary
  # head runs only through TileLang's logits projection: under B12X the head's apply() refuses it at the first step.
  [[ $KERNEL_BACKEND == tilelang ]] && env+=(VLLM_L2_PREFETCH_KERNELS=tilelang VLLM_DS41_PACKED_BF16_LM_HEAD=1)
  for e in "${env[@]}"; do printf '%s\n' -e "$e"; done
}
# DSpark's verifier prices its steps from cost curves rank 0 times on a boot and pins in SPARK3_DSPARK_COST_DIR.
# As in spark-ds41f (bin/spark build_fingerprint), the default directory is per build: a digest of the image
# recipe (docker/), the tag, the engine environment and the serve arguments, so a changed build or flag profiles
# once on its first boot and never schedules on another build's curves.
build_fingerprint() {
  local docker_digest
  docker_digest=$(cd "$ROOT" && find docker -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort \
    | while IFS= read -r f; do printf '%s %s\n' "$f" "$(sha256 <"$f")"; done | sha256)
  { echo "$docker_digest"; echo "$IMAGE"; profile_env; serve_common; } | sha256 | cut -c1-16
}
cost_env() {
  local dir
  case $DSPARK_COST_DIR in
    off) return 0 ;;
    "") dir=$DSPARK_COST_ROOT/build-$(build_fingerprint) || exit 1 ;;
    /cache/?*) dir=$DSPARK_COST_DIR ;;
    *) die "DSPARK_COST_DIR must be empty, off, or an absolute path under /cache (got '$DSPARK_COST_DIR')" ;;
  esac
  printf '%s\n' -e "SPARK3_DSPARK_COST_DIR=$dir"
}

docker_args() {  # docker_args RANK: docker run argv up to and including the image (spark-ds41f container.docker_run_args)
  local r=$1
  if is_true "$DISPLAY_CARVEOUT" && [[ -z $DRI_CARD ]]; then die "DISPLAY_CARVEOUT=1 needs DRI_CARD (the nvidia DRM card)"; fi
  printf '%s\n' run -d --name "$CONTAINER" --network=host --ipc=host --gpus=all \
    --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK \
    --ulimit=memlock=-1:-1 --ulimit=stack=67108864:67108864 --ulimit=core=0 \
    --security-opt=seccomp=unconfined --security-opt=label=disable --shm-size=16g
  [[ -n $CONTAINER_MEMORY ]] && printf '%s\n' "--memory=$CONTAINER_MEMORY" "--memory-swap=$CONTAINER_MEMORY"
  # The display carve-out allocates through the nvidia DRM card.
  [[ -n $DRI_CARD ]] && printf '%s\n' "--mount=type=bind,source=$DRI_CARD,target=/dev/dri/card0" '--device-cgroup-rule=c 226:* rw'
  [[ -n $CPUSET ]] && printf '%s\n' "--cpuset-cpus=$CPUSET"
  profile_env
  cost_env
  ring_env "$r"
  # The key is named, never valued, here: docker copies it from the remote shell's environment (cmd_start sends
  # it over ssh stdin), so it stays out of every process argv and print-serve. vLLM reads VLLM_API_KEY.
  if (( r == 0 )) && [[ -n $API_KEY ]]; then printf '%s\n' -e VLLM_API_KEY; fi
  printf '%s\n' -v "$CACHE_DIR:/cache:rw" -v "$MODEL_DIR:/models:ro"
  if [[ -n $EXTRA_DOCKER_ARGS ]]; then local -a extra; read -r -a extra <<<"$EXTRA_DOCKER_ARGS"; printf '%s\n' "${extra[@]}"; fi
  echo "$IMAGE"
}
quoted() { local a out=""; while IFS= read -r a; do out+=" $(printf '%q' "$a")"; done; echo "${out# }"; }

# ── commands ───────────────────────────────────────────────────────────────────────────────────
# The ten Cortex-X925 cores: cpu_capacity above the midpoint of the smallest and largest (sparknet's "big" rule).
FAST_CPUS_PY='
import glob, re
caps = {int(re.search(r"cpu(\d+)/cpu_capacity$", p).group(1)): int(open(p).read()) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpu_capacity")}
mid = (min(caps.values()) + max(caps.values())) / 2 if caps else 0
print(",".join(str(c) for c in sorted(caps) if caps[c] > mid))'
cpu_list() {   # cpu_list "5-9,15-19": one CPU number per comma, ascending
  local part lo hi
  local -a parts out=()
  IFS=, read -r -a parts <<<"$1"
  for part in "${parts[@]}"; do lo=${part%-*} hi=${part#*-}; out+=($(seq "$lo" "$hi")); done
  printf '%s\n' "${out[@]}" | sort -n | paste -sd, -
}
cmd_doctor() {
  local bad=0 out r scan=0
  log "ring"
  if [[ -n ${RING_PREFIX:-} ]]; then
    ring_config
    ring_verify || scan=$?
    (( scan != 1 )) || { echo "  FAIL RING_* pins in .env differ from the live ring (above): update them"; bad=1; }
  else
    detect_ring || scan=2
  fi
  if (( scan != 2 )); then
    for r in 0 1 2 3; do
      echo "  rank $r ${NODE[$r]}: NCCL $(ring_env "$r" | grep -o 'NCCL_IB_HCA=.*'), GID index ${GID[$r]}, bootstrap ${MGMT[$r]}"
      echo "    to rank $(( (r + 3) % 4 )) via ${HCA_PREV[$r]}; to rank $(( (r + 1) % 4 )) via ${HCA_NEXT[$r]} (${NETS[$r]}), next rank at ${PEER_IP[$r]}"
      awk '$1 !~ /^@/ {printf "    %-16s %-14s %-15s %-16s gid %-4s mtu %-5s roce_mtu %s\n", $3, $2, $4, $1, $5, $6, $7}' <<<"${SCAN[$r]}"
      awk -v r="$r" '$1 !~ /^@/ && $6 != 9000 {print "  WARN rank " r ": " $3 " MTU " $6 " (the ring runs 9000-byte frames)"}' <<<"${SCAN[$r]}"
      remote "$r" 'ping -c 2 -W 1 "$PEER" >/dev/null' PEER="${PEER_IP[$r]}" || { echo "  FAIL rank $r cannot ping ${PEER_IP[$r]}"; bad=1; }
    done
    [[ -n ${RING_PREFIX:-} ]] || echo "  pin these with RING_* in .env so print-serve needs no ssh (.env.example)"
  else bad=1; fi
  log "nodes"
  out=$(every '
    command -v docker >/dev/null && echo docker=ok || echo docker=missing
    img=$(docker image inspect "$IMAGE" --format "{{.Id}}" 2>/dev/null | grep . | tail -1 || true); echo "image=${img:-none}"
    echo "engine_rev=$(docker image inspect "$IMAGE" --format "{{index .Config.Labels \"local.dsv41.spark-ds41f.revision\"}}" 2>/dev/null || true)"
    echo "gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
    echo "mem_avail_gb=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)"
    echo "nvme_free_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc 0-9)"
    echo "root_free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9) cache_free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)"
    echo "kernel=$(uname -r) pagesize=$(getconf PAGESIZE)"
    echo "watermark_boost=$(cat /proc/sys/vm/watermark_boost_factor) compaction=$(cat /proc/sys/vm/compaction_proactiveness) swappiness=$(cat /proc/sys/vm/swappiness)"
    echo "fast_cpus=$(python3 -c "$FAST_CPUS_PY" 2>/dev/null || true)"
    if [ -z "$DRI" ]; then echo dri=unused; elif [ -e "$DRI" ]; then echo dri=ok; else echo dri=missing; fi
    sudo -n true 2>/dev/null && echo sudo=ok || echo sudo=no
    command -v systemd-run >/dev/null && echo systemd_run=ok || echo systemd_run=missing
    if [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
      need=$(python3 -c "import json,sys; print(len(set(json.load(open(sys.argv[1]))[\"weight_map\"].values())))" "$MODEL_DIR/model.safetensors.index.json")
      have=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l); echo "shards=$have/$need"
    else echo shards=none; fi
    # Engram shards: filesystem and ext4 extent counts (e4defrag -c only reads extent metadata)
    for s in $ENGRAM; do
      f=$(readlink -f "$MODEL_DIR/$s" || true)
      if [ -z "$f" ] || [ ! -f "$f" ]; then echo "engram $s fs=missing extents=-"; continue; fi
      fs=$(stat -f -c %T "$f" 2>/dev/null || echo unknown)
      ext=$(sudo -n e4defrag -c "$f" 2>/dev/null | awk "/Total\/best extents/ {print \$NF; exit}" || true)
      echo "engram $s fs=$fs extents=${ext:-unknown} path=$f"
    done
    systemctl is-active --quiet "$UNIT" && echo guard=active || echo guard=inactive
    docker ps --format "{{.Names}}" | grep -qx "$CONTAINER" && echo running=yes || echo running=no
  ' IMAGE="$IMAGE" MODELS_DIR="$MODELS_DIR" MODEL_DIR="$MODEL_DIR" CONTAINER="$CONTAINER" DRI="$DRI_CARD" \
    FAST_CPUS_PY="$FAST_CPUS_PY" ENGRAM="$ENGRAM_SHARDS" UNIT="$GUARD_UNIT")
  echo "$out" | sed 's/^/  /'
  local img0 want_cpus= o shard fs ext path cur best rootfree
  img0=$(awk '/^\[r0\] image=/{sub(/.*image=/,""); print}' <<<"$out")
  [[ -z $CPUSET ]] || want_cpus=$(cpu_list "$CPUSET")
  for r in 0 1 2 3; do
    o=$(grep "^\[r$r\]" <<<"$out")
    grep -q docker=ok <<<"$o" || { echo "FAIL rank $r: docker missing"; bad=1; }
    grep -q "image=$img0" <<<"$o" && [[ $img0 != none ]] || { echo "FAIL rank $r: image $IMAGE missing or differs from rank 0 (./dsv41.sh build)"; bad=1; }
    grep -q "engine_rev=$GUARD_REV" <<<"$o" || { echo "FAIL rank $r: image $IMAGE is not spark-ds41f@$GUARD_REV (label local.dsv41.spark-ds41f.revision)"; bad=1; }
    grep -qE 'shards=([0-9]+)/\1$' <<<"$o" || { echo "FAIL rank $r: checkpoint incomplete at $MODEL_DIR (./dsv41.sh fetch)"; bad=1; }
    grep -q sudo=ok <<<"$o" || { echo "FAIL rank $r: no passwordless sudo: the memory guard runs as a systemd unit (sudo -n systemd-run)"; bad=1; }
    grep -q systemd_run=ok <<<"$o" || { echo "FAIL rank $r: systemd-run missing (memory guard)"; bad=1; }
    grep -q dri=missing <<<"$o" && { echo "FAIL rank $r: DRI_CARD $DRI_CARD does not exist (ls -l /dev/dri/by-path; or set DRI_CARD= with DISPLAY_CARVEOUT=0)"; bad=1; }
    if [[ -n $CPUSET ]] && ! grep -q "fast_cpus=$want_cpus\$" <<<"$o"; then
      echo "FAIL rank $r: CPUSET $CPUSET is not this node's fast cores ($(grep -o 'fast_cpus=.*' <<<"$o"))"; bad=1
    fi
    # Optional (spark-ds41f recommends 0): a watermark boost hides up to ~0.9 GiB from MemAvailable, which the guard reads.
    grep -q 'watermark_boost=0 ' <<<"$o" || echo "NOTE rank $r: vm.watermark_boost_factor=$(grep -o 'watermark_boost=[0-9]*' <<<"$o" | cut -d= -f2) (optional: 0 keeps MemAvailable exact for the memory guard)"
    rootfree=$(grep -o 'root_free_gb=[0-9]*' <<<"$o" | cut -d= -f2 || true)
    (( r != 0 || ${rootfree:-0} >= 100 )) || echo "WARN rank 0: ${rootfree:-?} GiB free on / (under 100: build and docker images)"
    grep -q 'compaction=0 ' <<<"$o" || echo "WARN rank $r: vm.compaction_proactiveness != 0 (decode stalls)"
    while read -r _ _ shard fs ext path; do   # [rN] engram NAME fs=FS extents=CUR/BEST path=P
      fs=${fs#fs=} ext=${ext#extents=} path=${path#path=}
      case $fs in
        missing) echo "FAIL rank $r: Engram shard $shard missing"; bad=1; continue ;;
        nfs*|fuse*|cifs|smb*) echo "FAIL rank $r: Engram shard $shard is on $fs: the row reader needs a local file"; bad=1 ;;
      esac
      case $ext in
        */*) cur=${ext%/*} best=${ext#*/}
             if (( cur > best )); then echo "WARN rank $r: Engram shard $shard fragmented: $cur extents, best $best (service stopped: sudo e4defrag -v $path)"; fi ;;
        *) echo "WARN rank $r: Engram shard $shard: extent count unreadable (sudo -n e4defrag -c)" ;;
      esac
    done < <(grep "^\[r$r\] engram " <<<"$o" || true)
    if grep -q running=no <<<"$o"; then
      grep -q gpu_procs=0 <<<"$o" || { echo "FAIL rank $r: other GPU processes are running (./dsv41.sh stop-other)"; bad=1; }
      [[ $(grep -o 'mem_avail_gb=[0-9]*' <<<"$o" | cut -d= -f2) -ge 100 ]] || echo "WARN rank $r: under 100 GiB available"
    else
      grep -q guard=active <<<"$o" || echo "WARN rank $r: $CONTAINER runs without its memory guard ($GUARD_UNIT)"
    fi
  done
  (( bad == 0 )) && log "doctor: OK" || die "doctor found blocking problems"
}

# fetch: each rank downloads a size-balanced quarter of the pinned files over its own uplink (a Spark's management
# port is 1 GbE), rank 0 gathers the quarters over the cables, then fans the full tree back out. Resumable.
cmd_fetch() {
  [[ -f $ROOT/$CHECKSUMS ]] || die "missing $CHECKSUMS"
  detect_ring || die "ring detection failed (./dsv41.sh doctor)"
  # Files some rank already holds at full size are not downloaded again; gather and fan-out move them.
  local have; have=$(every 'cd "$MODEL_DIR" 2>/dev/null && find . -maxdepth 2 -type f -not -path "./.cache/*" -printf "%s %P\n" || true' \
    MODEL_DIR="$MODEL_DIR")
  local -a part
  mapfile -t part < <(HAVE="$have" python3 - "$MODEL_REPO" "$MODEL_REVISION" "$ROOT/$CHECKSUMS" <<'PY'
import json, os, re, sys, urllib.request
repo, rev, manifest = sys.argv[1:]
wanted = [line.split(None, 1)[1].strip() for line in open(manifest) if line.strip()]
sizes = {}
def walk(path=""):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}" + (f"/{path}" if path else "")
    for e in json.loads(urllib.request.urlopen(url).read()):
        if e["type"] == "directory": walk(e["path"])
        else: sizes[e["path"]] = e["size"]
walk()
present = set()
for line in os.environ.get("HAVE", "").splitlines():
    m = re.match(r"\[r\d\] (\d+) (.+)$", line)
    if m and sizes.get(m.group(2)) == int(m.group(1)):
        present.add(m.group(2))
bins = [[0, []] for _ in range(4)]
for f in sorted((f for f in wanted if f not in present), key=lambda f: -sizes[f]):   # largest to the lightest rank
    b = min(bins, key=lambda b: b[0]); b[0] += sizes[f]; b[1].append(f)
for total, files in bins:
    print(f"{total / 2**30:.0f} {len(present)} " + " ".join(files))
PY
)
  (( ${#part[@]} == 4 )) || die "could not partition $CHECKSUMS"
  local r files
  log "download $MODEL_REPO@$MODEL_REVISION in four parts, one per rank uplink ($(cut -d' ' -f2 <<<"${part[0]}") files already present)"
  for r in 0 1 2 3; do
    files=$(cut -d' ' -f3- <<<"${part[$r]}")
    log "  rank $r: ${part[$r]%% *} GiB, $(wc -w <<<"$files") files"
    [[ -n $files ]] || continue
    remote "$r" 'mkdir -p "$MODEL_DIR"
      HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR" $FILES >/dev/null
      echo "rank $RANK: part downloaded"' MODEL_DIR="$MODEL_DIR" HF_BIN="$HF_BIN" REPO="$MODEL_REPO" REV="$MODEL_REVISION" \
      FILES="$files" RANK="$r" & pids+=($!)
  done
  wait_all "a rank's download failed"
  log "gather on rank 0: 1 -> 0 and 3 -> 0 in parallel, 2 -> 1 -> 0"
  ring_copy 1 0 & pids+=($!); ring_copy 3 0 & pids+=($!); ring_copy 2 1 & pids+=($!); wait_all "gather failed"
  ring_copy 1 0
  log "fan out: 0 -> 1 and 0 -> 3 in parallel, then 1 -> 2"
  ring_copy 0 1 & pids+=($!); ring_copy 0 3 & pids+=($!); wait_all "fan-out failed"
  ring_copy 1 2
  cmd_verify || die "checksum verification failed"
  log "fetch: checkpoint verified on all four ranks"
}
# wait_all MSG: wait on the jobs recorded in pids (`job & pids+=($!)`), die with MSG if any failed. Explicit PIDs,
# not `jobs -p`: a job that already exited is gone from the job table, and its failure would go unseen.
pids=()
wait_all() { local p rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; pids=(); (( rc == 0 )) || die "$1"; }
cable_addr() {  # cable_addr FROM TO: TO's address on the cable it shares with FROM (adjacent ranks only)
  local from=$1 to=$2 net
  if (( to == (from + 1) % 4 )); then net=${LINK[$from]}
  elif (( from == (to + 1) % 4 )); then net=${LINK[$to]}
  else die "rank $from and rank $to share no cable"; fi
  awk -v net="$net" '$1 == net {print $4; exit}' <<<"${SCAN[$to]}"
}
ring_copy() {   # ring_copy FROM TO: rsync FROM's checkpoint tree to TO across their cable (skips files TO already has)
  local from=$1 to=$2 ip; ip=$(cable_addr "$from" "$to")
  [[ -n $ip ]] || die "no cable address from rank $from to rank $to"
  log "rank $from -> rank $to over $ip"
  remote "$from" '
    ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER@$IP" mkdir -p "$MODEL_DIR"
    rsync -a --whole-file --exclude ".*" "$MODEL_DIR/" "$USER@$IP:$MODEL_DIR/"' USER="$SSH_USER" IP="$ip" MODEL_DIR="$MODEL_DIR"
}
verify_rank() {   # verify_rank RANK: sha256 of every pinned file on that rank, 12 files at a time
  local r=$1
  remote "$r" '
    cd "$MODEL_DIR"
    printf "%s\n" "$MANIFEST" > /tmp/dsv41-1017.sha256
    if awk "{print \$2}" /tmp/dsv41-1017.sha256 | xargs -P 12 -n 1 sha256sum 2>/dev/null | sort -k2 | diff -q - <(sort -k2 /tmp/dsv41-1017.sha256) >/dev/null
    then echo "rank $RANK: $(wc -l < /tmp/dsv41-1017.sha256) files verified"
    else echo "rank $RANK: sha256 mismatch or missing file under $MODEL_DIR" >&2; exit 1; fi' \
    MODEL_DIR="$MODEL_DIR" RANK="$r" MANIFEST="$(<"$ROOT/$CHECKSUMS")"
}
cmd_verify() { local r rc=0; for r in 0 1 2 3; do verify_rank "$r" & done; for r in 0 1 2 3; do wait -n || rc=1; done; return $rc; }

cmd_build() {
  log "build $IMAGE on rank 0"
  remote 0 'mkdir -p "$BUILD_DIR"' BUILD_DIR="$BUILD_DIR"
  rsync -a --delete -e "ssh ${SSH_OPTS[*]}" "$ROOT/docker" "$SSH_USER@$MASTER:$BUILD_DIR/"
  remote 0 'cd "$BUILD_DIR" && docker build -f docker/Dockerfile -t "$IMAGE" . 2>&1 | tail -25' BUILD_DIR="$BUILD_DIR" IMAGE="$IMAGE"
  # spark-ds41f's post-build GPU import smoke (B12X, TileLang, sparknet); it needs ~24 GiB free, so it is skipped
  # while another stack holds rank 0's memory.
  remote 0 'if [ "$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)" -lt 24 ]; then
      echo "image smoke skipped: under 24 GiB available on rank 0"
    else
      docker run --rm --gpus all --memory=16g --memory-swap=16g --ulimit=core=0 --entrypoint /usr/bin/python3 \
        "$IMAGE" /opt/spark-ds41f/smoke.py 2>&1 | tail -5
    fi' IMAGE="$IMAGE" || die "image smoke failed on rank 0"
  detect_ring || die "ring detection failed (./dsv41.sh doctor)"
  ship_image 0 1 & pids+=($!); ship_image 0 3 & pids+=($!); wait_all "image ship from rank 0 failed"
  ship_image 1 2
  every 'docker image inspect "$IMAGE" --format "{{.Id}}"' IMAGE="$IMAGE"
}
ship_image() {   # ship_image FROM TO: stream the image across the cable between adjacent ranks
  local from=$1 to=$2 ip; ip=$(cable_addr "$from" "$to")
  log "image rank $from -> rank $to over $ip"
  remote "$from" 'docker save "$IMAGE" | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER@$IP" docker load' \
    IMAGE="$IMAGE" USER="$SSH_USER" IP="$ip"
}

drop_caches() { every 'sync; sudo -n sysctl -q -w vm.drop_caches=$LEVEL 2>/dev/null || true' LEVEL="${1:-3}"; }

cmd_stop() {
  every 'sudo -n systemctl stop "$UNIT" >/dev/null 2>&1 || true
         if docker ps -a --format "{{.Names}}" | grep -qx "$C"; then
           docker stop -t "$G" "$C" >/dev/null 2>&1 || true
           docker logs "$C" > "$HOME/$C.prev.log" 2>&1 || true
           docker rm -f "$C" >/dev/null; echo stopped
         else echo "not running"; fi' C="$CONTAINER" UNIT="$GUARD_UNIT" G="$STOP_GRACE"
}

# ── memory guard ───────────────────────────────────────────────────────────────────────────────
# Fail-closed, as in spark-ds41f: before each rank's docker run a startup guard (STARTUP_MEMGUARD_GIB, sampled every
# STARTUP_MEMGUARD_INTERVAL s, waiting up to STARTUP_MEMGUARD_WAIT s for the container) is armed as a systemd unit;
# once the API answers, every node switches to the steady guard (MEMGUARD_GIB every MEMGUARD_INTERVAL s). A guard
# SIGKILLs the rank when MemAvailable drops below its threshold or the container can no longer be inspected, and
# start aborts if any guard is gone before the API is up.
guard_check() {
  [[ $CONTAINER =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "CONTAINER must be a container name (it also names the memory guard unit)"
  [[ $STARTUP_MEMGUARD_GIB =~ ^[1-9][0-9]*$ && $MEMGUARD_GIB =~ ^[1-9][0-9]*$ ]] || die "STARTUP_MEMGUARD_GIB and MEMGUARD_GIB must be positive integers"
  (( STARTUP_MEMGUARD_GIB > MEMGUARD_GIB )) || die "STARTUP_MEMGUARD_GIB must exceed MEMGUARD_GIB"
  [[ $STARTUP_MEMGUARD_WAIT =~ ^[1-9][0-9]*$ ]] || die "STARTUP_MEMGUARD_WAIT must be a positive integer"
  local i; for i in "$STARTUP_MEMGUARD_INTERVAL" "$MEMGUARD_INTERVAL"; do
    [[ $i =~ ^[0-9]+([.][0-9]+)?$ && ! $i =~ ^0+([.]0+)?$ ]] || die "memory guard intervals must be positive numbers (got '$i')"
  done
}
# guard_cmd startup|steady: the node command that arms a guard (memguard-start.sh CONTAINER PHASE GIB INTERVAL
# WAIT MODE); $HOME expands on the node. The startup guard is armed before its container exists (prelaunch).
guard_cmd() {
  guard_check
  case $1 in
    startup) printf '"$HOME/.local/lib/%s-memguard/scripts/memguard-start.sh" %s startup %s %s %s prelaunch\n' \
               "$CONTAINER" "$CONTAINER" "$STARTUP_MEMGUARD_GIB" "$STARTUP_MEMGUARD_INTERVAL" "$STARTUP_MEMGUARD_WAIT" ;;
    steady) printf '"$HOME/.local/lib/%s-memguard/scripts/memguard-start.sh" %s steady %s %s 0 running\n' \
               "$CONTAINER" "$CONTAINER" "$MEMGUARD_GIB" "$MEMGUARD_INTERVAL" ;;
  esac
}
# install_guards: copy spark-ds41f's scripts/ out of the image on every node (docker create + cp, nothing runs) and
# refuse it unless memguard.sh and memguard-start.sh are byte-identical to the pinned commit. A node whose installed
# copy already matches the pinned sums keeps it (the content is what is pinned, not the image it came from).
install_guards() {
  every '
    sudo -n true 2>/dev/null || { echo "no passwordless sudo: the memory guard cannot run"; exit 1; }
    G=$HOME/.local/lib/$C-memguard
    if [ -d "$G/scripts" ] && (cd "$G/scripts" && printf "%s\n" "$SUMS" | sha256sum --check --status); then
      echo "memory guard: spark-ds41f@$REV (installed)"; exit 0
    fi
    rm -rf "$G.new"; mkdir -p "$G.new"
    id=$(docker create "$IMAGE")
    if docker cp "$id:$SRC" "$G.new/scripts" >/dev/null 2>&1; then ok=1; else ok=0; fi
    docker rm "$id" >/dev/null
    [ "$ok" = 1 ] || { echo "image $IMAGE has no $SRC"; exit 1; }
    (cd "$G.new/scripts" && printf "%s\n" "$SUMS" | sha256sum --check --status) \
      || { echo "memguard scripts in $IMAGE differ from spark-ds41f@$REV"; exit 1; }
    rm -rf "$G"; mv "$G.new" "$G"; echo "memory guard: spark-ds41f@$REV"' \
    C="$CONTAINER" IMAGE="$IMAGE" SRC="$GUARD_SRC" SUMS="$GUARD_SUMS" REV="$GUARD_REV"
}
steady_guards() { local g; g=$(guard_cmd steady) || exit 1; every "$g"; }

cmd_start() {
  ring_config || die "ring detection failed (./dsv41.sh doctor)"
  local r running allowed out guard
  local -A cmd=()
  # Every rank's command first: a configuration error must not leave some ranks started.
  guard=$(guard_cmd startup) || exit 1
  for r in 0 1 2 3; do cmd[$r]=$( { docker_args "$r"; serve_args "$r"; } | quoted ) || exit 1; done
  # Pinned RING_* values are what print-serve shows; start re-scans the ring and refuses if a pin is stale (a
  # link flap moves GID indices).
  [[ -z ${RING_PREFIX:-} ]] || ring_verify || die "RING_* pins in .env differ from the live ring (./dsv41.sh doctor)"
  allowed=$(tr ' ' '\n' <<<"$CONTAINER ${ALLOWED_CONTAINERS:-}" | grep . | paste -sd'|' -)
  running=$(every 'docker ps --format "{{.Names}}" | grep -vxE "$A" || true' A="$allowed")
  [[ -z $(sed 's/^\[r[0-9]\] //' <<<"$running" | grep .) ]] || die "other containers are running (./dsv41.sh stop-other, or ALLOWED_CONTAINERS):"$'\n'"$running"
  out=$(cmd_stop 2>&1) || die "stop before start failed:"$'\n'"$out"
  out=$(drop_caches 3 2>&1) || die "page cache drop failed:"$'\n'"$out"
  if [[ $GPU_CLOCKS =~ ^[1-9][0-9]*$ ]]; then
    out=$(every 'sudo -n nvidia-smi -lgc "$F,$F" >/dev/null' F="$GPU_CLOCKS" 2>&1) || die "GPU clock lock failed:"$'\n'"$out"
  elif [[ $GPU_CLOCKS != keep ]]; then die "GPU_CLOCKS must be a clock in MHz or keep (got '$GPU_CLOCKS')"; fi
  out=$(every 'mkdir -p "$D"' D="$CACHE_DIR" 2>&1) || die "cannot create $CACHE_DIR:"$'\n'"$out"
  out=$(install_guards 2>&1) || die "memory guard install failed:"$'\n'"$out"
  log "prepared: stopped, caches dropped, clocks, guards"
  # Workers start together, the head last (spark-ds41f's deployment order: the head's rendezvous finds them up).
  launch_rank() {
    remote "$1" "$guard"$'\n'"docker ${cmd[$1]} >/dev/null || { sudo -n systemctl stop \"\$UNIT\" >/dev/null 2>&1 || true; exit 1; }" \
      UNIT="$GUARD_UNIT" VLLM_API_KEY="$API_KEY"
  }
  local -a lp=(); local failed=""
  for r in 3 2 1; do launch_rank "$r" & lp+=($!); done
  for r in 3 2 1; do wait "${lp[$((3 - r))]}" || failed+=" $r"; done
  [[ -z $failed ]] || { cmd_stop >/dev/null 2>&1; die "guarded docker run failed on rank(s)$failed"; }
  log "ranks 3 2 1 started under their startup memory guards"
  launch_rank 0 || { cmd_stop >/dev/null 2>&1; die "guarded docker run failed on rank 0"; }
  log "rank 0 started on ${NODE[0]} under its startup memory guard"
  log "waiting for $URL (up to ${WAIT_TIMEOUT}s; first boot prepares the TileLang kernels into $CACHE_DIR)"
  local t0=$SECONDS dead last=$SECONDS
  while (( SECONDS - t0 < WAIT_TIMEOUT )); do
    if curl -sf -m 5 "$URL/health" >/dev/null; then
      log "up after $((SECONDS - t0))s: $URL/v1"
      out=$(steady_guards 2>&1) || { echo "$out"; cmd_stop >/dev/null 2>&1 || true; die "could not switch to the steady memory guard"; }
      echo "$out"
      cmd_smoke; return
    fi
    # Health is polled every second; the liveness check and the page-cache drop keep their 10 s cadence.
    if (( SECONDS - last >= 10 )); then
      last=$SECONDS
      dead=$(every 'docker ps --format "{{.Names}}" | grep -qx "$C" || echo DEAD; systemctl is-active --quiet "$UNIT" || echo UNGUARDED' \
        C="$CONTAINER" UNIT="$GUARD_UNIT" | grep -E 'DEAD|UNGUARDED' || true)
      if [[ -n $dead ]]; then
        echo "$dead"
        for r in 0 1 2 3; do echo "── rank $r"; remote "$r" 'docker logs --tail 40 "$C" 2>&1 || tail -40 "$HOME/$C.prev.log"; journalctl -t spark-memguard --since "-15min" --no-pager 2>/dev/null | tail -5 || true' C="$CONTAINER" || true; done
        cmd_stop >/dev/null 2>&1 || true; die "a rank exited or lost its memory guard during startup"
      fi
      drop_caches 1 >/dev/null || log "WARN: drop_caches failed on a rank"
    fi
    sleep 1
  done
  die "API not up after ${WAIT_TIMEOUT}s (./dsv41.sh logs)"
}

# Authorization header via a process-substitution file (curl -H @file), so the key never appears in curl's argv.
auth_header() { [[ -z $API_KEY ]] || printf 'Authorization: Bearer %s\n' "$API_KEY"; }
cmd_status() {
  every 'docker ps -a --filter "name=^$C$" --format "{{.Names}} {{.Status}}"
         echo "memguard=$(systemctl is-active "$UNIT" 2>/dev/null || true) mem_avail_gb=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)"' \
    C="$CONTAINER" UNIT="$GUARD_UNIT"
  curl -sf -m 5 -H @<(auth_header) "$URL/v1/models" && echo || echo "API not answering at $URL"
}

cmd_logs() {
  local r=${1:-0} n=${2:-200}
  remote "$r" 'docker logs --tail "$N" "$C" 2>&1' N="$n" C="$CONTAINER"
}

cmd_smoke() {
  local out
  out=$(curl -sf -m 120 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -H @<(auth_header) \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Reply with the integer only.\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}") \
    || die "smoke: request failed"
  grep -q '391' <<<"$out" && log "smoke: OK (17*23=391)" || die "smoke: wrong answer: $out"
}

# The python tools read the key from env API_KEY, so it stays out of argv.
cmd_api_test() { API_KEY=$API_KEY python3 "$ROOT/tests/api_conformance.py" --base-url "$URL/v1" --model "$SERVED_NAME" "$@"; }
cmd_parity() { API_KEY=$API_KEY python3 "$ROOT/tests/test_encoding_parity.py" --base-url "$URL" --revision "$MODEL_REVISION" --model "$SERVED_NAME"; }
cmd_bench() { API_KEY=$API_KEY python3 "$ROOT/bench/bench.py" --base-url "$URL" --model "$SERVED_NAME" "$@"; }

cmd_stop_other() { [[ -n $STOP_OTHER ]] || die "STOP_OTHER is not set (.env.example)"; log "STOP_OTHER"; bash -c "$STOP_OTHER"; }
cmd_restore_other() {
  [[ -n $RESTORE_OTHER ]] || die "RESTORE_OTHER is not set (.env.example)"
  cmd_stop; log "RESTORE_OTHER"; bash -c "$RESTORE_OTHER"
}

cmd_print_serve() {
  local r=${1:?rank} cmd guard
  ring_config || die "ring detection failed"
  # Captured first: a die inside docker_args/serve_args must not leave a partial command on stdout.
  guard=$(guard_cmd startup) || exit 1
  cmd=$( { docker_args "$r"; serve_args "$r"; } | quoted ) || exit 1
  echo "$guard"
  echo "docker $cmd"
}

cmd_help() { sed -n '3,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

case ${1:-help} in
  doctor) cmd_doctor ;;
  fetch) cmd_fetch ;;
  verify) cmd_verify ;;
  build) cmd_build ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "$@" ;;
  smoke) cmd_smoke ;;
  api-test) shift; cmd_api_test "$@" ;;
  parity) cmd_parity ;;
  bench) shift; cmd_bench "$@" ;;
  stop-other) cmd_stop_other ;;
  restore-other) cmd_restore_other ;;
  print-serve) shift; cmd_print_serve "$@" ;;
  help|-h|--help) cmd_help ;;
  *) echo "unknown command: $1 (./dsv41.sh help)" >&2; exit 2 ;;
esac
