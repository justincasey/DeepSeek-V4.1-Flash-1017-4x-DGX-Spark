#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# dsv41.sh: DeepSeek-V4.1-Flash-1017 (native MXFP4/FP8) on four DGX Sparks, TP4 over the switchless ring, vLLM.
# Run from any control machine with key-based ssh to the four nodes. Nothing runs on the nodes outside docker,
# except the Hugging Face client for `fetch`.
#
# Usage: ./dsv41.sh <command> [args]
#   doctor               read-only: ssh, docker, image, GPU idle, memory, NVMe, checkpoint, NCCL bundle, sysctls, ring cables
#   fetch                download the pinned checkpoint on rank 0, fan it out over the ring, sha256-verify every node
#   verify               sha256-verify the checkpoint on every node (no download)
#   build                build the serving image on rank 0, ship it over the ring, verify the image ID everywhere
#   start                start the four ranks (workers first), wait for the API, run smoke
#   stop | restart | status | smoke
#   logs [rank] [lines]  tail one rank's engine log (default rank 0, 200 lines)
#   api-test [args]      full API conformance suite against the live endpoint (tests/api_conformance.py)
#   parity               server prompt rendering == publisher encoding.py (tests/test_encoding_parity.py)
#   bench STEP [args]    decode | prefill | needle | spec | pin | check | all (bench/bench.py); results in bench/runs/
#   stop-other           run STOP_OTHER: take down the stack that normally owns these GPUs
#   restore-other        stop this stack, then run RESTORE_OTHER
#   print-serve RANK     print the exact docker run + vllm serve command for one rank (no side effects)
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

: "${NODES:=192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18}"   # management addresses, ring cable order, rank 0 first
: "${SSH_USER:=spark}"
: "${MODELS_DIR:=/home/spark/models}"
: "${MODEL_REPO:=deepseek-ai/DeepSeek-V4.1-Flash-1017}"
: "${MODEL_REVISION:=1bae0e1088993b9b82de08f59de5be5a34e42170}"
: "${MODEL_NAME:=DeepSeek-V4.1-Flash-1017}"
: "${CHECKSUMS:=config/DeepSeek-V4.1-Flash-1017.sha256}"
: "${HF_BIN:=/home/spark/venvs/hf/bin/hf}"
: "${IMAGE:=dsv41-1017-spark:local}"
: "${BUILD_DIR:=/home/spark/build/dsv41-1017}"
: "${SWITCHLESS_NCCL:=/home/spark/nccl-switchless-v0.0.1}"
: "${CACHE_DIR:=/home/spark/.cache/dsv41-1017}"
: "${CONTAINER:=dsv41-1017}"
: "${SERVED_NAME:=deepseek-v4.1-flash}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${API_KEY:=}"
: "${MASTER_PORT:=29511}"
: "${MAX_MODEL_LEN:=1048576}"
: "${MAX_NUM_SEQS:=4}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${KV_CACHE_MEMORY_BYTES=13958643712}"   # 13 GiB per rank: 2.67M KV tokens at 1M (measured); set empty to profile
: "${GPU_MEMORY_UTILIZATION:=0.80}"
: "${MOE_MODE:=ep}"                     # ep: TP4 + EP4 (96 experts/rank) | tp: MoE tensor-parallel (EP1)
: "${MOE_BACKEND:=marlin}"              # MXFP4 expert kernel: marlin | deep_gemm | humming | auto
: "${DSPARK:=1}"
: "${NUM_SPECULATIVE_TOKENS:=3}"
: "${DRAFT_SAMPLE_METHOD:=probabilistic}"   # DSpark drafter sampling: probabilistic | greedy
: "${EAGER:=0}"
: "${COMPILATION_CONFIG:=}"
: "${TEXT_ONLY:=0}"
: "${PREFIX_CACHING:=1}"
: "${MXFP8_BACKEND:=cutlass}"          # FlashInfer backend for the FP8 dense layers: cutlass | b12x | auto
: "${NCCL_TRIM:=1}"                     # smaller NCCL buffers: more free unified memory per node
: "${TOPK_SORT:=1}"                     # fixed sparse-indexer top-k order: reproducible outputs under CUDA graphs
: "${EXTRA_VLLM_ARGS:=}"
: "${EXTRA_DOCKER_ARGS:=}"
: "${WAIT_TIMEOUT:=2400}"
: "${STOP_OTHER:=}"
: "${RESTORE_OTHER:=}"
: "${ALLOWED_CONTAINERS:=}"            # other containers that may keep running alongside the ranks (space separated)

read -r -a NODE <<<"$NODES"
(( ${#NODE[@]} == 4 )) || { echo "NODES must list exactly four management addresses (got ${#NODE[@]})" >&2; exit 1; }
MASTER=${NODE[0]}
MODEL_DIR=$MODELS_DIR/$MODEL_NAME
URL="http://$MASTER:$PORT"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }
is_true() { case ${1:-} in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }

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
# Per node: every up RoCE netdev with exactly one IPv4 address, as "subnet/prefix hca netdev addr", plus the
# management netdev that routes to rank 0. Devices are matched by subnet, never by name.
SCAN_SCRIPT='
for d in /sys/class/net/*; do
  n=${d##*/}; [ -d "$d/device/infiniband" ] || continue
  [ "$(cat "$d/operstate" 2>/dev/null)" = up ] || continue
  hca=$(ls "$d/device/infiniband" | head -1)
  mapfile -t a < <(ip -4 -o addr show dev "$n" | awk "{print \$4}")
  [ ${#a[@]} -eq 1 ] || { echo "@bad $n ${#a[@]}"; continue; }
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "${a[0]}")
  echo "$net $hca $n ${a[0]%/*}"
done
mg=$(ip -4 -o addr show | awk -v ip="$MASTER" "index(\$4, ip \"/\") == 1 {print \$2; exit}")
[ -n "$mg" ] || mg=$(ip -4 -o route get "$MASTER" | sed -n "s/.* dev \([^ ]*\).*/\1/p")
echo "@mgmt $mg"'
declare -a SCAN=() MGMT=() RHCA=() LINK=() PEER_IP=()
PREFIX=
shared_net() { comm -12 <(awk '$1 !~ /^@/ {print $1}' <<<"${SCAN[$1]}" | sort) <(awk '$1 !~ /^@/ {print $1}' <<<"${SCAN[$2]}" | sort) | head -1; }
detect_ring() {
  local r n p net bad=0
  for r in 0 1 2 3; do
    SCAN[$r]=$(remote "$r" "$SCAN_SCRIPT" MASTER="$MASTER") || die "ssh to ${NODE[$r]} failed"
    MGMT[$r]=$(awk '$1 == "@mgmt" {print $2}' <<<"${SCAN[$r]}")
    grep -q '^@bad' <<<"${SCAN[$r]}" && { echo "rank $r: RoCE port with !=1 IPv4 address: $(grep '^@bad' <<<"${SCAN[$r]}" | tr '\n' ' ')"; bad=1; }
  done
  for r in 0 1 2 3; do n=$(( (r + 1) % 4 ))
    LINK[$r]=$(shared_net "$r" "$n")
    [[ -n ${LINK[$r]} ]] || { echo "no cable between rank $r (${NODE[$r]}) and rank $n (${NODE[$n]}): NODES must be in cable order"; bad=1; }
  done
  for r in 0 1; do
    [[ -z $(shared_net "$r" $((r + 2))) ]] || { echo "rank $r and rank $((r + 2)) share a subnet: this is not a switchless ring"; bad=1; }
  done
  (( bad == 0 )) || return 1
  PREFIX=${LINK[0]#*/}
  for r in 0 1 2 3; do n=$(( (r + 1) % 4 )); p=$(( (r + 3) % 4 ))
    [[ ${LINK[$r]#*/} == "$PREFIX" ]] || { echo "cable subnets differ in prefix length"; bad=1; }
    RHCA[$r]=$(for net in "${LINK[$p]}" "${LINK[$r]}"; do awk -v net="$net" '$1 == net {print $2; exit}' <<<"${SCAN[$r]}"; done | paste -sd, -)
    PEER_IP[$r]=$(awk -v net="${LINK[$r]}" '$1 == net {print $4; exit}' <<<"${SCAN[$n]}")   # rank n's address on cable r->n
    [[ -n ${MGMT[$r]} && ${MGMT[$r]} != lo ]] || { echo "rank $r: no management route to $MASTER"; bad=1; }
  done
  return $bad
}
# ring_config: NCCL rails and bootstrap interface for every rank, from RING_* pins in .env when RING_PREFIX is set
# (no ssh), else from detect_ring. Copies over the cables (fetch, build) always scan.
ring_config() {
  if [[ -n ${RING_PREFIX:-} ]]; then
    local r h m
    PREFIX=$RING_PREFIX
    for r in 0 1 2 3; do
      h=RING_HCA_$r m=RING_MGMT_$r
      [[ -n ${!h:-} && -n ${!m:-} ]] || die "RING_PREFIX is set, so $h and $m must be set too"
      RHCA[$r]=${!h} MGMT[$r]=${!m}
    done
    return 0
  fi
  detect_ring
}

# ── serve command ───────────────────────────────────────────────────────────────────────────────
serve_args() {   # serve_args RANK: the `vllm` argv after the image name (the image entrypoint is set to vllm)
  local r=$1 spec
  printf '%s\n' serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT" \
    --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --tensor-parallel-size 4 --nnodes 4 --node-rank "$r" --master-addr "$MASTER" --master-port "$MASTER_PORT" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --engram-config '{"cpu_offload":false,"disk_backed":true}'
  (( r > 0 )) && echo --headless
  # The utilization fraction also gates vLLM's startup free-memory check, so it is always passed; a fixed KV
  # budget replaces profiling-based KV sizing on top of it.
  printf '%s\n' --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  [[ -n $KV_CACHE_MEMORY_BYTES ]] && printf '%s\n' --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  case $MOE_MODE in
    # The ring has no cable between ranks 0-2 and 1-3: expert dispatch must use collectives, never point-to-point.
    ep) printf '%s\n' --enable-expert-parallel --enable-ep-weight-filter --all2all-backend allgather_reducescatter ;;
    tp) ;;
    *) die "MOE_MODE must be ep or tp" ;;
  esac
  # MXFP4 expert kernel. Marlin (W4A16: MXFP4 weights, BF16 activations) measured 15-40% faster decode than vLLM's
  # auto pick on sm_121 (DeepGEMM MXFP4 with FP8 activations); the DSpark drafter inherits the same backend.
  case $MOE_BACKEND in
    auto) ;;
    marlin|deep_gemm|humming) printf '%s\n' --moe-backend "$MOE_BACKEND" ;;
    *) die "MOE_BACKEND must be marlin, deep_gemm, humming or auto" ;;
  esac
  if is_true "$PREFIX_CACHING"; then echo --enable-prefix-caching; else echo --no-enable-prefix-caching; fi
  if is_true "$TEXT_ONLY"; then echo --language-model-only; else printf '%s\n' --mm-encoder-tp-mode data; fi
  if is_true "$EAGER"; then
    echo --enforce-eager
  elif [[ -n $COMPILATION_CONFIG ]]; then
    printf '%s\n' --compilation-config "$COMPILATION_CONFIG"
  else
    # Decode-only CUDA graphs (Engram lookups are hoisted out of the graph) at every batch shape decode can take:
    # each sequence verifies k+1 rows with DSpark, one row without.
    local rows=1 i sizes=""
    is_true "$DSPARK" && rows=$((NUM_SPECULATIVE_TOKENS + 1))
    for (( i = 1; i <= MAX_NUM_SEQS; i++ )); do sizes+="${sizes:+,}$((i * rows))"; done
    printf '%s\n' --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$sizes]}"
  fi
  if is_true "$DSPARK"; then
    case $DRAFT_SAMPLE_METHOD in probabilistic|greedy) ;; *) die "DRAFT_SAMPLE_METHOD must be probabilistic or greedy" ;; esac
    spec=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s","rejection_sample_method":"block"}' \
      "$NUM_SPECULATIVE_TOKENS" "$DRAFT_SAMPLE_METHOD")
    printf '%s\n' --speculative-config "$spec"
  fi
  if [[ -n $EXTRA_VLLM_ARGS ]]; then local -a extra; read -r -a extra <<<"$EXTRA_VLLM_ARGS"; printf '%s\n' "${extra[@]}"; fi
  return 0
}
docker_args() {  # docker_args RANK: docker run argv (switchless NCCL contract + disk Engram + caches)
  local r=$1 lib=/opt/switchless-nccl/libnccl.so.2
  case $MXFP8_BACKEND in cutlass|b12x|auto) ;; *) die "MXFP8_BACKEND must be cutlass, b12x or auto (got '$MXFP8_BACKEND')" ;; esac
  printf '%s\n' run -d --name "$CONTAINER" --entrypoint vllm --gpus all --network host --ipc host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit core=0 --ulimit stack=67108864 \
    --shm-size 16g \
    -v "$MODELS_DIR:$MODELS_DIR:ro" \
    -v "$SWITCHLESS_NCCL:/opt/switchless-nccl:ro" \
    -v "$CACHE_DIR/root-cache:/root/.cache" \
    -v "$CACHE_DIR/triton:/root/.triton" \
    -e LD_PRELOAD=$lib -e VLLM_NCCL_SO_PATH=$lib -e TORCH_USE_RTLD_GLOBAL=1 \
    -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_ALGO=Ring -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA="${RHCA[$r]}" -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN="$PREFIX" -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=1 \
    -e NCCL_SOCKET_IFNAME="${MGMT[$r]}" -e GLOO_SOCKET_IFNAME="${MGMT[$r]}" -e VLLM_HOST_IP="${NODE[$r]}" \
    -e VLLM_ENGRAM_DISK_BACKED=1 -e VLLM_ENGRAM_MODEL_DIR="$MODEL_DIR" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    -e DSV41_MXFP8_BACKEND="$MXFP8_BACKEND" \
    -e DSV41_TOPK_SORT="$(is_true "$TOPK_SORT" && echo 1 || echo 0)" \
    -e HF_HUB_OFFLINE=1
  # The key is named, never valued, here: docker copies it from the remote shell's environment (cmd_start sends
  # it over ssh stdin), so it stays out of every process argv and print-serve. vLLM reads VLLM_API_KEY.
  if (( r == 0 )) && [[ -n $API_KEY ]]; then printf '%s\n' -e VLLM_API_KEY; fi
  # NCCL connection buffers live in pinned host RAM, which on GB10 is GPU memory. Smaller buffers, no LL128 and
  # 8 channels gave 3-4 GiB more MemAvailable per node at 1M context with no measured decode or prefill cost
  # (README Benchmarks). Settings from MiaAI-Lab's DeepSeek-v4.1-Flash-DGX-Sparks measurements.
  if is_true "$NCCL_TRIM"; then
    printf '%s\n' -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e 'NCCL_PROTO=^LL128' -e NCCL_MAX_NCHANNELS=8
  fi
  if [[ -n $EXTRA_DOCKER_ARGS ]]; then local -a extra; read -r -a extra <<<"$EXTRA_DOCKER_ARGS"; printf '%s\n' "${extra[@]}"; fi
  echo "$IMAGE"
}
quoted() { local a out=""; while IFS= read -r a; do out+=" $(printf '%q' "$a")"; done; echo "${out# }"; }

# ── commands ───────────────────────────────────────────────────────────────────────────────────
cmd_doctor() {
  local bad=0 out r
  log "ring"
  if detect_ring; then
    for r in 0 1 2 3; do
      echo "  rank $r ${NODE[$r]}: NCCL ${RHCA[$r]}, bootstrap ${MGMT[$r]}, next rank at ${PEER_IP[$r]} (${LINK[$r]})"
      remote "$r" 'ping -c 2 -W 1 "$PEER" >/dev/null' PEER="${PEER_IP[$r]}" || { echo "  FAIL rank $r cannot ping ${PEER_IP[$r]}"; bad=1; }
    done
  else bad=1; fi
  log "nodes"
  out=$(every '
    command -v docker >/dev/null && echo docker=ok || echo docker=missing
    img=$(docker image inspect "$IMAGE" --format "{{.Id}}" 2>/dev/null | grep . | tail -1 || true); echo "image=${img:-none}"
    echo "gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
    echo "mem_avail_gb=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)"
    echo "nvme_free_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc 0-9)"
    echo "compaction=$(cat /proc/sys/vm/compaction_proactiveness) swappiness=$(cat /proc/sys/vm/swappiness)"
    sudo -n true 2>/dev/null && echo sudo=ok || echo sudo=no
    if [ -f "$SWITCHLESS_NCCL/libnccl.so.2" ] && (cd "$SWITCHLESS_NCCL" && sha256sum --check --status --ignore-missing SHA256SUMS); then echo nccl=ok; else echo nccl=bad; fi
    if [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
      need=$(python3 -c "import json,sys; print(len(set(json.load(open(sys.argv[1]))[\"weight_map\"].values())))" "$MODEL_DIR/model.safetensors.index.json")
      have=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l); echo "shards=$have/$need"
    else echo shards=none; fi
    docker ps --format "{{.Names}}" | grep -qx "$CONTAINER" && echo running=yes || echo running=no
  ' IMAGE="$IMAGE" MODELS_DIR="$MODELS_DIR" MODEL_DIR="$MODEL_DIR" SWITCHLESS_NCCL="$SWITCHLESS_NCCL" CONTAINER="$CONTAINER")
  echo "$out" | sed 's/^/  /'
  local img0; img0=$(awk '/^\[r0\] image=/{sub(/.*image=/,""); print}' <<<"$out")
  for r in 0 1 2 3; do
    local o; o=$(grep "^\[r$r\]" <<<"$out")
    grep -q docker=ok <<<"$o" || { echo "FAIL rank $r: docker missing"; bad=1; }
    grep -q "image=$img0" <<<"$o" && [[ $img0 != none ]] || { echo "FAIL rank $r: image $IMAGE missing or differs from rank 0 (./dsv41.sh build)"; bad=1; }
    grep -q nccl=ok <<<"$o" || { echo "FAIL rank $r: switchless NCCL at $SWITCHLESS_NCCL missing or fails its SHA256SUMS"; bad=1; }
    grep -qE 'shards=([0-9]+)/\1$' <<<"$o" || { echo "FAIL rank $r: checkpoint incomplete at $MODEL_DIR (./dsv41.sh fetch)"; bad=1; }
    grep -q compaction=0 <<<"$o" || echo "WARN rank $r: vm.compaction_proactiveness != 0 (decode stalls)"
    grep -q sudo=ok <<<"$o" || echo "WARN rank $r: no passwordless sudo: page cache cannot be dropped during loads"
    if grep -q running=no <<<"$o"; then
      grep -q gpu_procs=0 <<<"$o" || { echo "FAIL rank $r: other GPU processes are running (./dsv41.sh stop-other)"; bad=1; }
      [[ $(grep -o 'mem_avail_gb=[0-9]*' <<<"$o" | cut -d= -f2) -ge 100 ]] || echo "WARN rank $r: under 100 GiB available"
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
  rsync -a --delete -e "ssh ${SSH_OPTS[*]}" "$ROOT/docker" "$ROOT/config" "$SSH_USER@$MASTER:$BUILD_DIR/"
  remote 0 'cd "$BUILD_DIR" && docker build -f docker/Dockerfile -t "$IMAGE" . 2>&1 | tail -25' BUILD_DIR="$BUILD_DIR" IMAGE="$IMAGE"
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
  every 'if docker ps -a --format "{{.Names}}" | grep -qx "$C"; then
           docker stop -t 60 "$C" >/dev/null 2>&1 || true
           docker logs "$C" > "$HOME/$C.prev.log" 2>&1 || true
           docker rm -f "$C" >/dev/null; echo stopped
         else echo "not running"; fi' C="$CONTAINER"
}

cmd_start() {
  ring_config || die "ring detection failed (./dsv41.sh doctor)"
  local r running allowed out
  allowed=$(tr ' ' '\n' <<<"$CONTAINER ${ALLOWED_CONTAINERS:-}" | grep . | paste -sd'|' -)
  running=$(every 'docker ps --format "{{.Names}}" | grep -vxE "$A" || true' A="$allowed")
  [[ -z $(sed 's/^\[r[0-9]\] //' <<<"$running" | grep .) ]] || die "other containers are running (./dsv41.sh stop-other, or ALLOWED_CONTAINERS):"$'\n'"$running"
  out=$(cmd_stop 2>&1) || die "stop before start failed:"$'\n'"$out"
  out=$(drop_caches 3 2>&1) || die "page cache drop failed:"$'\n'"$out"
  out=$(every 'mkdir -p "$D/root-cache" "$D/triton"' D="$CACHE_DIR" 2>&1) || die "cannot create $CACHE_DIR:"$'\n'"$out"
  for r in 3 2 1 0; do
    local run; run=$( { docker_args "$r"; serve_args "$r"; } | quoted )
    # The key rides the prelude (ssh stdin); docker_args names it for rank 0 only.
    remote "$r" "docker $run >/dev/null" VLLM_API_KEY="$API_KEY" || { cmd_stop >/dev/null 2>&1; die "docker run failed on rank $r"; }
    log "rank $r started on ${NODE[$r]}"
  done
  log "waiting for $URL (up to ${WAIT_TIMEOUT}s; page cache is dropped while the ranks load)"
  local t0=$SECONDS dead
  while (( SECONDS - t0 < WAIT_TIMEOUT )); do
    if curl -sf -m 5 "$URL/health" >/dev/null; then log "up after $((SECONDS - t0))s: $URL/v1"; cmd_smoke; return; fi
    dead=$(every 'docker ps --format "{{.Names}}" | grep -qx "$C" || echo DEAD' C="$CONTAINER" | grep DEAD || true)
    if [[ -n $dead ]]; then
      echo "$dead"
      for r in 0 1 2 3; do echo "── rank $r"; remote "$r" 'docker logs --tail 40 "$C" 2>&1 || tail -40 "$HOME/$C.prev.log"' C="$CONTAINER" || true; done
      cmd_stop >/dev/null 2>&1 || true; die "a rank exited during startup"
    fi
    drop_caches 1 >/dev/null || log "WARN: drop_caches failed on a rank"
    sleep 10
  done
  die "API not up after ${WAIT_TIMEOUT}s (./dsv41.sh logs)"
}

# Authorization header via a process-substitution file (curl -H @file), so the key never appears in curl's argv.
auth_header() { [[ -z $API_KEY ]] || printf 'Authorization: Bearer %s\n' "$API_KEY"; }
cmd_status() {
  every 'docker ps -a --filter "name=^$C$" --format "{{.Names}} {{.Status}}"; echo "mem_avail_gb=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)"' C="$CONTAINER"
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
  local r=${1:?rank} cmd
  ring_config || die "ring detection failed"
  # Captured first: a die inside docker_args/serve_args must not leave a partial command on stdout.
  cmd=$( { docker_args "$r"; serve_args "$r"; } | quoted ) || exit 1
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
