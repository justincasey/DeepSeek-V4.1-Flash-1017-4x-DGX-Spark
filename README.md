# DeepSeek-V4.1-Flash-1017 on four DGX Sparks

Serve [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) in its native
checkpoint (MXFP4 routed experts, FP8 dense) from four NVIDIA DGX Sparks joined in a switchless QSFP ring, behind
an OpenAI-compatible API, with a 1,048,576-token window and 16 concurrent sequences.

The engine is Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f) release **r6w**
(Apache-2.0): Local Inference Lab's vLLM fork with his patch series, the TileLang SM121 kernel family, DeepSeek's
TileKernels, sparknet one-shot RoCE collectives and a patched NCCL. `docker/Dockerfile` rebuilds his image from the
pinned upstreams (spark-ds41f at `2dc95329`), adds his MoE scale-codes overlay and his linear DSpark draft budget
(r6x's vLLM 0184), and patches of ours (effort aliases, weight snapshots for a quick boot, two upstream vLLM fixes
ported, two exact prefill savings, a persistent prefix KV cache); `dsv41.sh` serves his `tp4-ring` configuration,
tuned for this ring, through our controller, on the stock DGX OS 4 KiB-page kernel.

One script, run from a control machine, does everything:

```bash
cp .env.example .env        # set NODES (management addresses, ring cable order, rank 0 first)
./dsv41.sh doctor           # read-only checks on all four nodes; scans the ring and prints the RING_* pins
./dsv41.sh fetch            # pinned checkpoint on every node, sha256-verified
./dsv41.sh build            # serving image on rank 0, shipped over the ring
./dsv41.sh stop-other       # optional: free the GPUs from the stack that normally owns them
./dsv41.sh start            # memory guards armed, ranks 3..0 started; http://<rank0>:8000/v1
./dsv41.sh api-test         # full API conformance suite
./dsv41.sh parity           # server prompts == DeepSeek's encoder
```

## Headline numbers

Measured on this ring with the same harnesses, checkpoint `1bae0e1`: v1.0.0 (the previous engine,
[fallback](#fallback)) on 2026-10-17; r6v, tagged `v2.0.0`, on 2026-10-17/18 (image `sha256:875c7c9edc08…`); and
this repo's current defaults (r6w with the scale-codes overlay and this ring's tuning,
[what changed](#current-defaults-against-r6v)) on 2026-10-18, with the GPUs locked at 2,200 MHz (site policy; when
this work started, a leftover lock held them at ~2,184 MHz under load). Decode: greedy, thinking off, `max_tokens`
512, 30 s windows, tokens counted from `usage`. "Not rerun" marks measurements taken on r6v only. Method and full
tables under [Benchmarks](#benchmarks).

| Measurement | v1.0.0 | r6v (`v2.0.0`) | Current defaults (r6w) |
|---|---|---|---|
| Decode, one request, prose / code | 35.8 / 55.1 tok/s | 61.4 / 87.2 tok/s | **65.1 / 93.9 tok/s** |
| Decode, one request, structured / json | 48.8 / 48.0 tok/s | 80.6 / 88.5 tok/s | **84.5 / 93.7 tok/s** |
| Decode, 4 requests aggregate, prose / code | 89.9 / 132.4 tok/s | 204.4 / 229.4 tok/s | not rerun |
| Decode, 16 requests aggregate, prose / code | not possible (4 sequences) | 551.8 / 574.7 tok/s | not rerun |
| Prefill, unique text, 32K / 128K / 512K tokens | 1,902 / 1,724 / 1,282 tok/s | 6,013 / 5,916 / 5,042 tok/s | **6,622 / 6,505 / 5,507 tok/s** |
| Prefill, unique text, 1M tokens | 968 tok/s | 4,160 tok/s | not rerun |
| Needle at ~1M prompt tokens | PASS, 985 s (1,000,006 tokens) | PASS, 236.6 s (1,000,006 tokens) | **PASS, 229 s** (1,039,976 tokens) |
| Same 133,855-token prompt again after a restart | full prefill | full prefill | **TTFT 0.81 s** (20.7 s cold; [Persistent prefix cache](#persistent-prefix-cache)) |
| Salted cold needle, ~545K prompt tokens | PASS, 547,579 tokens at 1,157 tok/s | PASS, 545,008 tokens at 5,030 tok/s | not rerun |
| qeval, 75 tasks, greedy | 71/75 (median 48.4 tok/s) | 71/75 (median 86.3 tok/s) | not rerun |
| Quality gate, 50 exact-answer questions, greedy | 49/50 | 49/50, all 50 answers identical to v1.0.0 | 49/50 |
| API conformance (`api-test`) | 25/26 | 26/26 | not rerun |
| Encoding parity (`parity`) | 15/15 | 15/15 | not rerun |
| Token-id oracle (`bench check`) | 3/6 prompts match | 6/6, three runs | **6/6 against r6v's pin** |
| KV pool per rank | 2,666,467 tokens in 13 GiB | 8,580,566 tokens in 10.5 GiB (8.18× a full 1M request) | same 10.5 GiB budget; tokens not re-measured |
| Concurrent sequences at 1M | 4 | 16 | 16 |
| Boot | ~340 s to API | 396 s first boot, including the TileLang kernel preparation; restart 139-140 s | **restart 121 s** including the prefix KV save and restore (engine up after 97 s), from weight snapshots; ~184 s once per build and configuration ([Quick boot](#quick-boot)) |

Restart is `stop` + `start` + `smoke`, wall time. Temperature-0 output does not depend on load: on r6v an identical
request returned the same tokens and logprobs alone, eight at once, staggered, mixed with other prompts, and with or
without a prefix-cache hit ([Determinism](#determinism)); the current defaults still match r6v's token-id oracle on
all six prompts.

Memory: r6v at 8,192 batched tokens kept MemAvailable at 17.6-19.6 GiB per node idle and 17.6-19.5 GiB after a 1M
prefill and needle. The current 24,576 batched tokens spend some of that headroom: the lowest MemAvailable during
the 1M needle was 13.1 GiB on the head and 15.7-15.9 GiB on the other nodes, and 13.7 GiB on the tightest node
during the benchmarks, all well above the 3 GiB steady guard.

Sustained load (r6v): a 20-minute soak (16 rounds of 16 concurrent streams × 1,024 tokens) finished with 0 failed
requests and no NCCL, sparknet or CUDA errors on any rank; throughput held steady (217.1 → 217.4 tok/s for that
prompt set) and MemAvailable stayed at 17.7-19.5 GiB per node.

## What you need

- **Four DGX Sparks** (GB10, sm_121) on DGX OS with its **stock 4 KiB-page kernel**: no 64 KiB kernel, no memory
  saver. spark-ds41f's own site runs `tp4-ring` on a 64 KiB kernel with its memory saver; its external site receipt
  ran the same sizing on a stock 4 KiB kernel with 18-21 GiB MemAvailable per node ([Credits](#credits)). Nothing
  else may hold the GPUs: `start` refuses to run next to any container not in `ALLOWED_CONTAINERS`.
- **A ring of four QSFP cables**: each Spark cabled to its two neighbours; ranks 0-2 and 1-3 share no cable.
  **Both PCIe RoCE functions of every cable must be up** (on GB10 each cable shows as `rocep1s0f*` and
  `roceP2p1s0f*`), each with exactly one IPv4 address, one subnet per function per cable, the same prefix length
  everywhere, MTU 9000, and the IPv4 RoCE v2 GID at the same index on all of a node's ring ports
  ([The ring](#the-ring)).
- **A management LAN** reaching all four nodes (bootstrap, ssh, the API).
- **~480 GiB free NVMe per node** at the same `MODELS_DIR` path (the checkpoint is 475 GiB). Engram shards 47 and
  48 must be regular files on a local filesystem (not NFS, FUSE or SMB): the io_uring row reader reads them.
- **Passwordless sudo and `systemd-run` on every node**: the fail-closed memory guard runs as a systemd unit per
  node (`sudo -n systemd-run`), and `start` drops the page cache while the ranks load. `doctor` fails without them.
- **Rank 0 can build**: ~64 GiB MemAvailable to start the build (build while the node is idle), ~100 GiB free on
  `/`, and outbound HTTPS to github.com, PyPI and docker.io.
- **Docker** with GPU support on every node, and the **Hugging Face CLI** at `HF_BIN` (`fetch` downloads a quarter
  of the checkpoint on each rank).
- **A control machine** (a Mac works) with bash 4 or newer (`brew install bash`), python3, curl, rsync, and
  key-based ssh to all four nodes as `SSH_USER`; the nodes need key-based ssh to their ring neighbours' cable
  addresses (checkpoint and image copies run node to node).
- Optional host tuning that `doctor` reports but never applies: `vm.compaction_proactiveness=0` (WARN otherwise:
  decode stalls) and `vm.watermark_boost_factor=0` (NOTE otherwise: a watermark boost hides memory from
  MemAvailable, which the guard reads).

## Install

```bash
git clone <this repo> DeepSeek-V4.1-Flash-1017-4x-DGX-Spark
cd DeepSeek-V4.1-Flash-1017-4x-DGX-Spark
cp .env.example .env
```

Edit `.env`: `NODES` lists the four management addresses in ring cable order, rank 0 first. The `RING_*` pins in
`.env.example` are this site's; on another ring comment out `RING_PREFIX` (the ring is then scanned at every
`start`), run `doctor`, and paste the values it prints if you want pins back. Everything else has a working
default ([Configuration](#configuration)). Then:

| Step | What it does |
|---|---|
| `doctor` | read-only. Ring: scans every node, checks pins against the live ring, pings each next rank across its cable, warns on MTU ≠ 9000. Per node: docker, image ID equal to rank 0's and built from spark-ds41f `2dc95329` (image label), GPU processes, MemAvailable, NVMe and `/` free space, kernel and page size, sysctls, `CPUSET` equal to the node's fast cores, `DRI_CARD` present, passwordless sudo, `systemd-run`, checkpoint shard count, Engram shard filesystem and fragmentation (`e4defrag -c`), memory guard state |
| `fetch` | sizes the pinned files from the Hugging Face API and splits them into four size-balanced parts; each rank downloads its part over its own uplink. Rank 0 gathers the parts over the cables, fans the full tree back out, then every node is sha256-verified against `config/DeepSeek-V4.1-Flash-1017.sha256`. Resumable: files a node already holds at full size are not downloaded again |
| `verify` | the sha256 check alone, all four nodes in parallel |
| `build` | rsyncs `docker/` to `BUILD_DIR` on rank 0 and builds `IMAGE` there ([The image](#the-image)), runs spark-ds41f's GPU import smoke when rank 0 has 24 GiB available, streams the image over the cables with `docker save \| docker load` (0→1 and 0→3, then 1→2), and prints the image ID on every node |

## Run

```bash
./dsv41.sh start        # guards armed, workers first, waits for /health, steady guards, then smoke
./dsv41.sh smoke        # one request: 17*23 must come back as 391
./dsv41.sh status       # containers, memory guard state, available memory, /v1/models
./dsv41.sh logs 0 200   # tail one rank's engine log
./dsv41.sh bench decode # benchmarks (bench/bench.py; results in bench/runs/)
./dsv41.sh stop         # with KV_PERSIST=1 saves the prefix KV cache first; stops the guards and all ranks (STOP_GRACE s, then SIGKILL); each log is kept as ~/$CONTAINER.prev.log
./dsv41.sh restart
./dsv41.sh print-serve 0   # the exact guard arm + docker run + vllm serve for one rank, no side effects
```

`start` builds every rank's command first, so a configuration error starts nothing. It re-scans the ring and
refuses stale `RING_*` pins, refuses if foreign containers run, stops any old ranks, drops the page cache, locks the
GPU clocks on every node at `GPU_CLOCKS` MHz (2,200, site policy), and installs the memory guard on every node (a
node whose installed copy already matches the pinned checksums keeps it). Then it arms each node's startup guard and
runs the container on ranks 3, 2 and 1 (headless) in parallel, and finally on rank 0. It polls `/health` every
second for up to `WAIT_TIMEOUT` seconds; every 10 s it checks that each rank's container and guard are alive and
drops the page cache. If any rank's container exits or loses its guard it prints the last 40 log lines and the
guard's journal for every rank, stops the rest and fails. When the API answers it switches every node to the steady
guard and runs `smoke`. With `WEIGHT_SNAPSHOT=1` each rank restores its loaded weights from a snapshot in
`CACHE_DIR` instead of reading the checkpoint ([Quick boot](#quick-boot)); with `KV_PERSIST=1` the ranks restore the
prefix KV cache the last `stop` saved ([Persistent prefix cache](#persistent-prefix-cache)). The first boot of a new
image prepares the TileLang kernels into `CACHE_DIR` and writes the snapshots, and takes longer
([Known limits](#known-limits)).

### Go-live and rollback

These Sparks normally run another stack. `STOP_OTHER` and `RESTORE_OTHER` in `.env` are shell commands run on the
control machine; `.env.example` carries the GLM-5.3 gr8 TP4 stack as the example (`gr8 glm53 down` on rank 0;
`docker compose up -d` on all four nodes to bring it back).

```bash
./dsv41.sh stop-other      # take the other stack down
./dsv41.sh start           # this one up; smoke runs at the end
./dsv41.sh restore-other   # roll back: stop this stack, then RESTORE_OTHER
```

Use it like any OpenAI endpoint:

```bash
curl http://192.168.1.121:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Is 221 prime?"}],
  "reasoning_effort": "high",
  "max_tokens": 16384
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.121:8000/v1", api_key="unused")  # or your API_KEY

# thinking off
r = client.chat.completions.create(model="deepseek-v4.1-flash", max_tokens=4096,
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"thinking": False}})

# thinking on, top tier "max"; reasoning arrives in message.reasoning_content
r = client.chat.completions.create(model="deepseek-v4.1-flash", reasoning_effort="max", max_tokens=65536,
    messages=[{"role": "user", "content": "How many primes are below 60?"}])

# tools
weather = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
r = client.chat.completions.create(model="deepseek-v4.1-flash", tools=[weather], max_tokens=4096,
    messages=[{"role": "user", "content": "What's the weather in Rome?"}])
print(r.choices[0].message.tool_calls)

# image input (up to MAX_IMAGES per prompt)
r = client.chat.completions.create(model="deepseek-v4.1-flash", max_tokens=4096, messages=[{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}])
```

The API has no authentication unless `API_KEY` is set; set one if anything beyond your own machines can reach the
port. The key never appears in a process argv or in `print-serve`: `start` passes it over ssh stdin and into rank
0's container as `VLLM_API_KEY` (rank 0 only), `smoke`/`status` send it through a curl header file, and
`api-test`, `parity` and `bench` read it from the environment variable `API_KEY`.

## API behaviour

- **Thinking** is on when the request does not say otherwise. Turn it off with
  `chat_template_kwargs: {"thinking": false}` or `reasoning_effort: "none"`. Reasoning comes back as
  `reasoning_content`, separated by the `deepseek_v41` reasoning parser, and streams before the answer.
- **`reasoning_effort`** renders DeepSeek's 1017 thinking budget. The tiers are DeepSeek's; the other words the
  OpenAI schema accepts map onto them, so no OpenAI client gets a 400 for its effort value:

  | `reasoning_effort` | Budget | From |
  |---|---:|---|
  | `low` | 50 | DeepSeek 1017 tier |
  | `high` (default when unset) | 75 | DeepSeek 1017 tier |
  | `max` | 100 | DeepSeek 1017 tier |
  | `xhigh` | 75 (= `high`) | Local Inference Lab's vLLM fork, as in spark-ds41f |
  | `minimal` | 50 (= `low`) | our patch, `docker/0001-*.patch` |
  | `medium` | 75 (= `high`) | our patch, `docker/0001-*.patch` |
  | `none` | thinking off | vLLM |

  `parity` checks every tier and alias against DeepSeek's encoder.
- **Sampling defaults**: requests that set no sampling parameters get DeepSeek's 1017 defaults, temperature 1.0
  and top_p 1.0, server-side (`GENERATION_DEFAULTS`, rendered as `--override-generation-config`; the checkpoint's
  `generation_config.json` is not read). Anything the request sets wins.
- **`max_tokens`**: there is no server-side cap, by design. vLLM's `max_new_tokens` generation override is a hard
  server-wide cap, not a default: it would silently truncate requests that ask for more. A request that omits
  `max_tokens` may generate until the context is full, so always send it. `prompt_tokens + max_tokens` must not
  exceed 1,048,576 (`MAX_MODEL_LEN`), or vLLM answers 400.
  Reasoning tokens count toward it: leave room for both. As a guide, 4-8K with thinking off, 32-64K with thinking at
  `low`/`high`, 128K or more for hard `max`-effort work.
- **Tool calling**: `--tool-call-parser deepseek_v41 --enable-auto-tool-choice`; `tool_choice` `auto`, `none`,
  `required` and named functions; 1017 tool namespaces and the 1017 tool-loop turn rule.
- **Structured output**: `response_format` (JSON schema / JSON object).
- **Image input**: `image_url` content blocks, up to `MAX_IMAGES` (4) per prompt; DeepSeek's encoder places the
  image tokens. `TEXT_ONLY=1` drops image input.
- **Context**: 1,048,576 tokens per request (`MAX_MODEL_LEN`), `MAX_NUM_SEQS` (16) sequences at once; more
  requests queue.

## The ring

Four Sparks, four cables, no switch: rank r is cabled to r+1 and r−1 (mod 4); ranks 0-2 and 1-3 share no cable.
Every cable carries both PCIe RoCE functions, so each neighbour is reached over two lanes.

**How the pins are derived.** `doctor` (and `start` when `RING_PREFIX` is unset) runs a read-only scan on every
node: each up RoCE netdev with exactly one IPv4 address, its subnet, its RDMA device, the index of its IPv4 RoCE v2
GID, its MTU and RoCE MTU, and the management netdev that holds or routes to rank 0's address. Devices are matched
by subnet, never by name, so any port-to-neighbour cabling works. From the scan:

- the subnets rank r shares with rank r+1 are that cable's lanes (sorted, so both ends list them in the same order);
  ranks 0-2 and 1-3 must share none, and every cable subnet must have the same prefix length (`RING_PREFIX`);
- `RING_HCA_PREV_r` / `RING_HCA_NEXT_r` are rank r's RDMA devices on the lanes to rank r−1 / r+1;
- `RING_GID_r` is the one GID index those devices share (NCCL and sparknet take one index per process; `doctor`
  fails if they differ);
- `RING_MGMT_r` is the bootstrap netdev.

`doctor` then pings each next rank across its cable and prints the values. Pinned (all seventeen keys, or none),
`print-serve` needs no ssh; `doctor` and `start` still re-scan and refuse a pin that no longer matches the live ring
(a link flap can move a GID index). `fetch` and `build` always scan, because they need the cable addresses.

**What each rank gets** (spark-ds41f's `scripts/topology.py` mapping): `NCCL_IB_HCA` is the exact-match set of the
node's ring devices, `NCCL_IB_GID_INDEX` and `SPARKNET_ROCE_GID_INDEX` its GID index, `NCCL_IB_SUBNET_PREFIX_LEN`
the prefix, `SPARKNET_ROCE_PEER_HCAS` maps each neighbour rank to the devices whose cable reaches it, and NCCL,
Gloo and torch bootstrap on the management netdev with `VLLM_HOST_IP` the node's management address.

**Collectives.** Small all-reduces and all-gathers (up to 2 MiB) go through sparknet's one-shot RoCE collectives
over the ring (fail-stop: a timed-out wait stops instead of computing on unwritten buffers); larger ones use the
image's patched NCCL 2.30.7 with bidirectional switchless rings (`NCCL_SWITCHLESS_BIDIRECTIONAL=2`, ring algorithm,
no tree/PAT/NVLS, runtime connect). Traffic between the two ranks that share no cable (Engram's sequence-parallel
row exchange) is relayed through the neighbours by spark-ds41f's vLLM patch 0180 (PyNCCL all-to-all over the ring).
Nothing from the host is mounted over NCCL.

The rendezvous port (`MASTER_PORT`) and the cables are not authenticated: keep them private.

## Configuration

Precedence: environment, then `.env`, then the defaults in `dsv41.sh`. `.env` is parsed, not sourced; `#` starts a
comment only at the start of a value or after whitespace. Defaults are spark-ds41f r6w at shape `tp4-ring`
(`config/cluster.json` at `2dc95329`, unchanged since `cbf47749`) with this ring's tuning on top: 24,576 batched
tokens, NVFP4 drafter heads, DSpark k=6 with spark-ds41f's linear draft budget, GPU clocks locked at 2,200 MHz at
`start` (site policy), weight snapshots, a persistent prefix cache and a 5 s stop grace
([Current defaults against r6v](#current-defaults-against-r6v)). [`.env.example`](.env.example) documents every
key; the groups:

| Group | Keys | Defaults and notes |
|---|---|---|
| The Sparks | `NODES`, `SSH_USER` | four management addresses in ring cable order, rank 0 first (rank 0 serves the API); `spark` |
| Checkpoint | `MODELS_DIR`, `MODEL_REPO`, `MODEL_REVISION`, `MODEL_NAME`, `CHECKSUMS`, `HF_BIN` | `/home/spark/models`, same absolute path on every node (Engram shards 47/48 must be local files); revision `1bae0e1` pinned with a sha256 manifest; mounted read-only at `/models` |
| Image and fabric | `IMAGE`, `BUILD_DIR`, `CACHE_DIR`, `CONTAINER`, `MASTER_PORT` | `dsv41-1017-ds41f:r6w`; `CACHE_DIR` is mounted at `/cache` on every node and keeps the JIT/TileLang/Triton caches, weight snapshots and KV snapshots (~83 GiB per node) across boots; `CONTAINER` also names the guard unit (`$CONTAINER-memguard.service`); rendezvous on port 9999 |
| Serving | `SERVED_NAME`, `HOST`, `PORT`, `API_KEY` | `deepseek-v4.1-flash` on `0.0.0.0:8000`; `API_KEY` empty = no authentication |
| Sizing | `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`, `KV_CACHE_MEMORY_BYTES` | 1,048,576 tokens, 16 sequences, 24,576 tokens per step (the long-prefill threshold is this minus the decode rows, `MAX_NUM_SEQS` × (k+1): 24,464 at the defaults); a fixed 10.5 GiB KV budget per rank (`11274289152`; empty = vLLM profiling); utilization 0.82 is always passed and then only gates vLLM's startup free-memory check |
| Engine | `PREFIX_CACHING`, `PREFIX_CACHE_RETENTION_INTERVAL`, `KERNEL_BACKEND`, `GENERATION_DEFAULTS`, `TEXT_ONLY`, `MAX_IMAGES` | prefix caching on with retention interval 8192; `tilelang` (the TileLang SM121 family) or `b12x` (spark-ds41f's pre-r6 backend, in the same image); `{"temperature":1.0,"top_p":1.0}`; images on, 4 per prompt |
| Boot and stop | `WEIGHT_SNAPSHOT`, `KV_PERSIST`, `KV_SAVE_TIMEOUT`, `WAIT_TIMEOUT`, `STOP_GRACE` | weight snapshots on: each rank snapshots its loaded weights on the first boot of a build and configuration and restores them on later boots (`0` = always read the checkpoint; [Quick boot](#quick-boot)); persistent prefix cache on: `stop`/`restart` save the prefix KV cache, waiting up to 120 s, and `start` restores it (`0` = cold cache after every restart; [Persistent prefix cache](#persistent-prefix-cache)); `start` waits up to 3600 s; `stop` gives vLLM 5 s to exit before SIGKILL (its graceful exit took ~14 s) |
| DSpark and CUDA graphs | `DSPARK`, `NUM_SPECULATIVE_TOKENS`, `DRAFT_SAMPLE_METHOD`, `ADAPTIVE_VERIFICATION_ROW_COST`, `EAGER`, `COMPILATION_CONFIG` | DSpark on (the drafter ships in the checkpoint), k=6, greedy drafts, block verification, adaptive verification by spark-ds41f's linear draft budget at row cost 0.01 (`docker/0007`; no boot-time profiling; empty = off); `FULL_AND_PIECEWISE` graphs on spark-ds41f's capture ladder up to `MAX_NUM_SEQS` × (k+1) rows (112 at the defaults); `EAGER=1` disables graphs; `COMPILATION_CONFIG` replaces the compilation JSON verbatim |
| Host resources | `CPUSET`, `CONTAINER_MEMORY`, `DRI_CARD`, `DISPLAY_CARVEOUT`, `GPU_CLOCKS` | ranks pinned to GB10's ten Cortex-X925 cores `5-9,15-19` (`doctor` checks them per node); `--memory`/`--memory-swap` 112g; the nvidia DRM card bound at `/dev/dri/card0`; embedding and head in the display carve-out off (`1` needs `DRI_CARD` and `nvidia-drm` modeset); empty disables each of these. `GPU_CLOCKS=2200` (site policy) locks every node's GPU at 2,200 MHz with `nvidia-smi -lgc` at `start` (2,184-2,197 MHz under load); `keep` leaves the clocks as they are |
| Memory guard | `STARTUP_MEMGUARD_GIB`, `STARTUP_MEMGUARD_INTERVAL`, `STARTUP_MEMGUARD_WAIT`, `MEMGUARD_GIB`, `MEMGUARD_INTERVAL` | 5 GiB sampled every 0.25 s while loading (waits up to 120 s for its container); 3 GiB every 2 s once the API answers. The startup threshold must exceed the steady one ([Known limits](#known-limits)) |
| Other stacks | `ALLOWED_CONTAINERS`, `STOP_OTHER`, `RESTORE_OTHER` | containers that may keep running beside the ranks (`.env.example`: `mentatd glm53-nfs`); shell commands for `stop-other` / `restore-other` |
| Advanced | `EXTRA_VLLM_ARGS`, `EXTRA_DOCKER_ARGS` | appended to `vllm serve` / to `docker run` before the image (`-e K=V` overrides an engine variable) |
| Ring pins | `RING_PREFIX`, `RING_HCA_PREV_0..3`, `RING_HCA_NEXT_0..3`, `RING_GID_0..3`, `RING_MGMT_0..3` | all seventeen or none ([The ring](#the-ring)); `.env.example` carries this site's values |

`./dsv41.sh print-serve RANK` prints the exact command a rank runs with your configuration.

## How it is put together

| Piece | What |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017, native checkpoint (MXFP4 routed experts, FP8 dense, BF16 drafter head), revision `1bae0e1`, every file pinned by sha256 in `config/`; every node keeps a full local copy |
| Engine | spark-ds41f r6w by Christopher Owen: Local Inference Lab's vLLM `integration/karmic-kraken-beta` at `19f2c20` + his r6w vLLM series (173 patches: r6v's 172 plus 0181, the routed SwiGLU and MXFP8 cast over the live tiles), his lossless MoE scale-codes overlay (`a250ace`) and his linear DSpark draft budget (r6x's vLLM 0184, rebased as `docker/0007-*.patch`); B12X `236ddff` + 11 patches (checkpoint loader, vision tower kernels, the alternative kernel backend); TileLang 0.1.15 `b95ee4f` + 4 patches (among them SM12x block-scaled MMA); DeepSeek TileKernels 2.0.0 `66258df` + 1 patch; sparknet 0.3.0 `3bc8d10`; NCCL 2.30.7 rebuilt for SM121 + 4 patches (the IB send-path fence NVIDIA/nccl#2393, bidirectional switchless rings, balanced channels, adaptive small-ring threads); runtime base the vLLM nightly image `af1c014` (CUDA 13.0.2), pinned by digest |
| Our patches | `docker/0001-*.patch`: `reasoning_effort` `minimal` → 50 and `medium` → 75, which the fork rejected with a 400; `docker/0002-*.patch`: per-rank snapshots of the loaded weights, restored on later boots ([Quick boot](#quick-boot)); `docker/0003-*.patch` and `docker/0004-*.patch`: ports of vllm-project/vllm#60945 (no lookahead past the last logit in block verification's residual mass) and #61161 (fresh acceptance draws per draft step); `docker/0005-*.patch`: the CED decoder skipped on non-final prefill chunks no later window reads (exact: token oracle, greedy text and logprobs unchanged); `docker/0006-*.patch`: the MTP hidden buffer DSpark never reads dropped; `docker/0008-*.patch`: the prefix KV cache saved before a stop and restored at start ([Persistent prefix cache](#persistent-prefix-cache)) |
| Kernels | the TileLang SM121 family (sparse MLA, MXFP4 indexer, FP8 and BF16 projections, FP8 × FP4 routed experts, vocabulary heads, RoPE, compressor, KV writers, Engram gate/hash/gather, WO), DeepSeek's TileKernels for the MoE gate, SwiGLU/quant casts and mHC, L2 weight prefetch during decode (`KERNEL_BACKEND=tilelang`) |
| Precision | spark-ds41f follows DeepSeek's reference `inference/model.py`: FP32 logits from an exact packed BF16 head, FP32 mHC projection, shared-expert SwiGLU in FP32, MoE partials rounded once per rank, WO-A input through `act_quant`; batch-invariant reductions and rank-ordered collectives (his description; [Credits](#credits)) |
| Engram | tables read from local disk by a native io_uring row reader (`--engram-config` `table_memory: disk`), projections sharded over TP, sequence-parallel row exchange relayed on the ring |
| Drafting | DSpark (the drafter in the checkpoint): k=6, greedy drafts, block verification, adaptive verification by spark-ds41f's linear draft budget (`docker/0007-*.patch`): each step's draft budget priced at a fixed row cost of 0.01, nothing timed at boot; the draft vocabulary and Markov heads run in NVFP4 (`VLLM_DS41_DRAFT_NVFP4_HEAD=1`, `VLLM_DS41_MARKOV_NVFP4=1`: draft step 2.60 → 1.91 ms), and the target still commits every token |
| Graphs | `FULL_AND_PIECEWISE` CUDA graphs captured at 1, 2, 3, 4, 6, 8, 12 … 32, then every 8th row up to `MAX_NUM_SEQS` × (k+1) = 112; kernels prepared before memory profiling, `--jit-monitor-mode warn` logs any JIT after ready |
| Split | TP4 across the four Sparks (attention, dense and experts), vLLM's multiprocess executor over `--nnodes 4`; rank 0 serves the API, ranks 1-3 headless; one prefill at a time |
| KV | fixed 10.5 GiB per rank: 8,580,566 tokens, 8.18× a full 1M request (measured at boot on r6v) |
| Fabric | sparknet one-shot collectives up to 2 MiB, patched NCCL ring above that, both PCIe functions per cable ([The ring](#the-ring)) |
| Host | per rank: the ten fast cores, a 112 GiB cgroup limit, `seccomp=unconfined` (Docker's default profile blocks io_uring), `--ipc=host`, unlimited memlock; spark-ds41f's fail-closed memory guard as a systemd unit per node; GPU clocks locked at 2,200 MHz (`nvidia-smi -lgc`, site policy) at every `start` |
| Boot | each rank restores its loaded weights (71.86 GiB) from a snapshot under `CACHE_DIR` at 10-12.7 GB/s and reruns post-processing on them; fail-closed to the checkpoint ([Quick boot](#quick-boot)); then the prefix KV cache (10.5 GiB per rank) the last `stop` saved, all ranks or none ([Persistent prefix cache](#persistent-prefix-cache)) |
| Controller | `dsv41.sh`: bash over ssh from the control machine; workers launched in parallel, `/health` polled every second; `print-serve` shows each rank's exact commands |

### The image

`docker build -f docker/Dockerfile` (linux/arm64) reads only `docker/`. Its `fetch` stage clones spark-ds41f at
`2dc95329` and runs **his own** build-preparation code (`bin/spark build prepare`) against the r6w lock
(`experiments/2026-10-18-r6w-swiglu`): each upstream is fetched at its pinned commit and its patch series applied
unmodified with `git am`; the patch-set fingerprint, patch head, tree and TileLang's submodule commits must equal his
manifest, and the Dockerfile's pins must equal his lock. The build stages after `fetch` are his `docker/Dockerfile`
at that commit with his build-time checks kept (B12X imports and liburing, NCCL version and sha256,
TileLang/TileKernels wheel records, sparknet proxy ABI). Ours on top: a `dsv41-vllm` stage that builds the serving
vLLM tree from his in this order: our `docker/0001-*.patch` and `docker/0002-*.patch`; his MoE scale-codes overlay
(spark-ds41f `a250ace`, three whole files, each pinned by sha256, committed as one commit); our
`docker/0003-*.patch` to `docker/0006-*.patch`; his DSpark profile-replays overlay (`2ce3644`, one file, committed
the same way); `docker/0007-*.patch`, his r6x vLLM 0184 rebased onto that tree, whose `adaptive_verification.py`
(replacing the profile-replays file) and `speculative.py` must equal his overlay's by sha256; and our
`docker/0008-*.patch`. All of it is Python, so his native extensions build unchanged; the patch head and tree after
each overlay, after 0007 and after 0008 are pinned. Then a TileLang SM12x block-scaled MMA check, file checks of our
patches, both overlays and the linear budget, his memory-guard and smoke scripts under `/opt/spark-ds41f`, and image
labels that `doctor` checks. Any mismatch fails the build. The image contains no weights. spark-ds41f's external
4 KiB site reports its build at 630 s with MemAvailable never below 73.7 GiB.

### Quick boot

Each rank used to spend 31.3 s of every boot loading its share of the checkpoint. With `WEIGHT_SNAPSHOT=1` (the
default; the engine sees `DSV41_WEIGHT_SNAPSHOT=1`), our `docker/0002-*.patch` hooks vLLM's model loader:

- **First boot of a build and configuration**: each rank loads the checkpoint as usual and, right after
  `load_weights` and before post-processing, writes every storage behind its parameters and buffers (71.86 GiB per
  rank) to one file under `CACHE_DIR/dsv41/weight-snapshots`, with a manifest (tensor layout, a CRC32 per 64 MiB
  chunk, the plain-data attributes `load_weights` set) and a completion marker. That boot took ~184 s here, once;
  ranks 0 and 1 wrote at 2.6-3.1 GB/s, ranks 2 and 3 at ~11.7 GB/s.
- **Later boots**: each rank builds the model, reads its snapshot with O_DIRECT straight into the same storages at
  10-12.7 GB/s, CRC32-checked, and then runs the usual post-processing, which sees bit-identical inputs and rebuilds
  every derived tensor itself. Model loading drops from 31.3 to 18.4 s per rank.
- **Fail-closed**: every rank must hold a valid snapshot for the same key, or every rank loads the checkpoint; a
  snapshot that fails to read or check is deleted. The key covers the checkpoint files (sizes, mtimes, safetensors
  headers), the rank, the vLLM and B12X sources, Torch, the device, vLLM's config hash, the engine environment and
  `DSV41_WEIGHT_SNAPSHOT_TAG`, so a rebuilt image or a changed configuration writes a new snapshot.

With the controller changes (memory guard reused when its pinned checksums match, workers launched in parallel,
`/health` polled every second, `STOP_GRACE=5`), a restart (`stop` + `start` + `smoke`) took 110 s instead of
139-140 s, with the engine up after 88 s instead of 109 s; at the current defaults, with the prefix KV save and
restore on top, it takes 121 s, engine up after 97 s. The patch's own knobs, set with `EXTRA_DOCKER_ARGS`
(`-e K=V`): `DSV41_WEIGHT_SNAPSHOT_DIR` (default `/cache/dsv41/weight-snapshots`), `DSV41_WEIGHT_SNAPSHOT_TAG`
(change it to invalidate every snapshot), `DSV41_WEIGHT_SNAPSHOT_KEEP` (snapshots kept per rank, default 1),
`DSV41_WEIGHT_SNAPSHOT_VERIFY` (`0` skips the CRC32 check) and `DSV41_WEIGHT_SNAPSHOT_DIGEST` (`1` writes per-tensor
CRC32 digests of the processed model, to compare a restored boot with a checkpoint boot). The idea of per-rank
weight snapshots comes from [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10)'s
processed-weight snapshots; the implementation is ours, no code taken ([Credits](#credits)).

### Persistent prefix cache

A restart used to drop the prefix cache: the first request for a long prompt after it paid the full prefill again
(minutes at 128K tokens). With `KV_PERSIST=1` (the engine sees `DSV41_KV_PERSIST=1` on every rank), our
`docker/0008-*.patch` keeps it across `stop`/`restart`:

- **Save**: `stop` (and so `restart`) first sends `POST /v1/dsv41/kv_snapshot?abort=true` (under `/v1`, so
  `API_KEY` guards it) and waits up to `KV_SAVE_TIMEOUT` (120 s). The engine pauses its scheduler, aborting any
  running request, waits until no step is in flight, and every rank writes its whole KV backing buffer (10.5 GiB,
  O_DIRECT, a CRC32 per 64 MiB chunk) under `CACHE_DIR/dsv41/kv-snapshots`; the scheduler's prefix-cache table
  (block hashes, the hash map, the eviction order) is written last, only if every rank succeeded. If the save
  fails or times out, `stop` logs it and stops anyway. An empty cache saves nothing.
- **Restore**: at the next start, after warmup and before the API answers, every rank checks its snapshot and
  reads it back, and the table is installed only if every rank succeeded, so the same prompt is a prefix hit at
  the same block ids, byte for byte the cache the engine held before the stop. Otherwise the engine starts with an
  empty cache and logs why (`KV snapshot not restored (...)`, naming the key factors that changed).
- **Invalidation**: the key covers the checkpoint files, the vLLM, B12X and TileLang sources, Torch, the device,
  Python and the block hash function, vLLM's config hash and HF config, the cache and KV cache configuration
  (block sizes, groups, number of blocks, layout), the parallel layout, the engine environment and
  `DSV41_KV_PERSIST_TAG`. A rebuilt image or any changed flag starts cold and the next stop writes a new snapshot,
  replacing the old (one kept per node). A snapshot that fails a check (marker, size, CRC32, table consistency)
  is deleted; a rank whose snapshot is from another save, or any other mismatch, leaves every rank cold.
- **Limits**: only a graceful `stop`/`restart` saves; a crash or a guard kill starts cold, or from the last save if
  nothing has changed since (its blocks are still valid for the same key). Restored blocks hold prompt contents:
  files are owner-only, and `DSV41_KV_PERSIST_TAG` or deleting the directory discards them.

The patch's own knobs, set with `EXTRA_DOCKER_ARGS` (`-e K=V`): `DSV41_KV_PERSIST_DIR` (default
`/cache/dsv41/kv-snapshots`), `DSV41_KV_PERSIST_TAG`, `DSV41_KV_PERSIST_KEEP` (default 1) and
`DSV41_KV_PERSIST_VERIFY` (`0` skips the CRC32 check on restore). The weight snapshot key leaves the
`DSV41_KV_PERSIST*` switches out, so turning this on keeps the weight snapshots.

## Benchmarks

`./dsv41.sh bench STEP` runs [`bench/bench.py`](bench/bench.py) (stdlib only) from the control machine: `decode`,
`prefill`, `needle`, `spec`, `pin`, `check` or `all`. Runs land in `bench/runs/<stamp>-<step>/` (gitignored);
kept runs are copied unchanged into `bench/records/` ([how](bench/records/README.md)). `pin` writes a token-id
regression oracle to `bench/reference.json`; `check` replays it. Decode cells: greedy, thinking off, `max_tokens`
512, c workers issuing back-to-back streaming requests for a 30 s window; aggregate tok/s = sum of
`usage.completion_tokens` / cell wall time (DSpark emits several tokens per chunk, so chunks are never counted).

The qualification battery adds three external clients, not shipped here: MiaAI-Lab's `qeval.py` (75 tasks, greedy),
the salted cold needle (`needle_salted.py`) from spark-ds41f's external site receipt, and spark-ds41f's
`token_determinism_text.py` (`experiments/2026-10-12-r6g`), so results compare with their records.

### Current defaults against r6v

Measured on 2026-10-18 at the 2,200 MHz clock lock with the same client and settings as the r6v record
([`bench/records/2026-10-18-r6v`](bench/records/2026-10-18-r6v/summary.md)): decode aggregate tok/s, greedy,
thinking off, `max_tokens` 512, 30 s windows; prefill on unique prompts (a per-run nonce, so no restored prefix
hits). The current defaults are `autoresearch.sh` run #26 after a discarded warm-up pass, recorded in
[`bench/records/2026-10-18-autoresearch`](bench/records/2026-10-18-autoresearch/summary.md).

| Measurement | r6v (`v2.0.0`) | Current defaults | Gain |
|---|---:|---:|---:|
| Decode c=1, prose (tok/s) | 61.4 | 65.1 | +6.0% |
| Decode c=1, code | 87.2 | 93.9 | +7.7% |
| Decode c=1, structured | 80.6 | 84.5 | +4.8% |
| Decode c=1, json | 88.5 | 93.7 | +5.9% |
| Decode c=8, prose / code / structured / json | 274.8 / 422.7 / 436.2 / 307.3 | 386.3 / 321.7 / 479.7 / 370.2 | +8.5 to +24% (mean, boot-dependent) |
| Prefill 32K (tok/s) | 6,013 | 6,622 | +10.1% |
| Prefill 128K | 5,916 | 6,505 | +10.0% |
| Prefill 512K | 5,042 | 5,507 | +9.2% |
| Needle at ~1M prompt tokens | PASS, 236.6 s (1,000,006 tokens) | PASS, 229 s (1,039,976 tokens) | |
| Quality gate, 50 exact answers | 49/50 | 49/50 | |
| Corruption probe (thinking on) | 9/13 | 9/13 | |
| Token-id oracle (r6v's pin) | 6/6 | 6/6 | |
| Restart (`stop` + `start` + `smoke`) / engine up | 139-140 s / 109 s | 121 s / 97 s, including the prefix KV save and restore | |
| Model loading per rank | 31.3 s | 18.4 s | |
| Same 133,855-token prompt after a restart, TTFT | 20.7 s (cold) | 0.81 s | |

Mean gain: decode c=1 **+6.09%** (best run +6.18%), prefill **+9.78%** (+9.4 to +9.8 across boots). c=8 is
boot-bimodal: the same build gave code 321 or 456 and json 370 or 506 tok/s depending on the boot, stable within a
boot (three repeats within ±0.1%; [Known limits](#known-limits)). Before the site's clock lock, with the clocks
unlocked at 2,470-2,502 MHz, prefill on this branch read +11.6 to +12.2%; every number here is at the lock. What
changed, in the order it landed (the branch's commit messages carry each measurement):

1. **GPU clocks locked at 2,200 MHz** (`GPU_CLOCKS=2200`, site policy; 2,184-2,197 MHz under load): replaced the
   branch's earlier `reset` default; the clock is policy, never tuned for speed.
2. **Engine r6v → r6w** (spark-ds41f `2dc95329`): vLLM 0181 runs the routed SwiGLU over the live tiles only; outputs
   are bit-identical.
3. **NVFP4 drafter heads** (`VLLM_DS41_DRAFT_NVFP4_HEAD=1`, `VLLM_DS41_MARKOV_NVFP4=1`): a draft step takes 1.91 ms
   instead of 2.60 ms (decode c=1 +1.3%); only the proposals change, the target still commits every token.
4. **Quick boot**: the controller changes, `STOP_GRACE=5` and the weight snapshots of `docker/0002-*.patch`
   ([Quick boot](#quick-boot)).
5. **Scale-codes overlay** (spark-ds41f `a250ace`): the routed experts' scales as row bases and 4-bit codes,
   bit-identical outputs; single stream +2.1%, 2.1 GiB per rank freed.
6. **k=6** (`NUM_SPECULATIVE_TOKENS`): the cheaper drafter heads make a deeper draft pay; the graph ladder grows from
   96 to 112 rows.
7. **`docker/0003-*.patch` to `docker/0006-*.patch`**: ports of vllm-project/vllm#60945 and #61161; ours, the CED
   decoder skipped on non-final prefill chunks no later window reads (exact: token oracle 6/6, greedy text and
   logprobs identical at 30K, 116K and 28K prompts); ours, the write-only MTP hidden buffer dropped.
8. **`MAX_NUM_BATCHED_TOKENS` 8,192 → 16,384 → 24,576** (long-prefill threshold 24,464): larger prefill steps, the
   second step +2.4 points of prefill, paid for in MemAvailable (below).
9. **Linear draft budget** (`docker/0007-*.patch`, spark-ds41f's r6x vLLM 0184, rebased) at
   `ADAPTIVE_VERIFICATION_ROW_COST=0.01` (screen: 0.005 +6.04%, 0.01 +6.35%, 0.015 +6.18%, 0.02 +6.12% decode). It
   replaces the cost curves timed at boot (and the 25-replay profile and cost scale 1.0 that briefly preceded it): no
   boot-time profiling (restart 168 → 120 s) and no per-build profile noise. The controller keys
   `ADAPTIVE_VERIFICATION_COST_SCALE` and `DSPARK_COST_DIR` are gone.
10. **Persistent prefix KV cache** (`docker/0008-*.patch`, `KV_PERSIST=1`): a 133,855-token prompt's TTFT went from
    20.7 s cold to 0.81 s after a restart (133,632 tokens hit), tokens and top-5 logprobs bit-identical to uncached,
    cold and in-process-hit runs; save 1.2-5 s, restore 1.89 s; one flipped byte on rank 2 failed the CRC32, every
    rank started cold with correct output and the corrupt snapshot was deleted
    ([Persistent prefix cache](#persistent-prefix-cache)).
11. **Harness**: `autoresearch.sh` discards a warm-up pass after each restart (decode is 1-2.5% slower in the first
    minutes after a boot).

**Checks** on the current defaults: `bench check` against the oracle pinned on r6v, 6/6 prompts match; quality gate
49/50; the thinking-mode corruption probe at 9/13, as on r6v ([Known limits](#known-limits)). `api-test`, `parity`,
qeval, the salted needle, the determinism suite, 1M prefill and c=2/c=4/c=16 decode were not rerun.

**Memory**: the lowest MemAvailable during the 1M needle at 24,576 batched tokens was 13.1 GiB on the head and
15.7-15.9 GiB on the other nodes; during the benchmarks the tightest node sat at 13.7 GiB.

**Why c=1 decode stops at ~+6%**: under the profiler a c=1 step takes 28.2 ms. The routed MoE is ~36% of kernel
time at the LPDDR5x bandwidth floor (~11 distinct experts per layer across the verification rows); the attention
MXFP8 projections ~17%; 88 one-shot all-reduces at ~46 µs each ~14% (fabric latency and rank skew); the drafter
~2.8 ms per step (~10%); the BF16 target LM head 1.21 ms, its floor (331 MB per rank); idle gaps 0.13 ms per step.
DSpark commits 1.94 (prose), 2.82 (structured), 3.20 (json) and 3.29 (code) tokens per step. Copy or suffix
drafting would add 0-2% on the bench prompts. What is left is acceptance: draft trees (ancestor masks through sparse
MLA, SWA, the compressor, indexer, Engram and CED) or a better drafter; neither was pursued, nor lossy speedups.
Prefill: collectives are ~26% of a 128K prefill (the AllGather of the compressor layers' 200 MB hidden state 11.5%;
the MoE's rank-ordered reduce-scatter serial on the main stream); overlapping them needs MoE row slicing that
re-reads the expert weights, which cancels the gain.

**Measured and rejected** (on this ring, defaults kept):

- DSpark dead-row tau 0, 0.1, 0.15, 0.35 and 0.5 (0.2 stays);
- adaptive-verification cost scales 0.75 to 12 on the timed curves (moot since `docker/0007-*.patch`);
- L2 weight-prefetch grid 1 and 4 (2 stays), and 24 MB prefetch budgets;
- k=4 and k=7 (k=6 stays);
- a single RoCE lane per cable for sparknet;
- unpinned CPUs (`CPUSET` stays);
- `VLLM_DS41_HEAD_DTYPE=model`: its apparent +3% was a different greedy trajectory, with ms per round unchanged;
- memory compaction;
- `KERNEL_BACKEND=b12x`;
- probabilistic drafting (spark-ds41f #393) and a fixed-threshold draft budget (#397);
- NCCL with 8 channels; `NCCL_PROTO=Simple` and `NCCL_BUFFSIZE` 16 MiB for prefill (+9.49% and +8.93% against a
  +9.82% control);
- `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=2048`;
- `MAX_NUM_BATCHED_TOKENS` 28,672 (prefill +9.15%, MemAvailable down to 11.2 GiB) and 32,768 (~1 point more prefill
  than 24,576, idle MemAvailable 13.7 → 9.1 GiB; declined to keep the headroom).

### r6v against v1.0.0

Both measured on this ring at 1M context, checkpoint `1bae0e1`, each with its own defaults: v1.0.0 (the previous
engine, [fallback](#fallback)) on 2026-10-17 with 4 sequences and DSpark k=3; r6v on 2026-10-17/18 with
16 sequences and DSpark k=5, image `sha256:875c7c9edc08…`. Records:
[`bench/records/2026-10-17-v1.0.0`](bench/records/2026-10-17-v1.0.0/summary.md) and
[`bench/records/2026-10-18-r6v`](bench/records/2026-10-18-r6v/summary.md) (c=16 in its `decode-c16/`).

**Decode**, aggregate tok/s, v1.0.0 → r6v (speedup):

| Prompt | c=1 | c=2 | c=4 | c=8 | c=16 (r6v only) |
|---|---:|---:|---:|---:|---:|
| prose | 35.8 → 61.4 (1.7×) | 60.4 → 116.6 (1.9×) | 89.9 → 204.4 (2.3×) | 85.5 → 274.8 (3.2×) | 551.8 |
| code | 55.1 → 87.2 (1.6×) | 88.2 → 159.3 (1.8×) | 132.4 → 229.4 (1.7×) | 139.1 → 422.7 (3.0×) | 574.7 |
| structured | 48.8 → 80.6 (1.7×) | 77.4 → 147.5 (1.9×) | 116.6 → 258.1 (2.2×) | 112.3 → 436.2 (3.9×) | 566.0 |
| json | 48.0 → 88.5 (1.8×) | 78.4 → 161.0 (2.1×) | 115.7 → 293.1 (2.5×) | 115.0 → 307.3 (2.7×) | 639.1 |

One r6v stream decodes at 62.3 (prose) to 89.0 (code) tok/s; at c=16 each stream still gets 34.2-43.3 tok/s. TTFT:
r6v 0.13-0.16 s at c=1, 0.24-0.39 s at c=8 and 0.34-0.45 s at c=16; v1.0.0 0.16-0.20 s at c=1, and 14-24 s at c=8
because half its requests queued behind 4 sequences.

**DSpark acceptance**, accepted tokens per draft step at c=1 (`bench spec`):

| Prompt | v1.0.0 (k=3) | r6v (k=5) |
|---|---:|---:|
| prose | 1.045 | 0.936 |
| code | 2.183 | 2.179 |
| structured | 1.824 | 1.845 |
| json | 1.735 | 2.072 |
| all | 1.662 | 1.741 |

Acceptance is about level between the two engines, so the decode speedup comes from shorter steps.

**Draft length** (`NUM_SPECULATIVE_TOKENS`), k=3 against the default k=5 on r6v, aggregate tok/s
(`.work` logs in the site receipt):

| prompt | c=1 k=5 → k=3 | c=4 k=5 → k=3 | c=16 k=5 → k=3 |
|---|---:|---:|---:|
| prose | 61.4 → 61.2 | 204.4 → 214.3 | 551.8 → 474.7 |
| code | 87.2 → 82.5 | 229.4 → 229.5 | 574.7 → 487.9 |
| structured | 80.6 → 77.0 | 258.1 → 254.6 | 566.0 → 473.6 |
| json | 88.5 → 81.4 | 293.1 → 279.6 | 639.1 → 521.3 |

k=5 stayed on r6v: k=3 is level to 8% slower at c=1 and 14-18% slower at c=16; only prose at c=4 gains (5%). The
current defaults use k=6, which the NVFP4 drafter heads made pay ([Current defaults against r6v](#current-defaults-against-r6v)).

**Prefill**, unique text (no prefix-cache hits), tok/s:

| Prompt tokens | 4K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.0.0 | 2,138 | 1,817 | 1,902 | 1,810 | 1,724 | 1,473 | 1,282 | 968 |
| r6v | 5,105 | 5,753 | 6,013 | 6,004 | 5,916 | 5,579 | 5,042 | 4,160 |
| speedup | 2.4× | 3.2× | 3.2× | 3.3× | 3.4× | 3.8× | 3.9× | 4.3× |

**Checks:**

| Check | v1.0.0 | r6v |
|---|---|---|
| Needle, 1,000,006 prompt tokens (`bench needle`) | PASS, 985 s | PASS, 236.6 s (4.2×) |
| Salted cold needle | PASS, 547,579 tokens in 473 s (1,157 tok/s) | PASS, 545,008 tokens in 108 s (5,030 tok/s; 4.3×) |
| qeval, 75 tasks | 71/75, median 48.4 tok/s | 71/75, median 86.3 tok/s |
| Quality gate, 50 exact-answer questions, greedy | 49/50 | 49/50, all 50 answers identical to v1.0.0's |
| `api-test` | 25/26 (`greedy_is_deterministic` failed on a near-tie) | 26/26 |
| `parity` | 15/15 | 15/15 |
| `bench check` against the engine's own `bench pin` oracle | 3/6 prompts match (near-tie divergence) | 6/6 prompts match, in each of three runs |

qeval's per-category result is the same on both engines: code 24/25, format 7/7, json 6/8, math 13/14, prose 5/5,
reason 16/16.

### Determinism

spark-ds41f's `token_determinism_text.py --tokens 192 --long-lines 700 --wide` against r6v compares every response
with a reference run of the same prompt, temperature 0, with logprobs. Every scenario came back identical, 0 token
and 0 logprob differences: eight requests at once (`c8`), staggered starts, mixed with other prompts (`mixed`,
`mixed15`), a prefix-cache hit against a miss (`cached`), and a repeat (`again`). The token-id oracle
(`bench pin`, then `bench check` three times) matched on all six prompts every time.
The current defaults still match that r6v oracle on all six prompts; the determinism suite was not rerun on them.

### Boot and memory (r6v)

| Measurement | Result |
|---|---|
| First boot to API, including the TileLang kernel preparation | 396 s |
| CUDA graph capture | 34 s, 1.83 GiB |
| KV pool | 8,580,566 tokens in 10.5 GiB per rank (8.18× a full 1M request) |
| MemAvailable per node, idle | 17.6-19.6 GiB |
| MemAvailable per node after a 1M prefill and the 1M needle | 17.6-19.5 GiB |

On r6v the startup guard (5 GiB) and steady guard (3 GiB) left ~14 GiB of margin on the tightest node. The current
defaults leave ~10 GiB above the steady guard at the lowest point of the 1M needle (13.1 GiB on the head).

### For reference: spark-ds41f's published numbers

Their numbers, from their harnesses and sites, not measured here and not directly comparable with the tables above:

- spark-ds41f r6v, TP4 at 1M on its own ring (64 KiB kernel), `release` suite, 2026-10-17: prefill 5,982 tok/s at
  32K and 4,405 at 1M; 16-stream decode 243.8 tok/s; one conversation at a time 71.1 tok/s (TTFT p50 0.10 s);
  KV capacity 8,580,566 tokens in 10.5 GiB per rank.
- spark-ds41f's external site receipt (`experiments/2026-10-17-site-tp4-4k-1017`), r6u image with r6v's vLLM
  0180 on a stock 4 KiB kernel, same 1017 checkpoint and sizing as here: salted needle PASS at 992,974 tokens in
  225 s (4,413 tok/s) and at ~545K tokens (5,225-5,239 tok/s); qeval 71/75; MemAvailable 18-21 GiB per node idle
  and through a 993K-token prefill; ready 216.7 s after a first boot that prepared 188 TileLang calls per rank.

## Tests

| Test | Needs | What it guards |
|---|---|---|
| `tests/test-recipe.sh` | nothing (no GPU, ssh or network) | generates every rank's guard arm + `docker run` + `vllm serve` with pinned ring values and checks the invariants the deployment depends on (spark-ds41f `tp4-ring` with this repo's defaults, the weight-snapshot and KV-persist switches equal on every rank); shell and python syntax |
| `tests/api_conformance.py` (`./dsv41.sh api-test`) | live API | 26 cases: endpoints, sampling controls, thinking/reasoning, every effort tier, tool calling, structured output, image input, long context, auth and error handling; one PASS/FAIL line per case |
| `tests/test_encoding_parity.py` (`./dsv41.sh parity`) | live API, network | the server's rendered prompt equals DeepSeek's `encoding.py` at the pinned revision, token for token: golden fixtures, thinking on/off and default, every effort tier and alias, namespaced tools, the tool-loop turn rule |
| `bench check` | live API | token-id regression against `bench/reference.json` |

## Known limits

- **Open defect: thinking-mode answers can collapse.** With thinking on (the default), some short answers hit a
  position where the next-token distribution is flat garbage: stray duplicated fragments at temperature 0,
  unrelated text at the default temperature 1.0. Prefill reproduces it, so CUDA graphs and DSpark are ruled out.
  Thinking-off requests were clean in every sample. Root cause unknown; see
  [docs/findings/2026-10-18-thinking-mode-output-corruption.md](docs/findings/2026-10-18-thinking-mode-output-corruption.md).
  The current defaults leave it unchanged: the corruption probe (`bench/corruption_probe.py`) reports 9/13 on r6v and
  on them.
- **Image token in text → 400.** A message whose text contains DeepSeek's literal image token
  (`<｜deepseek_image｜>`) is rejected: "Images should be provided as image content blocks". This is DeepSeek's
  encoder validation, applied to every role (spark-ds41f's vLLM 0178 follows it). Agents that paste DeepSeek source
  code or docs into a prompt or tool result hit it.
- **First boot is slow.** Each rank prepares its kernels before memory profiling into `CACHE_DIR` and writes its
  weight snapshot: the first r6v boot took 396 s to the API, and a boot that writes the snapshots ~184 s here. Later
  boots reuse both; since `docker/0007-*.patch` nothing is profiled for DSpark at boot. Keep `CACHE_DIR` on
  persistent local disk; `WAIT_TIMEOUT` defaults to 3600 s.
- **Weight snapshots use disk.** Each node keeps ~72 GiB (71.86 GiB for its rank) under
  `CACHE_DIR/dsv41/weight-snapshots`, written once per build and configuration. Rebuilding the image or changing
  anything in the snapshot key (engine environment, vLLM configuration, checkpoint files,
  `DSV41_WEIGHT_SNAPSHOT_TAG`) invalidates them: the next boot reads the checkpoint and writes new ones, replacing
  the old (one kept per rank). A rank whose cache filesystem cannot hold the snapshot with 4 GiB to spare skips the
  write, and every rank then keeps loading the checkpoint. `WEIGHT_SNAPSHOT=0` turns snapshots off; delete the
  directory to reclaim the space.
- **KV snapshots use disk and stop time.** With `KV_PERSIST=1` each node keeps ~10.5 GiB (its rank's KV buffer) plus
  the head's table (a few MB) under `CACHE_DIR/dsv41/kv-snapshots`, rewritten at every `stop`; a node that cannot
  hold it with 4 GiB to spare fails the save (logged; the next start is cold). Each save adds the write time of
  10.5 GiB per node to `stop` and each restore its read time to `start`: measured 1.2-5 s per save and 1.9 s per
  restore (slowest rank 6 GB/s). `KV_PERSIST=0` turns it off; delete the directory to reclaim the space.
- **c=8 decode depends on the boot.** Across boots, c=8 aggregate decode came out at one of two levels per cell (code
  321 or 456 tok/s, json 370 or 506), stable within a boot (three repeats within ±0.1%); against r6v that is +8.5%
  to +24% on the mean. The cause is unexplained; it is not the persistent prefix cache (`docker/0008-*.patch`).
  Prefill varied far less across boots (+9.4% to +9.8% against r6v).
- **The memory guard kills, by design.** A guard SIGKILLs its rank when MemAvailable drops below its threshold
  (5 GiB while loading, 3 GiB serving) or when the container can no longer be inspected. TP4 cannot continue on three
  ranks, so a kill stops the service: read `./dsv41.sh logs` and the guard journal (`journalctl -t spark-memguard`),
  then `./dsv41.sh restart`. `start` fails if a guard is gone before the API answers; `stop` stops the guards with
  the ranks. Without passwordless sudo there is no guard, and `start` refuses to run.
- **No `max_tokens` cap.** An omitted `max_tokens` can run to the end of the 1M window; clients must send it.
- **Queueing.** Above `MAX_NUM_SEQS` concurrent requests queue; their TTFT includes the wait. One prefill runs at a
  time.
- **Fragmented Engram shards** slow the row reader; `doctor` warns and prints the `e4defrag` command to run while the
  service is stopped.
- Publisher golden fixtures 3-4 use an internal role the OpenAI API cannot express, so `parity` skips them;
  fixture 5 (images) is covered by the live image test in `api-test`.

## Fallback

**v2.0.0** tags this recipe on r6v with its defaults: image `dsv41-1017-ds41f:r6v`, 8,192 batched tokens, drafter
heads not in NVFP4, no GPU clock setting at `start`, no weight snapshots, no persistent prefix cache. To go back:
`./dsv41.sh stop`, `git checkout v2.0.0`, start from that tag's `.env.example`, run `./dsv41.sh build` if the nodes
no longer hold `dsv41-1017-ds41f:r6v`, then `./dsv41.sh start`. That image uses neither snapshot; delete
`CACHE_DIR/dsv41/weight-snapshots` and `CACHE_DIR/dsv41/kv-snapshots` to free the disk. The current defaults
described above live on the untagged branch `autoresearch/session-20261018`, pending review.

**v1.0.0** is the previous engine of this recipe; its image is `dsv41-1017-spark:v1.0.0` on all four nodes. To go
back: `./dsv41.sh stop`, `git checkout v1.0.0`, start from that tag's own `.env.example` (its keys differ) with
`IMAGE=dsv41-1017-spark:v1.0.0`, then `./dsv41.sh start`. Its README and NOTICE describe and credit that engine.

## Credits

**Engine**

- [spark-ds41f](https://github.com/christopherowen/spark-ds41f) by **Christopher Owen**, Apache-2.0, release r6w at
  `2dc95329`: the engine this recipe serves. His build preparation and image recipe build our image; his patch series
  for vLLM (172 of 173 patches, among them r6w's 0181, the routed SwiGLU and MXFP8 cast over the live tiles), B12X,
  TileLang, TileKernels and NCCL (three of four) carry the TileLang kernel family, reference precision, batch
  invariance, the Engram reader and ring relay, the DSpark verification work, the NVFP4 drafter heads, and kernel
  preparation; his linear DSpark draft budget (r6x's vLLM 0184, rebased as `docker/0007-*.patch`), his lossless MoE
  scale-codes overlay (`a250ace`) and his DSpark profile-replays overlay (`2ce3644`, applied under 0007) are in the
  image too; our defaults start from his `tp4-ring` configuration, and the memory guard is his
  `scripts/memguard*.sh`, run unmodified. His external site receipt showed that `tp4-ring` runs on a stock 4 KiB
  kernel. Also from his series: carlosmolina0615 (vLLM 0045, the CED decoder's MoE routed with its own padding mask
  in TileLang); issues reported by huxiaofengtiger (NVIDIA Developer Forums); a port of vllm-project/vllm#51694.
- [vLLM](https://github.com/vllm-project/vllm) by the vLLM project, Apache-2.0: the engine, via
  [Local Inference Lab's fork](https://github.com/local-inference-lab/vllm) (`integration/karmic-kraken-beta`, the
  DeepSeek-V4.1 integration it is built on, including the `xhigh` effort alias), and the vLLM nightly image as the
  runtime base. `docker/0003-*.patch` and `docker/0004-*.patch` port vllm-project/vllm#60945 by Chauncey
  (chaunceyjiang) and #61161 by DimensionSTP; `docker/0005-*.patch` follows the bounded decoder replay idea of
  #58132 (ivanium) and #59894 (gcanlin), porting no upstream code.
- [B12X](https://github.com/local-inference-lab/b12x) by Luke Alonso and Local Inference Lab contributors,
  Apache-2.0: checkpoint loader, vision tower kernels, the alternative kernel backend, and the RoCEnante module
  sparknet's collectives derive from.
- [TileLang](https://github.com/tile-ai/tilelang) by Tile-AI, MIT (with Apache TVM and CUTLASS as pinned
  submodules): the kernel language of spark-ds41f's kernel family.
- [TileKernels](https://github.com/deepseek-ai/TileKernels) by DeepSeek, MIT: MoE gate, SwiGLU and quantization
  casts, mHC.
- [sparknet / dgx-spark-networking](https://github.com/christopherowen/dgx-spark-networking) by Christopher Owen,
  Apache-2.0: the one-shot RoCE collectives; its NOTICE credits SparkRing
  ([FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring)) for the mesh4 flow-label mechanism.
- [NCCL](https://github.com/NVIDIA/nccl) by NVIDIA, 2.30.7, with **Stanislav Bardyuk**'s AArch64 IB send-path
  acquire fence (NVIDIA/nccl#2393), which prevents a proxy-thread hang; [CUTLASS](https://github.com/NVIDIA/cutlass)
  by NVIDIA for the SM121 extension rebuild. NVIDIA's DGX Spark subnet-aware routing work by Zifu Yang and Xiaofan Li
  (`NCCL_IB_SUBNET_AWARE_ROUTING`).

**Model**

- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek, MIT
  License: the weights (downloaded by `fetch`, not in this repo or the image) and the reference encoder
  `encoding/encoding.py` that `parity` checks the server against.

**Benchmarks, conventions and prior art**

- [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks):
  `scripts/qeval.py`, run as an external client in the qualification battery (not included). MiaAI-Lab's and
  [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)'s
  SGLang recipes are prior art for this model on four Sparks (no code used).
- `needle_salted.py` from spark-ds41f's external site receipt and spark-ds41f's `token_determinism_text.py`
  (`experiments/2026-10-12-r6g`), run as external clients (not included).
- The ring container contract and choosing rails by subnet come from our keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark
  recipe, and knapcio's GLM-5.3-gr8-4x-DGX-Spark-TP4 recipe (MIT) for the 4-node ring conventions.
- [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10): prior art for quick boot from
  per-rank processed-weight snapshots. It carries no license, so only the idea is used: `docker/0002-*.patch` is our
  own implementation, no code taken.

## Licenses

| Component | License | In this repo? |
|---|---|---|
| This repository's scripts, tests, bench client, docs, `docker/0001-*.patch` to `docker/0006-*.patch` and `docker/0008-*.patch` | Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)) | yes |
| `docker/0007-*.patch`: spark-ds41f's vLLM 0184 by Christopher Owen (`03e8775c`), rebased by us | Apache-2.0 | yes |
| spark-ds41f (`2dc95329`): build preparation, image recipe, patch series, memory-guard scripts; the scale-codes (`a250ace`) and profile-replays (`2ce3644`) overlays | Apache-2.0 | no: fetched at image build |
| vLLM: Local Inference Lab fork (`19f2c20`) and the vLLM nightly runtime image | Apache-2.0 | no: fetched / pulled at image build |
| B12X (`236ddff`) | Apache-2.0 | no: fetched at image build |
| TileLang (`b95ee4f`) with its submodules (Apache TVM, Apache-2.0; CUTLASS, BSD-3-Clause) | MIT (its LICENSE notes collaboration terms with Microsoft for 2024-12-01 to 2025-03-14) | no: fetched at image build |
| DeepSeek TileKernels (`66258df`) | MIT | no: fetched at image build |
| sparknet / dgx-spark-networking (`3bc8d10`) | Apache-2.0 | no: fetched at image build |
| NVIDIA NCCL 2.30.7 | Apache-2.0, parts under their original BSD license | no: fetched and rebuilt at image build |
| NVIDIA CUTLASS 4.7.1 headers | BSD-3-Clause | no: fetched at image build |
| `nvidia-cutlass-dsl` 4.7.1 wheels | NVIDIA proprietary license | no: installed hash-locked from PyPI at image build |
| DeepSeek-V4.1-Flash-1017 weights and reference encoder | MIT | no: downloaded by `fetch` and `parity` |

Each patch series changes its upstream under that project's license. The image you build contains all of the
fetched components; check their licenses before you distribute it.
