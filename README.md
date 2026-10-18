# DeepSeek-V4.1-Flash-1017 on four DGX Sparks

Serve [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) in its native
checkpoint (MXFP4 routed experts, FP8 dense) from four NVIDIA DGX Sparks joined in a switchless QSFP ring, behind
an OpenAI-compatible API, with a 1,048,576-token window and 16 concurrent sequences.

The engine is Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f) release **r6v**
(Apache-2.0): Local Inference Lab's vLLM fork with his patch series, the TileLang SM121 kernel family, DeepSeek's
TileKernels, sparknet one-shot RoCE collectives and a patched NCCL. `docker/Dockerfile` rebuilds his image from the
pinned upstreams (spark-ds41f at `cbf47749`) and adds one patch of ours; `dsv41.sh` serves his `tp4-ring`
configuration through our controller, on the stock DGX OS 4 KiB-page kernel.

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

This engine has not been qualified on this ring yet. Its column below is filled at qualification, with the same
harnesses that measured the v1.0.0 baseline beside it ([Benchmarks](#benchmarks)).

| Measurement | v1.0.0 baseline (measured 2026-10-17) | r6v engine: measured at qualification |
|---|---|---|
| Decode, one request, prose / code (`bench decode`) | 35.8 / 55.1 tok/s | |
| Decode, 4 requests aggregate, prose / code | 89.9 / 132.4 tok/s | |
| Prefill, unique text, 128K / 1M tokens (`bench prefill`) | 1,724 / 968 tok/s | |
| Needle at 1,000,006 prompt tokens (`bench needle`) | PASS, 985 s | |
| Salted needle at 547,579 prompt tokens | PASS, 1,157 tok/s | |
| qeval, 75 tasks, greedy | 71/75 | |
| Quality gate, 50 exact-answer questions, greedy | 49/50 | |
| API conformance (`api-test`) | 25/26 | |
| Encoding parity (`parity`) | 15/15 | |

Pending: every r6v cell above, plus boot time, weights per rank and the KV pool in tokens; they land in
`bench/records/` with the qualification run.

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
| `doctor` | read-only. Ring: scans every node, checks pins against the live ring, pings each next rank across its cable, warns on MTU ≠ 9000. Per node: docker, image ID equal to rank 0's and built from spark-ds41f `cbf47749` (image label), GPU processes, MemAvailable, NVMe and `/` free space, kernel and page size, sysctls, `CPUSET` equal to the node's fast cores, `DRI_CARD` present, passwordless sudo, `systemd-run`, checkpoint shard count, Engram shard filesystem and fragmentation (`e4defrag -c`), memory guard state |
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
./dsv41.sh stop         # stops the guards and all ranks; each log is kept as ~/$CONTAINER.prev.log
./dsv41.sh restart
./dsv41.sh print-serve 0   # the exact guard arm + docker run + vllm serve for one rank, no side effects
```

`start` builds every rank's command first, so a configuration error starts nothing. It re-scans the ring and
refuses stale `RING_*` pins, refuses if foreign containers run, stops any old ranks, drops the page cache, and
installs the memory guard on every node. Then, for ranks 3, 2, 1 (headless) and finally rank 0, it arms that node's
startup guard and runs the container. It polls `/health` for up to `WAIT_TIMEOUT` seconds, dropping the page cache
between polls; if any rank's container exits or loses its guard it prints the last 40 log lines and the guard's
journal for every rank, stops the rest and fails. When the API answers it switches every node to the steady guard
and runs `smoke`. The first boot of a new image prepares the TileLang kernels into `CACHE_DIR` and takes longer
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
- **`max_tokens`**: there is no server-side cap. A request that omits it may generate until the context is full,
  so always send it. `prompt_tokens + max_tokens` must not exceed 1,048,576 (`MAX_MODEL_LEN`), or vLLM answers 400.
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
comment only at the start of a value or after whitespace. Defaults are spark-ds41f r6v at shape `tp4-ring`
(`config/cluster.json` at `cbf47749`). [`.env.example`](.env.example) documents every key; the groups:

| Group | Keys | Defaults and notes |
|---|---|---|
| The Sparks | `NODES`, `SSH_USER` | four management addresses in ring cable order, rank 0 first (rank 0 serves the API); `spark` |
| Checkpoint | `MODELS_DIR`, `MODEL_REPO`, `MODEL_REVISION`, `MODEL_NAME`, `CHECKSUMS`, `HF_BIN` | `/home/spark/models`, same absolute path on every node (Engram shards 47/48 must be local files); revision `1bae0e1` pinned with a sha256 manifest; mounted read-only at `/models` |
| Image and fabric | `IMAGE`, `BUILD_DIR`, `CACHE_DIR`, `CONTAINER`, `MASTER_PORT` | `dsv41-1017-ds41f:r6v`; `CACHE_DIR` is mounted at `/cache` on every node and keeps the JIT/TileLang/Triton caches and DSpark cost curves across boots; `CONTAINER` also names the guard unit (`$CONTAINER-memguard.service`); rendezvous on port 9999 |
| Serving | `SERVED_NAME`, `HOST`, `PORT`, `API_KEY` | `deepseek-v4.1-flash` on `0.0.0.0:8000`; `API_KEY` empty = no authentication |
| Sizing | `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`, `KV_CACHE_MEMORY_BYTES` | 1,048,576 tokens, 16 sequences, 8,192 tokens per step (the long-prefill threshold is this minus the decode rows); a fixed 10.5 GiB KV budget per rank (`11274289152`; empty = vLLM profiling); utilization 0.82 is always passed and then only gates vLLM's startup free-memory check |
| Engine | `PREFIX_CACHING`, `PREFIX_CACHE_RETENTION_INTERVAL`, `KERNEL_BACKEND`, `GENERATION_DEFAULTS`, `TEXT_ONLY`, `MAX_IMAGES`, `WAIT_TIMEOUT` | prefix caching on with retention interval 8192; `tilelang` (r6v) or `b12x` (spark-ds41f's pre-r6 backend, in the same image); `{"temperature":1.0,"top_p":1.0}`; images on, 4 per prompt; `start` waits up to 3600 s |
| DSpark and CUDA graphs | `DSPARK`, `NUM_SPECULATIVE_TOKENS`, `DRAFT_SAMPLE_METHOD`, `ADAPTIVE_VERIFICATION_COST_SCALE`, `DSPARK_COST_DIR`, `EAGER`, `COMPILATION_CONFIG` | DSpark on (the drafter ships in the checkpoint), k=5, greedy drafts, block verification, adaptive verification at cost scale 2.0 (empty = off); cost curves in one directory per build (`off` or a `/cache/...` path to override); `FULL_AND_PIECEWISE` graphs on spark-ds41f's capture ladder up to `MAX_NUM_SEQS` × (k+1) rows (96 at the defaults); `EAGER=1` disables graphs; `COMPILATION_CONFIG` replaces the compilation JSON verbatim |
| Host resources | `CPUSET`, `CONTAINER_MEMORY`, `DRI_CARD`, `DISPLAY_CARVEOUT` | ranks pinned to GB10's ten Cortex-X925 cores `5-9,15-19` (`doctor` checks them per node); `--memory`/`--memory-swap` 112g; the nvidia DRM card bound at `/dev/dri/card0`; embedding and head in the display carve-out off (`1` needs `DRI_CARD` and `nvidia-drm` modeset). Empty disables each |
| Memory guard | `STARTUP_MEMGUARD_GIB`, `STARTUP_MEMGUARD_INTERVAL`, `STARTUP_MEMGUARD_WAIT`, `MEMGUARD_GIB`, `MEMGUARD_INTERVAL` | 5 GiB sampled every 0.25 s while loading (waits up to 120 s for its container); 3 GiB every 2 s once the API answers. The startup threshold must exceed the steady one ([Known limits](#known-limits)) |
| Other stacks | `ALLOWED_CONTAINERS`, `STOP_OTHER`, `RESTORE_OTHER` | containers that may keep running beside the ranks (`.env.example`: `mentatd glm53-nfs`); shell commands for `stop-other` / `restore-other` |
| Advanced | `EXTRA_VLLM_ARGS`, `EXTRA_DOCKER_ARGS` | appended to `vllm serve` / to `docker run` before the image (`-e K=V` overrides an engine variable) |
| Ring pins | `RING_PREFIX`, `RING_HCA_PREV_0..3`, `RING_HCA_NEXT_0..3`, `RING_GID_0..3`, `RING_MGMT_0..3` | all seventeen or none ([The ring](#the-ring)); `.env.example` carries this site's values |

`./dsv41.sh print-serve RANK` prints the exact command a rank runs with your configuration.

## How it is put together

| Piece | What |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017, native checkpoint (MXFP4 routed experts, FP8 dense, BF16 drafter head), revision `1bae0e1`, every file pinned by sha256 in `config/`; every node keeps a full local copy |
| Engine | spark-ds41f r6v by Christopher Owen: Local Inference Lab's vLLM `integration/karmic-kraken-beta` at `19f2c20` + his r6v vLLM series (172 patches); B12X `236ddff` + 11 patches (checkpoint loader, vision tower kernels, the alternative kernel backend); TileLang 0.1.15 `b95ee4f` + 4 patches (among them SM12x block-scaled MMA); DeepSeek TileKernels 2.0.0 `66258df` + 1 patch; sparknet 0.3.0 `3bc8d10`; NCCL 2.30.7 rebuilt for SM121 + 4 patches (the IB send-path fence NVIDIA/nccl#2393, bidirectional switchless rings, balanced channels, adaptive small-ring threads); runtime base the vLLM nightly image `af1c014` (CUDA 13.0.2), pinned by digest |
| Our patch | `docker/0001-*.patch`: `reasoning_effort` `minimal` → 50 and `medium` → 75, which the fork rejected with a 400 |
| Kernels | the TileLang SM121 family (sparse MLA, MXFP4 indexer, FP8 and BF16 projections, FP8 × FP4 routed experts, vocabulary heads, RoPE, compressor, KV writers, Engram gate/hash/gather, WO), DeepSeek's TileKernels for the MoE gate, SwiGLU/quant casts and mHC, L2 weight prefetch during decode (`KERNEL_BACKEND=tilelang`) |
| Precision | spark-ds41f follows DeepSeek's reference `inference/model.py`: FP32 logits from an exact packed BF16 head, FP32 mHC projection, shared-expert SwiGLU in FP32, MoE partials rounded once per rank, WO-A input through `act_quant`; batch-invariant reductions and rank-ordered collectives (his description; [Credits](#credits)) |
| Engram | tables read from local disk by a native io_uring row reader (`--engram-config` `table_memory: disk`), projections sharded over TP, sequence-parallel row exchange relayed on the ring |
| Drafting | DSpark (the drafter in the checkpoint): k=5, greedy drafts, block verification, adaptive verification priced from cost curves rank 0 profiles once per build and keeps in `/cache` |
| Graphs | `FULL_AND_PIECEWISE` CUDA graphs captured at 1, 2, 3, 4, 6, 8, 12 … 32, then every 8th row up to 96; kernels prepared before memory profiling, `--jit-monitor-mode warn` logs any JIT after ready |
| Split | TP4 across the four Sparks (attention, dense and experts), vLLM's multiprocess executor over `--nnodes 4`; rank 0 serves the API, ranks 1-3 headless; one prefill at a time |
| KV | fixed 10.5 GiB per rank |
| Fabric | sparknet one-shot collectives up to 2 MiB, patched NCCL ring above that, both PCIe functions per cable ([The ring](#the-ring)) |
| Host | per rank: the ten fast cores, a 112 GiB cgroup limit, `seccomp=unconfined` (Docker's default profile blocks io_uring), `--ipc=host`, unlimited memlock; spark-ds41f's fail-closed memory guard as a systemd unit per node |
| Controller | `dsv41.sh`: bash over ssh from the control machine; `print-serve` shows each rank's exact commands |

### The image

`docker build -f docker/Dockerfile` (linux/arm64) reads only `docker/`. Its `fetch` stage clones spark-ds41f at
`cbf47749` and runs **his own** build-preparation code (`bin/spark build prepare`) against the r6v lock: each upstream
is fetched at its pinned commit and its patch series applied unmodified with `git am`; the patch-set fingerprint,
patch head, tree and TileLang's submodule commits must equal his manifest, and the Dockerfile's pins must equal his
lock. The build stages after `fetch` are his `docker/Dockerfile` at that commit with his build-time checks kept
(B12X imports and liburing, NCCL version and sha256, TileLang/TileKernels wheel records, sparknet proxy ABI). Ours on
top: the effort patch, a TileLang SM12x block-scaled MMA check, his memory-guard and smoke scripts under
`/opt/spark-ds41f`, and image labels that `doctor` checks. Any mismatch fails the build. The image contains no
weights. spark-ds41f's external 4 KiB site reports its build at 630 s with MemAvailable never below 73.7 GiB.

## Benchmarks

`./dsv41.sh bench STEP` runs [`bench/bench.py`](bench/bench.py) (stdlib only) from the control machine: `decode`,
`prefill`, `needle`, `spec`, `pin`, `check` or `all`. Runs land in `bench/runs/<stamp>-<step>/` (gitignored);
kept runs are copied unchanged into `bench/records/` ([how](bench/records/README.md)). `pin` writes a token-id
regression oracle to `bench/reference.json`; `check` replays it. Decode cells: greedy, thinking off, `max_tokens`
512, c workers issuing back-to-back streaming requests for a 30 s window; aggregate tok/s = sum of
`usage.completion_tokens` / cell wall time (DSpark emits several tokens per chunk, so chunks are never counted).

The qualification battery adds two external clients, not shipped here: MiaAI-Lab's `qeval.py` (75 tasks, greedy)
and the salted cold needle (`needle_salted.py`) from spark-ds41f's external site receipt, so results compare with
that receipt.

### v1.0.0 baseline (2026-10-17)

The previous engine of this recipe ([v1.0.0](#fallback-to-v100)), measured on this ring with its defaults: 1M
context, 4 sequences, DSpark k=3, checkpoint `1bae0e1`. Full record:
[`bench/records/2026-10-17-v1.0.0`](bench/records/2026-10-17-v1.0.0/summary.md). The r6v engine is measured with
the same harnesses at qualification; its record goes next to this one.

| Prompt | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| prose | 35.8 | 60.4 | 89.9 | 85.5 |
| code | 55.1 | 88.2 | 132.4 | 139.1 |
| structured | 48.8 | 77.4 | 116.6 | 112.3 |
| json | 48.0 | 78.4 | 115.7 | 115.0 |

Aggregate decode tok/s. At c=8 half the requests queued (4 sequences), so its TTFT was 14-24 s; c=1 TTFT was
0.16-0.20 s. DSpark accepted 1.05 (prose) to 2.18 (code) extra tokens per draft step.

| Prompt tokens | 4K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill tok/s | 2,138 | 1,817 | 1,902 | 1,810 | 1,724 | 1,473 | 1,282 | 968 |

| Check | v1.0.0 result |
|---|---|
| Needle, 1,000,006 prompt tokens (`bench needle`) | PASS, 985 s |
| Salted cold needle, 547,579 prompt tokens | PASS, 473 s (1,157 tok/s) |
| qeval, 75 tasks | 71/75 (code 24/25, format 7/7, json 6/8, math 13/14, prose 5/5, reason 16/16) |
| Quality gate, 50 exact-answer questions, greedy | 49/50 (CUDA graphs and eager alike) |
| `api-test` | 25/26 (`greedy_is_deterministic` failed on a near-tie) |
| `parity` | 15/15 |
| `bench check` | 3/6 prompts token-identical to the pinned oracle (near-tie divergence) |

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
| `tests/test-recipe.sh` | nothing (no GPU, ssh or network) | generates every rank's guard arm + `docker run` + `vllm serve` with pinned ring values and checks the invariants the deployment depends on (spark-ds41f r6v `tp4-ring`); shell and python syntax |
| `tests/api_conformance.py` (`./dsv41.sh api-test`) | live API | 26 cases: endpoints, sampling controls, thinking/reasoning, every effort tier, tool calling, structured output, image input, long context, auth and error handling; one PASS/FAIL line per case |
| `tests/test_encoding_parity.py` (`./dsv41.sh parity`) | live API, network | the server's rendered prompt equals DeepSeek's `encoding.py` at the pinned revision, token for token: golden fixtures, thinking on/off and default, every effort tier and alias, namespaced tools, the tool-loop turn rule |
| `bench check` | live API | token-id regression against `bench/reference.json` |

## Known limits

- **Image token in text → 400.** A message whose text contains DeepSeek's literal image token
  (`<｜deepseek_image｜>`) is rejected: "Images should be provided as image content blocks". This is DeepSeek's
  encoder validation, applied to every role (spark-ds41f's vLLM 0178 follows it). Agents that paste DeepSeek source
  code or docs into a prompt or tool result hit it.
- **First boot is slow.** Each rank prepares its kernels before memory profiling (about 188 TileLang calls per rank
  on spark-ds41f's external 4 KiB site) into `CACHE_DIR`; DSpark also profiles its cost curves once per build. Later
  boots reuse both. Keep `CACHE_DIR` on persistent local disk; `WAIT_TIMEOUT` defaults to 3600 s for that first boot.
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

## Fallback to v1.0.0

The previous engine of this recipe is tagged `v1.0.0` in git; its image is `dsv41-1017-spark:v1.0.0` on all four
nodes. To go back: `./dsv41.sh stop`, `git checkout v1.0.0`, start from that tag's own `.env.example` (its keys
differ) with `IMAGE=dsv41-1017-spark:v1.0.0`, then `./dsv41.sh start`. Its README and NOTICE describe and credit
that engine.

## Credits

**Engine**

- [spark-ds41f](https://github.com/christopherowen/spark-ds41f) by **Christopher Owen**, Apache-2.0, release r6v at
  `cbf47749`: the engine this recipe serves. His build preparation and image recipe build our image; his patch series
  for vLLM (171 of 172 patches), B12X, TileLang, TileKernels and NCCL (three of four) carry the TileLang kernel
  family, reference precision, batch invariance, the Engram reader and ring relay, the DSpark cost curves and
  verification work, and kernel preparation; our defaults are his `tp4-ring` configuration, and the memory guard is
  his `scripts/memguard*.sh`, run unmodified. His external site receipt showed that `tp4-ring` runs on a stock 4 KiB
  kernel. Also from his series: carlosmolina0615 (vLLM 0045, the CED decoder's MoE routed with its own padding mask
  in TileLang); issues reported by huxiaofengtiger (NVIDIA Developer Forums); a port of vllm-project/vllm#51694.
- [vLLM](https://github.com/vllm-project/vllm) by the vLLM project, Apache-2.0: the engine, via
  [Local Inference Lab's fork](https://github.com/local-inference-lab/vllm) (`integration/karmic-kraken-beta`, the
  DeepSeek-V4.1 integration it is built on, including the `xhigh` effort alias), and the vLLM nightly image as the
  runtime base.
- [B12X](https://github.com/local-inference-lab/b12x) by Luke Alonso and Local Inference Lab contributors,
  Apache-2.0: checkpoint loader, vision tower kernels, the alternative kernel backend, and the RoCEnante module
  sparknet's collectives derive from.
- [TileLang](https://github.com/tile-ai/tilelang) by Tile-AI, MIT (with Apache TVM and CUTLASS as pinned
  submodules): the kernel language of the r6v kernel family.
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

**Benchmarks and conventions**

- [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks):
  `scripts/qeval.py`, run as an external client in the qualification battery (not included). MiaAI-Lab's and
  [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)'s
  SGLang recipes are prior art for this model on four Sparks (no code used).
- `needle_salted.py` from spark-ds41f's external site receipt, run as an external client (not included).
- The ring container contract and choosing rails by subnet come from our keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark
  recipe, and knapcio's GLM-5.3-gr8-4x-DGX-Spark-TP4 recipe (MIT) for the 4-node ring conventions.

## Licenses

| Component | License | In this repo? |
|---|---|---|
| This repository's scripts, tests, bench client, docs and `docker/0001-*.patch` | Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)) | yes |
| spark-ds41f (`cbf47749`): build preparation, image recipe, patch series, memory-guard scripts | Apache-2.0 | no: fetched at image build |
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
