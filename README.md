# DeepSeek-V4.1-Flash-1017 on four DGX Sparks

Serve [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) in its native
checkpoint (MXFP4 routed experts, FP8 dense) from four NVIDIA DGX Sparks joined in a switchless QSFP ring, behind
an OpenAI-compatible API. The engine is [vLLM](https://github.com/vllm-project/vllm) from DeepSeek's V4.1 image
plus pinned GB10 fixes, run as TP4 + EP4 with DSpark speculative decoding and a 1M-token window.

One script, run from a control machine, does everything:

```bash
cp .env.example .env        # set NODES (management addresses, ring cable order, rank 0 first)
./dsv41.sh doctor           # read-only checks on all four nodes and the ring
./dsv41.sh fetch            # pinned checkpoint on every node, sha256-verified
./dsv41.sh build            # serving image on rank 0, shipped over the ring
./dsv41.sh stop-other       # optional: free the GPUs from the stack that normally owns them
./dsv41.sh start            # http://<rank0>:8000/v1
./dsv41.sh api-test         # full API conformance suite
./dsv41.sh parity           # server prompts == DeepSeek's encoder
```

## Headline numbers

Measured numbers land here after bring-up (see Benchmarks).

## What you need

- **Four DGX Sparks** (GB10, sm_121, 121 GiB unified memory each) with nothing else on their GPUs. `start`
  refuses to run next to any container not in `ALLOWED_CONTAINERS`.
- **A ring of four QSFP cables**: each Spark's QSFP cage cabled to its two neighbours, one IPv4 subnet per cable
  (one address per RoCE port). Ranks 0-2 and 1-3 share no cable ([The ring](#the-ring)).
- **A management LAN** reaching all four nodes (bootstrap, ssh, the API).
- **~480 GiB free NVMe per node** at the same `MODELS_DIR` path. Every node keeps its own copy, and it must be
  local NVMe: disk-backed Engram refuses network filesystems.
- **Passwordless sudo** recommended: `start` drops the page cache while the ranks load. Without it the drops are
  skipped silently.
- **The switchless NCCL bundle** (`nccl-switchless-v0.0.1`) at the same path on every node (`SWITCHLESS_NCCL`);
  `doctor` checks its `SHA256SUMS`.
- **Docker** with the NVIDIA runtime on every node.
- **The Hugging Face CLI** on every node at `HF_BIN`: `fetch` downloads a quarter of the checkpoint on each rank.
- **A control machine** (a Mac works) with bash 4 or newer (`brew install bash`), python3, curl, rsync, and
  key-based ssh to all four nodes as `SSH_USER`; the nodes need key-based ssh to their ring neighbours' cable
  addresses (checkpoint and image copies run node to node).

## Install

```bash
cd DeepSeek-V4.1-Flash-1017-4x-DGX-Spark   # this repository, on the control machine
cp .env.example .env
```

Edit `.env`: `NODES` lists the four management addresses in ring cable order, rank 0 first. Everything else has
a working default ([Configuration](#configuration)). Then:

| Step | What it does |
|---|---|
| `doctor` | read-only, per node: ssh, docker, image ID, GPU processes, available memory, NVMe free, checkpoint shard count, NCCL bundle checksum, sysctls, sudo; then scans and validates the ring and prints the detected rails |
| `fetch` | sizes the pinned files from the Hugging Face API and splits them into four size-balanced parts; each rank downloads its part over its own uplink (a Spark's management port is 1 GbE). Rank 0 gathers the parts over the cables (1→0, 3→0, 2→1→0), fans the full tree back out (0→1, 0→3, 1→2), then every node is sha256-verified against `config/DeepSeek-V4.1-Flash-1017.sha256`. Resumable: files already present at full size are not downloaded again |
| `verify` | the sha256 check alone, all four nodes in parallel |
| `build` | rsyncs `docker/` and `config/` to `BUILD_DIR` on rank 0, builds `IMAGE` there, streams it over the cables with `docker save \| docker load` (0→1 and 0→3, then 1→2), and prints the image ID on every node |

## Run

```bash
./dsv41.sh start        # workers first, waits for /health, then smoke
./dsv41.sh smoke        # one request: 17*23 must come back as 391
./dsv41.sh status       # containers, available memory, /v1/models
./dsv41.sh logs 0 200   # tail one rank's engine log
./dsv41.sh stop         # stops all ranks; each log is kept as ~/$CONTAINER.prev.log
./dsv41.sh restart
./dsv41.sh print-serve 0   # the exact docker run + vllm serve for one rank, no side effects
```

`start` resolves the ring (pins or scan), refuses if foreign containers run, stops any old ranks, drops the page
cache, and starts ranks 3, 2, 1 (headless) and then rank 0. It polls `/health` for up to `WAIT_TIMEOUT` seconds,
dropping the page cache between polls. If any rank's container exits during startup it prints the last 40 log
lines of every rank, stops the rest and fails. When the API answers it runs `smoke`.

`stop-other` runs `STOP_OTHER` on the control machine to take down the stack that normally owns these GPUs;
`restore-other` stops this stack and runs `RESTORE_OTHER`.

Use it like any OpenAI endpoint:

```bash
curl http://192.168.1.121:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "Is 221 prime?"}],
  "reasoning_effort": "high",
  "max_tokens": 4096
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.121:8000/v1", api_key="unused")  # or your API_KEY

# thinking off
r = client.chat.completions.create(model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"thinking": False}})

# thinking on, publisher tier "max"; reasoning arrives in message.reasoning_content
r = client.chat.completions.create(model="deepseek-v4.1-flash", reasoning_effort="max", max_tokens=16384,
    messages=[{"role": "user", "content": "How many primes are below 60?"}])

# tools
weather = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
r = client.chat.completions.create(model="deepseek-v4.1-flash", tools=[weather],
    messages=[{"role": "user", "content": "What's the weather in Rome?"}])
print(r.choices[0].message.tool_calls)

# image input
r = client.chat.completions.create(model="deepseek-v4.1-flash", messages=[{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}])
```

Reasoning counts toward `max_tokens`: leave room for both. The API has no authentication unless `API_KEY` is set;
set one if anything beyond your own machines can reach the port.

## The ring

Four Sparks, four cables, no switch: rank r is cabled to r+1 and r−1 (mod 4). Stock NCCL assumes every rank can
reach every other one over the fabric; on a ring it cannot (ranks 0-2 and 1-3 share no cable). The switchless
NCCL bundle (NCCL 2.30.7 with SparkRing's switchless-cycle patches) is mounted into the container and
`LD_PRELOAD`ed, with `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_SKIP_TREE_CONNECT=1` and `NCCL_ALGO=Ring`.

Rails are chosen by subnet, never by device name. At `start` (and always for `fetch`/`build`) the controller
scans each node for up RoCE netdevs with exactly one IPv4 address, finds the subnet each node shares with its next
neighbour, and checks that ranks 0-2 and 1-3 share none. Each rank's `NCCL_IB_HCA` is the two HCAs facing its
neighbours, with `NCCL_IB_SUBNET_AWARE_ROUTING=1` and the cable prefix length. NCCL bootstrap, Gloo and
`VLLM_HOST_IP` use the management NIC that routes to rank 0.

Without all-to-all connectivity, expert dispatch cannot use point-to-point transfers, so `MOE_MODE=ep` runs
`--enable-expert-parallel --all2all-backend allgather_reducescatter`. `MOE_MODE=tp` keeps the MoE
tensor-parallel (EP1) instead.

To skip the scan at `start`, pin all nine `RING_*` keys in `.env` (`doctor` prints the detected values).
`fetch` and `build` always scan, because they need the cable addresses. The rendezvous port (`MASTER_PORT`) and
the cables are not authenticated: keep them private.

## Configuration

Precedence: environment, then `.env`, then the defaults in `dsv41.sh`. `.env` is parsed, not sourced.

| Key | Default | Meaning |
|---|---|---|
| `NODES` | `192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18` | four management addresses in ring cable order, rank 0 first; rank 0 serves the API |
| `SSH_USER` | `spark` | ssh user on every node |
| `MODELS_DIR` | `/home/spark/models` | checkpoint parent, same absolute path on every node, local NVMe |
| `MODEL_REPO` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | Hugging Face repo |
| `MODEL_REVISION` | `1bae0e1088993b9b82de08f59de5be5a34e42170` | pinned checkpoint revision |
| `MODEL_NAME` | `DeepSeek-V4.1-Flash-1017` | directory under `MODELS_DIR` |
| `CHECKSUMS` | `config/DeepSeek-V4.1-Flash-1017.sha256` | sha256 manifest of the pinned files |
| `HF_BIN` | `/home/spark/venvs/hf/bin/hf` | Hugging Face CLI on the nodes (`fetch` only) |
| `IMAGE` | `dsv41-1017-spark:local` | serving image tag |
| `BUILD_DIR` | `/home/spark/build/dsv41-1017` | build context on rank 0 |
| `SWITCHLESS_NCCL` | `/home/spark/nccl-switchless-v0.0.1` | switchless NCCL bundle, same path on all four |
| `CACHE_DIR` | `/home/spark/.cache/dsv41-1017` | JIT and compile caches kept across boots |
| `CONTAINER` | `dsv41-1017` | container name on every node |
| `MASTER_PORT` | `29511` | torch.distributed rendezvous on rank 0 |
| `SERVED_NAME` | `deepseek-v4.1-flash` | model id clients send |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | API bind address and port |
| `API_KEY` | (empty) | require `Authorization: Bearer <key>` |
| `MAX_MODEL_LEN` | `1048576` | prompt + reply window, tokens |
| `MAX_NUM_SEQS` | `4` | concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | tokens per scheduler step (prefill chunk) |
| `GPU_MEMORY_UTILIZATION` | `0.80` | vLLM memory fraction; ignored when `KV_CACHE_MEMORY_BYTES` is set |
| `KV_CACHE_MEMORY_BYTES` | (empty) | fixed KV budget per rank in bytes; skips memory profiling |
| `MOE_MODE` | `ep` | `ep`: TP4 + EP4; `tp`: MoE tensor-parallel (EP1) |
| `DSPARK` | `1` | DSpark speculative decoding (the drafter ships in the checkpoint) |
| `NUM_SPECULATIVE_TOKENS` | `3` | draft tokens per step |
| `EAGER` | `0` | `1`: no CUDA graphs (first boot, debugging) |
| `COMPILATION_CONFIG` | (empty) | vLLM `--compilation-config` JSON |
| `TEXT_ONLY` | `0` | `1`: `--language-model-only` (drops image input) |
| `PREFIX_CACHING` | `1` | prefix caching |
| `WAIT_TIMEOUT` | `2400` | seconds `start` waits for the API |
| `ALLOWED_CONTAINERS` | (empty; `.env.example`: `mentatd glm53-nfs`) | other containers that may keep running alongside the ranks |
| `STOP_OTHER` | (empty) | shell command `stop-other` runs on the control machine |
| `RESTORE_OTHER` | (empty) | shell command `restore-other` runs after stopping this stack |
| `EXTRA_VLLM_ARGS` | (empty) | appended to `vllm serve` (space separated) |
| `EXTRA_DOCKER_ARGS` | (empty) | appended to `docker run` (space separated) |
| `RING_PREFIX`, `RING_HCA_0..3`, `RING_MGMT_0..3` | (unset) | ring pins: cable prefix length, each rank's two HCAs, each rank's management netdev. All nine or none |

## API behaviour

- **Thinking** is on when the request does not say otherwise (vLLM's V4.1 wrapper). Turn it off with
  `chat_template_kwargs: {"thinking": false}`. Reasoning comes back as `reasoning_content`, separated by the
  `deepseek_v41` reasoning parser, and streams before the answer.
- **`reasoning_effort`** uses the 1017 publisher tiers `low` / `high` / `max` = 50 / 75 / 100, default `high`.
  The other values the OpenAI schema accepts are mapped: `minimal` → `low`, `medium` → `high`, `xhigh` → `max`;
  `none` turns thinking off.
- **Tool calling**: `--tool-call-parser deepseek_v41 --enable-auto-tool-choice`; `tool_choice` `auto`, `none`,
  `required` and named functions; 1017 tool namespaces and tool results with content blocks and images.
- **Structured output**: `response_format` (JSON schema / JSON object) through vLLM's guided decoding.
- **Image input**: `image_url` content blocks; the 1017 encoder places the image tokens. `TEXT_ONLY=1` drops it.
- **Context**: `MAX_MODEL_LEN=1048576` (1M tokens) by default.

## How it is put together

| Piece | What |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017, native checkpoint (MXFP4 routed experts, FP8 dense), revision `1bae0e1`, every file pinned by sha256 in `config/` |
| Image base | `vllm/vllm-openai:deepseekv41-flash-0909`, arm64 manifest pinned by digest (`sha256:d84a1232…`) |
| Upstream fixes | vcruz305's DeepSeek-V4.1 recipe at `e120f52`, cloned at build and applied unmodified by `docker/apply-upstream.sh`: disk-backed Engram overlay, SM120 sparse-MLA fixes 01-08, DSpark `fixswa.py`, Engram replay keying, decode hoist, cold I/O, draft EP filter, SM12x persistent top-k. Every step fails closed on a missing anchor, then the patched modules are compiled and imported |
| Not applied | `tools/dspark/dq.py` and `mkdq8.py` (draft-quant override and FP8 draft dequant for EXL3 packs, whose config hides the drafter's native format; `mkdq8`'s anchor is not in the stock base), and all of vllm-exl3 / ExLlamaV3: this image serves the native checkpoint |
| 1017 encoder | `docker/encoding-1017.py` replaces the base image's pre-1017 `deepseek_v41_encoding.py` with DeepSeek's `encoding/encoding.py` from the pinned revision (sha256-checked). 1017 changed effort tiers to low/high/max = 50/75/100, default 75 (was 25/50/75/100 with xhigh, default 50), added tool namespaces, tool results carrying content blocks and images, tool-result-only user turns no longer starting a new turn, and image-placeholder validation for tool messages. vLLM's wrapper is adapted to pass content blocks and map the OpenAI effort aliases. The build fails unless the publisher's golden fixtures encode byte for byte |
| Engram | disk-backed (`--engram-config '{"cpu_offload":false,"disk_backed":true}'`, `VLLM_ENGRAM_DISK_BACKED=1`), read from the local checkpoint |
| Drafting | DSpark, the drafter shipped in the checkpoint: `num_speculative_tokens` 3, probabilistic draft sampling, block rejection sampling |
| Split | TP4 across the four Sparks, EP4 for routed experts (`--enable-ep-weight-filter`), `allgather_reducescatter` dispatch; rank 0 serves the API, ranks 1-3 headless |
| Fabric | switchless NCCL over RoCE on the ring; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`; `HF_HUB_OFFLINE=1` at serve time |
| Controller | `dsv41.sh`: bash over ssh from the control machine; `print-serve` shows each rank's exact command |

## Benchmarks

Harness and records: [`bench/`](bench/).

Results land here after bring-up.

## Tests

| Test | Needs | What it guards |
|---|---|---|
| `tests/test-recipe.sh` | nothing (no GPU, ssh or network) | generates every rank's `docker run` + `vllm serve` with pinned ring values and checks the invariants the deployment depends on; shell and python syntax |
| `tests/api_conformance.py` (`./dsv41.sh api-test`) | live API | endpoints, sampling controls, thinking/reasoning, every effort tier, tool calling, structured output, image input, long context, auth and error handling; one PASS/FAIL line per case |
| `tests/test_encoding_parity.py` (`./dsv41.sh parity`) | live API, network | the server's rendered prompt equals DeepSeek's `encoding.py` at the pinned revision: golden fixtures, thinking on/off and default, every effort tier and OpenAI alias, namespaced tools (image fixtures are covered by `api_conformance` instead) |

## Known limits

Filled after bring-up.

## Credits

**Model**

- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek, MIT
  License: the weights (downloaded by `fetch`, not in this repo) and the reference encoder `encoding/encoding.py`
  (installed into the image at build).

**Engine and runtime fixes**

- [vLLM](https://github.com/vllm-project/vllm) by the vLLM project, Apache-2.0: the engine, via DeepSeek's V4.1
  image `vllm/vllm-openai:deepseekv41-flash-0909`.
- [vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe)
  by vcruz305 (Victor Cruz) and contributors, AGPL-3.0-only, commit `e120f52`: fetched at build and applied
  unmodified. Disk-backed Engram overlay by @Blackwellboy (PR #2, hardened by vcruz305); SM120 sparse-MLA fixes
  01-08, DSpark fixswa, and the Engram fixes (replay keying, FULL_DECODE_ONLY hoist, cold I/O, draft EP filter,
  SM12x persistent top-k) by @fattchris. Not used: `dq.py`, `mkdq8.py`, vllm-exl3, ExLlamaV3 (EXL3-only).

**Fabric**

- Switchless NCCL bundle `nccl-switchless-v0.0.1`, mounted and `LD_PRELOAD`ed, not redistributed:
  NVIDIA NCCL 2.30.7 (Apache-2.0 / BSD-3-Clause); switchless-cycle patches by SparkRing
  ([FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring), Apache-2.0); hardening and packaging by
  Alex Ellis, OpenFaaS Ltd (Apache-2.0). Joseph Rose, for the earlier skip-Tree/skip-PAT approach (no code used).
  NVIDIA's DGX Spark subnet-aware routing work by Zifu Yang and Xiaofan Li.
- The ring container contract and choosing rails by subnet come from our keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark
  recipe, and knapcio's GLM-5.3-gr8-4x-DGX-Spark-TP4 recipe (MIT) for the 4-node ring conventions.

**Prior art and inspiration** (no code included)

- [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) and
  [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)
  (SGLang, AGPL): GB10 memory findings (NCCL buffers, allocator), the publisher reasoning-effort table, and the
  loop-abort and `max_tokens`-cap ideas. Their measured TP4 numbers are a comparison point.

**Future work**

- [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw):
  its routed experts graft onto vcruz305's loader contract (proven tensor-compatible), a path to EXL3 packs and
  TP2. Not used here.

## Licenses

| Component | License | In this repo? |
|---|---|---|
| This repository's scripts, tests and docs | Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)) | yes |
| DeepSeek-V4.1-Flash-1017 weights | MIT | no: downloaded by `fetch` |
| DeepSeek `encoding/encoding.py` | MIT | no: downloaded at image build |
| vLLM and the `deepseekv41-flash-0909` image | Apache-2.0 | no: pulled as the build base |
| vcruz305 recipe fixes (`e120f52`) | AGPL-3.0-only | no: cloned at image build, applied unmodified, not vendored |
| NVIDIA NCCL 2.30.7 | Apache-2.0 / BSD-3-Clause | no: mounted at run time |
| SparkRing switchless patches; OpenFaaS hardening and packaging | Apache-2.0 | no: part of the mounted bundle |

The image you build contains the AGPL-3.0-only fixes; if you distribute that image, AGPL-3.0 applies to those
files.
