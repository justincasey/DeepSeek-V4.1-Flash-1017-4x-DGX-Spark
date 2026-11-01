# Benchmark run: FABRIC=switched with the direct exchange

decode: ok | prefill: ok | quality: 49/50 | token oracle: 6/6 | long-prompt oracle: 3/3 | corruption probe: 9/13 |
switch drops: 0 | 1M needle: PASS

Image `dsv41-1017-ds41f:r6w-direct`: the current defaults (spark-ds41f r6w + scale codes, docker/0001-0008) plus
`docker/0009-deepseek-v41-direct-exchange-on-switched-fabric.patch` (vLLM head `ded5689`, tree `21153d2`): at four
ranks the rank-order reduce-scatter's chunk exchange and PyNCCL `all_to_all_rows` relay the opposite rank's rows
through a neighbour only when `SPARKNET_ROCE_TOPOLOGY=ring4`; on the switch every rank exchanges with every other
directly, in one group. sparknet's `switched` profile unchanged (8 NCCL channels, 1 MiB buffers, one-shot all-gathers
to 4 MiB, both rails). CRS804 with PFC/ECN on class 3, `ROCE_TC=106`.

`autoresearch.sh` session 2026-10-31, run #40 (this directory) against run #39, the relayed `:r6w` build on the same
harness (`relayed-baseline/`). Order: rebuild and ship, timed restart, discarded warm-up, decode c=1 (four prompt
types, 30 s windows, greedy, thinking off), prefill 32K/128K/512K (unique prompts) with the switch's port tx counters
read around it, quality gate, `bench check`, `bench/long_oracle.py check`, corruption probe. Then, on the deployed
build: `bench needle` (`needle/`), `parity` and `api-test` (both OK).

| measure | relayed (run #39) | direct (run #40) | change |
|---|---:|---:|---:|
| decode c=1 prose / code / structured / json (tok/s) | 66.4 / 96.5 / 86.7 / 96.2 | 66.0 / 96.0 / 86.1 / 95.6 | -0.6% (noise; this path is prefill-only) |
| prefill 32K (tok/s) | 6,493 | 6,717 | +3.4% |
| prefill 128K | 6,477 | 6,555 | +1.2% |
| prefill 512K | 5,490 | 5,552 | +1.1% |
| bytes the four switch ports sent during the prefill bench | 2,072 GB | 1,757 GB | -15.2% |
| switch drops / PFC pauses during it | 0 / 49 | 0 / 72 | |
| token oracle / long-prompt oracle (ids + top-5 logprobs) | 6/6, 3/3 | 6/6, 3/3 | bit-identical |
| 1M needle | | PASS, 999,996 tokens in 215.9 s (4,632 tok/s) | ring record: 1,039,976 in 229 s (4,541 tok/s) |

Against the ring record (2026-10-18-autoresearch, the same engine settings on the cable ring): decode c=1 +1.9%,
prefill +1.0% (32K +1.4%, 128K +0.8%, 512K +0.8%).

Rejected in the same session (all bit-identical, a restart each): `NCCL_BUFFSIZE` 4 MiB, `NCCL_P2P_NET_CHUNKSIZE`
512 KiB, `NCCL_MAX_NCHANNELS` 16 (did not reproduce), `NCCL_IB_MERGE_NICS=0`, sparknet over one rail, one-shot
all-gathers to 2 or 8 MiB. NCCL INFO logs: every prefill all-gather (5-250 MB) already runs Ring/Simple on 8 channels.
