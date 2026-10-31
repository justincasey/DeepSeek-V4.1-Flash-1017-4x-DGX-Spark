# Benchmark run: FABRIC=switched

decode: ok | prefill: ok | quality: 49/50 | corruption probe: 9/13 | token oracle: 6/6 | parity: OK | api-test: OK

The current defaults (spark-ds41f r6w with the scale-codes overlay, docker/0001-0008, k=6, linear draft budget at
row cost 0.01, `MAX_NUM_BATCHED_TOKENS` 24576, `KV_PERSIST=1`, GPU clocks locked at 2200 MHz), image
`sha256:24c3f51f59b2…`, unchanged, on the four Sparks behind a MikroTik CRS804-4DDQ instead of the cable ring:
`FABRIC=switched` (sparknet profile `switched`, transport `oneshot-switched`, `ROCE_TC=106`), one 200G cable per
Spark (rails `10.100.10.0/24` and `10.100.11.0/24`), PFC and ECN on the switch's traffic class 3 (DSCP 26), every
node trusting DSCP with PFC on priority 3.

Order: `./dsv41.sh start` (engine up after 104 s; weight snapshots and the prefix KV snapshot restored), a discarded
warm-up pass (`warmup-decode/`, `warmup-prefill/`), decode, prefill, quality gate, corruption probe, `bench check`,
`bench spec`, `parity`, `api-test`, all from the control machine on one boot, 2026-10-31.

## Against the ring (2026-10-18-autoresearch, the same defaults on the cable ring)

| measure | ring | switched | change |
|---|---:|---:|---:|
| c=1 prose | 65.08 | 67.1 | +3.1% |
| c=1 code | 93.91 | 97.7 | +4.0% |
| c=1 structured | 84.49 | 87.6 | +3.7% |
| c=1 json | 93.75 | 97.2 | +3.7% |
| c=1 mean | | | **+3.6%** |
| c=8 prose / code / structured / json | 386.3 / 321.7 / 479.7 / 370.2 | 347.3 / 335.2 / 418.3 / 522.6 | boot-bimodal (TD-010) |
| prefill 32k | 6622.3 | 6483.7 | -2.1% |
| prefill 128k | 6505.1 | 6478.0 | -0.4% |
| prefill 512k | 5507.5 | 5489.4 | -0.3% |

One boot each; the ring's numbers are its autoresearch run #26. DSpark acceptance (`spec.log`): 1.791 accepted
tokens per draft over the four prompt types, 2.79 tokens per step.

## Fabric during the run (`fabric.log`)

Switch counters reset just before the run; node counters read before and after.

- Switch: 337-338 million packets per Spark port in tx queue 3 (the lossless RoCE class), ~8,800 in queue 1;
  **0 tail drops** in any queue; 12 / 2 PFC pauses sent towards ranks 0 / 2, 6 / 8 received from ranks 1 / 3.
- Nodes, both rails: **0 new `out_of_sequence` and `packet_seq_err`** (the RoCE retransmit causes), 0
  `local_ack_timeout_err`; ECN-marked RoCE packets received went up by 31,514 / 33,393 / 39,193 / 70,517 on ranks
  0-3 and the reaction points handled 2.5-4k CNPs each (DCQCN at work, no loss).
