# Benchmark run

decode: ok | prefill: ok | quality: 49/50 | corruption probe: 9/13 | restart: ok

`autoresearch.sh` run #26 on branch `autoresearch/session-20261018` (`ad34b7d`), the current defaults: spark-ds41f
r6w with the scale-codes overlay, docker/0001-0008, k=6, linear draft budget at row cost 0.01,
`MAX_NUM_BATCHED_TOKENS` 24576, `KV_PERSIST=1`, GPU clocks locked at 2200 MHz (site policy; 2,184-2,197 MHz under
load). Order: timed `./dsv41.sh restart`, a discarded warm-up pass (`warmup-decode/`, `warmup-prefill/`), decode,
prefill, quality gate, corruption probe. `contract.log` is `tests/test-recipe.sh` before the deploy.

## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 | c=8 |
|---|---:|---:|
| prose | 65.1 (66.0) | 386.3 (49.1) |
| code | 93.9 (95.9) | 321.7 (41.5) |
| structured | 84.5 (86.2) | 479.7 (61.7) |
| json | 93.7 (96.2) | 370.2 (47.8) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 | c=8 |
|---|---:|---:|
| prose | 0.127 | 0.192 |
| code | 0.130 | 0.356 |
| structured | 0.142 | 0.257 |
| json | 0.147 | 0.303 |

_decode: ok (2026-10-18T14:26:23-0500)_

The c=8 cells move between boots (other boots gave code 456 and json 506 tok/s), but the engine does not: in four
restarts vLLM's counters showed 405-412 tok/s and 3.15-3.22 tokens per draft in every 30 s c=8 window. The spread is
in the client's window accounting (README, Known limits).

## Prefill

Unique prompts (no prefix-cache hits), max_tokens 1, sequential. Tokens from usage.prompt_tokens.

| target | prompt_tokens | seconds | tok/s |
|---|---:|---:|---:|
| 32k | 32750 | 4.95 | 6622.3 |
| 128k | 131057 | 20.15 | 6505.1 |
| 512k | 524256 | 95.19 | 5507.5 |

_prefill: ok (2026-10-18T14:30:48-0500)_

## Against the r6v record

Same client and settings as [`2026-10-18-r6v`](../2026-10-18-r6v/summary.md).

| measure | r6v | this run | gain |
|---|---:|---:|---:|
| c=1 prose | 61.39 | 65.08 | +6.0% |
| c=1 code | 87.23 | 93.91 | +7.7% |
| c=1 structured | 80.62 | 84.49 | +4.8% |
| c=1 json | 88.53 | 93.75 | +5.9% |
| c=1 mean | | | **+6.09%** |
| prefill 32k | 6012.6 | 6622.3 | +10.1% |
| prefill 128k | 5915.8 | 6505.1 | +10.0% |
| prefill 512k | 5042.0 | 5507.5 | +9.2% |
| prefill mean | | | **+9.78%** |

## Quality gate

50 exact-answer questions, greedy (`bench/quality_gate.py`; answers in `quality_answers.json`): accuracy 49/50
(gate: >= 48).

## Thinking-mode corruption probe

`bench/corruption_probe.py`: corrupt 9/13 offsets, as on r6v (`corruption.log`).

## Restart

`./dsv41.sh restart` (`restart.log`): KV snapshot saved in 1.2 s (642 cached blocks, 11,274,207,232 bytes per rank),
ranks stopped, engine up after 97 s, smoke OK; 121 s wall time including the KV save and restore.
