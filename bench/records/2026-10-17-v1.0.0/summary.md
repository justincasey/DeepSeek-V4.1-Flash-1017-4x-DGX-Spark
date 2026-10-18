# Benchmark run

decode: ok | prefill: ok | needle: ok | spec: ok | check: FAILED

## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| prose | 35.8 (36.7) | 60.4 (30.8) | 89.9 (23.2) | 85.5 (21.8) |
| code | 55.1 (56.9) | 88.2 (44.6) | 132.4 (35.9) | 139.1 (35.9) |
| structured | 48.8 (49.6) | 77.4 (39.0) | 116.6 (29.7) | 112.3 (28.4) |
| json | 48.0 (50.1) | 78.4 (39.9) | 115.7 (30.0) | 115.0 (29.5) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| prose | 0.169 | 0.226 | 0.316 | 23.582 |
| code | 0.186 | 0.264 | 0.328 | 14.435 |
| structured | 0.198 | 0.269 | 0.273 | 17.994 |
| json | 0.164 | 0.268 | 0.378 | 17.272 |

_decode: ok (2026-10-17T20:25:28-0500)_

## Prefill

Unique prompts (no prefix-cache hits), max_tokens 1, sequential. Tokens from usage.prompt_tokens.

| target | prompt_tokens | seconds | tok/s |
|---|---:|---:|---:|
| 4k | 4087 | 1.91 | 2137.7 |
| 16k | 16367 | 9.01 | 1817.2 |
| 32k | 32753 | 17.22 | 1902.3 |
| 64k | 65518 | 36.19 | 1810.2 |
| 128k | 131054 | 76.02 | 1724.0 |
| 256k | 262133 | 178.02 | 1472.5 |
| 512k | 524260 | 408.86 | 1282.2 |
| 1M | 1048079 | 1082.59 | 968.1 |

_prefill: ok (2026-10-17T20:36:57-0500)_

## Needle

| result | target | prompt_tokens | seconds | expected | answer |
|---|---:|---:|---:|---|---|
| PASS | 1000000 | 1000006 | 985.019 | 23459 | 23459 |

_needle: ok (2026-10-17T21:07:12-0500)_

## Speculative decoding (DSpark) acceptance

c=1, thinking off, greedy. accepted/draft = delta(vllm:spec_decode_num_accepted_tokens_total) / delta(vllm:spec_decode_num_drafts_total); tokens per step = 1 + accepted/draft.

| prompt | tok/s | accepted/draft | tokens/step | draft-token acceptance |
|---|---:|---:|---:|---:|
| prose | 36.0 | 1.045 | 2.045 | 0.348 |
| code | 55.0 | 2.183 | 3.183 | 0.728 |
| structured | 48.4 | 1.824 | 2.824 | 0.608 |
| json | 48.3 | 1.735 | 2.735 | 0.578 |
| **all** | - | 1.662 | 2.662 | 0.554 |

Accepted tokens per draft position: 0: 1863, 1: 1312, 2: 989

_spec: ok (2026-10-17T21:23:40-0500)_

## Token-id regression check

Reference: /Users/justin/recipes/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark/bench/reference.json (pinned 2026-10-17T15:39:18-0500, served /home/spark/models/DeepSeek-V4.1-Flash-1017)

| prompt | compared | result | first divergence |
|---|---|---|---:|
| prose | token_ids | MISMATCH | 3 |
| code | token_ids | MATCH | - |
| math | token_ids | MISMATCH | 17 |
| json | token_ids | MATCH | - |
| reasoning_off | token_ids | MISMATCH | 1 |
| summary | token_ids | MATCH | - |

_check: FAILED (2026-10-17T21:26:04-0500)_
