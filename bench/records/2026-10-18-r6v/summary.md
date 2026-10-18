# Benchmark run

decode: ok | prefill: ok | needle: ok | spec: ok | check: ok

## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| prose | 61.4 (62.3) | 116.6 (59.2) | 204.4 (52.1) | 274.8 (33.5) |
| code | 87.2 (89.0) | 159.3 (82.6) | 229.4 (57.5) | 422.7 (54.6) |
| structured | 80.6 (82.5) | 147.5 (75.4) | 258.1 (66.2) | 436.2 (55.8) |
| json | 88.5 (90.9) | 161.0 (82.5) | 293.1 (75.2) | 307.3 (39.0) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| prose | 0.147 | 0.147 | 0.276 | 0.306 |
| code | 0.132 | 0.142 | 0.299 | 0.344 |
| structured | 0.155 | 0.166 | 0.207 | 0.239 |
| json | 0.164 | 0.171 | 0.203 | 0.392 |

_decode: ok (2026-10-17T22:16:55-0500)_

## Prefill

Unique prompts (no prefix-cache hits), max_tokens 1, sequential. Tokens from usage.prompt_tokens.

| target | prompt_tokens | seconds | tok/s |
|---|---:|---:|---:|
| 4k | 4087 | 0.80 | 5105.4 |
| 16k | 16368 | 2.85 | 5752.6 |
| 32k | 32751 | 5.45 | 6012.6 |
| 64k | 65519 | 10.91 | 6003.5 |
| 128k | 131052 | 22.15 | 5915.8 |
| 256k | 262135 | 46.98 | 5579.4 |
| 512k | 524262 | 103.98 | 5042.0 |
| 1M | 1048080 | 251.92 | 4160.4 |

_prefill: ok (2026-10-17T22:26:27-0500)_

## Needle

| result | target | prompt_tokens | seconds | expected | answer |
|---|---:|---:|---:|---|---|
| PASS | 1000000 | 1000006 | 236.580 | 94440 | 94440 |

_needle: ok (2026-10-17T22:33:57-0500)_

## Speculative decoding (DSpark) acceptance

c=1, thinking off, greedy. accepted/draft = delta(vllm:spec_decode_num_accepted_tokens_total) / delta(vllm:spec_decode_num_drafts_total); tokens per step = 1 + accepted/draft.

| prompt | tok/s | accepted/draft | tokens/step | draft-token acceptance |
|---|---:|---:|---:|---:|
| prose | 61.3 | 0.936 | 1.936 | 0.342 |
| code | 87.4 | 2.179 | 3.179 | 0.532 |
| structured | 80.4 | 1.845 | 2.845 | 0.502 |
| json | 88.6 | 2.072 | 3.072 | 0.578 |
| **all** | - | 1.741 | 2.741 | 0.497 |

Accepted tokens per draft position: 0: 2874, 1: 1831, 2: 1041, 3: 715, 4: 391

_spec: ok (2026-10-17T22:37:57-0500)_

## Token-id regression check

Reference: /Users/justin/recipes/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark.ds41f/bench/reference.json (pinned 2026-10-17T22:16:47-0500, served /models)

| prompt | compared | result | first divergence |
|---|---|---|---:|
| prose | token_ids | MATCH | - |
| code | token_ids | MATCH | - |
| math | token_ids | MATCH | - |
| json | token_ids | MATCH | - |
| reasoning_off | token_ids | MATCH | - |
| summary | token_ids | MATCH | - |

_check: ok (2026-10-17T22:40:12-0500)_
