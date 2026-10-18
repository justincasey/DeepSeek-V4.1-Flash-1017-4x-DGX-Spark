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
