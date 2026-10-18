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
