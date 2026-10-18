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
