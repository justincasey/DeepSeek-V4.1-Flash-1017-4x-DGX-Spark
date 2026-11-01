## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 |
|---|---:|
| prose | 66.4 (67.4) |
| code | 96.5 (98.6) |
| structured | 86.7 (88.6) |
| json | 96.2 (98.6) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 |
|---|---:|
| prose | 0.122 |
| code | 0.117 |
| structured | 0.139 |
| json | 0.140 |

_decode: ok (2026-10-31T17:47:30-0500)_
