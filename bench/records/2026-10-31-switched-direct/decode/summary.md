## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 |
|---|---:|
| prose | 66.0 (67.0) |
| code | 96.0 (98.0) |
| structured | 86.1 (87.9) |
| json | 95.6 (98.1) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 |
|---|---:|
| prose | 0.123 |
| code | 0.115 |
| structured | 0.139 |
| json | 0.142 |

_decode: ok (2026-10-31T18:03:07-0500)_
