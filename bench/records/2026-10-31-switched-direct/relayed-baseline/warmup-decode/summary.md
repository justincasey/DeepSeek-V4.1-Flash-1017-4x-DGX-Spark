## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 20.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 |
|---|---:|
| prose | 66.3 (67.4) |
| code | 96.4 (98.5) |
| structured | 86.5 (88.4) |
| json | 96.2 (98.6) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 |
|---|---:|
| prose | 0.131 |
| code | 0.119 |
| structured | 0.139 |
| json | 0.144 |

_decode: ok (2026-10-31T17:45:55-0500)_
