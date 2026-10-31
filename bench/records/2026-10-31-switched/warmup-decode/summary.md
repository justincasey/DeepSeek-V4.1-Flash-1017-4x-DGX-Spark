## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 20.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 |
|---|---:|
| prose | 67.2 (68.1) |
| code | 97.7 (99.8) |
| structured | 87.8 (89.7) |
| json | 97.3 (99.9) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 |
|---|---:|
| prose | 0.121 |
| code | 0.119 |
| structured | 0.137 |
| json | 0.142 |

_decode: ok (2026-10-31T17:01:11-0500)_
