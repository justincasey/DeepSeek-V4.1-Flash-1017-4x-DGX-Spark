## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 20.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 |
|---|---:|
| prose | 66.2 (67.1) |
| code | 96.0 (98.1) |
| structured | 86.3 (88.2) |
| json | 95.6 (98.1) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 |
|---|---:|
| prose | 0.121 |
| code | 0.120 |
| structured | 0.140 |
| json | 0.141 |

_decode: ok (2026-10-31T18:01:32-0500)_
