## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 | c=8 |
|---|---:|---:|
| prose | 67.1 (68.1) | 347.3 (44.3) |
| code | 97.7 (99.7) | 335.2 (43.2) |
| structured | 87.6 (89.6) | 418.3 (56.6) |
| json | 97.2 (99.8) | 522.6 (67.8) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 | c=8 |
|---|---:|---:|
| prose | 0.121 | 0.185 |
| code | 0.115 | 0.352 |
| structured | 0.140 | 0.228 |
| json | 0.139 | 0.243 |

_decode: ok (2026-10-31T17:02:44-0500)_
