## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 20.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=1 |
|---|---:|
| prose | 64.8 (65.8) |
| code | 94.4 (96.6) |
| structured | 84.5 (86.3) |
| json | 93.8 (96.3) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=1 |
|---|---:|
| prose | 0.129 |
| code | 0.127 |
| structured | 0.144 |
| json | 0.143 |

_decode: ok (2026-10-18T14:24:46-0500)_
