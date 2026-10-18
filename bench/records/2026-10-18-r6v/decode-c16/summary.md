## Decode

Aggregate tok/s (median per-stream decode tok/s = (completion tokens - 1) / (last - first token time), prefill excluded); window 30.0s per cell, c streaming workers, max_tokens 512, thinking off, greedy. Tokens from usage.completion_tokens.

| prompt | c=16 |
|---|---:|
| prose | 551.8 (35.1) |
| code | 574.7 (37.1) |
| structured | 566.0 (34.2) |
| json | 639.1 (43.3) |

TTFT (median over the cell's own worker requests; seconds to the first content delta at concurrency c):

| prompt | c=16 |
|---|---:|
| prose | 0.385 |
| code | 0.335 |
| structured | 0.454 |
| json | 0.367 |

_decode: ok (2026-10-17T22:40:34-0500)_
