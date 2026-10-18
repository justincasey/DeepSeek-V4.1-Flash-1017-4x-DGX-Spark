## Token-id regression check

Reference: /Users/justin/recipes/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark/bench/reference.json (pinned 2026-10-17T15:39:18-0500, served /home/spark/models/DeepSeek-V4.1-Flash-1017)

| prompt | compared | result | first divergence |
|---|---|---|---:|
| prose | token_ids | MISMATCH | 3 |
| code | token_ids | MATCH | - |
| math | token_ids | MISMATCH | 17 |
| json | token_ids | MATCH | - |
| reasoning_off | token_ids | MISMATCH | 1 |
| summary | token_ids | MATCH | - |

_check: FAILED (2026-10-17T21:26:04-0500)_
