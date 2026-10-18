## Prefill

Unique prompts (no prefix-cache hits), max_tokens 1, sequential. Tokens from usage.prompt_tokens.

| target | prompt_tokens | seconds | tok/s |
|---|---:|---:|---:|
| 4k | 4087 | 1.91 | 2137.7 |
| 16k | 16367 | 9.01 | 1817.2 |
| 32k | 32753 | 17.22 | 1902.3 |
| 64k | 65518 | 36.19 | 1810.2 |
| 128k | 131054 | 76.02 | 1724.0 |
| 256k | 262133 | 178.02 | 1472.5 |
| 512k | 524260 | 408.86 | 1282.2 |
| 1M | 1048079 | 1082.59 | 968.1 |

_prefill: ok (2026-10-17T20:36:57-0500)_
