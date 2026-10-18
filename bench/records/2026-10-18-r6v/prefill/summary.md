## Prefill

Unique prompts (no prefix-cache hits), max_tokens 1, sequential. Tokens from usage.prompt_tokens.

| target | prompt_tokens | seconds | tok/s |
|---|---:|---:|---:|
| 4k | 4087 | 0.80 | 5105.4 |
| 16k | 16368 | 2.85 | 5752.6 |
| 32k | 32751 | 5.45 | 6012.6 |
| 64k | 65519 | 10.91 | 6003.5 |
| 128k | 131052 | 22.15 | 5915.8 |
| 256k | 262135 | 46.98 | 5579.4 |
| 512k | 524262 | 103.98 | 5042.0 |
| 1M | 1048080 | 251.92 | 4160.4 |

_prefill: ok (2026-10-17T22:26:27-0500)_
