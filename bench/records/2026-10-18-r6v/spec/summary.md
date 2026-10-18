## Speculative decoding (DSpark) acceptance

c=1, thinking off, greedy. accepted/draft = delta(vllm:spec_decode_num_accepted_tokens_total) / delta(vllm:spec_decode_num_drafts_total); tokens per step = 1 + accepted/draft.

| prompt | tok/s | accepted/draft | tokens/step | draft-token acceptance |
|---|---:|---:|---:|---:|
| prose | 61.3 | 0.936 | 1.936 | 0.342 |
| code | 87.4 | 2.179 | 3.179 | 0.532 |
| structured | 80.4 | 1.845 | 2.845 | 0.502 |
| json | 88.6 | 2.072 | 3.072 | 0.578 |
| **all** | - | 1.741 | 2.741 | 0.497 |

Accepted tokens per draft position: 0: 2874, 1: 1831, 2: 1041, 3: 715, 4: 391

_spec: ok (2026-10-17T22:37:57-0500)_
