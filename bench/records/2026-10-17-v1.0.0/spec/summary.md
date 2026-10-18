## Speculative decoding (DSpark) acceptance

c=1, thinking off, greedy. accepted/draft = delta(vllm:spec_decode_num_accepted_tokens_total) / delta(vllm:spec_decode_num_drafts_total); tokens per step = 1 + accepted/draft.

| prompt | tok/s | accepted/draft | tokens/step | draft-token acceptance |
|---|---:|---:|---:|---:|
| prose | 36.0 | 1.045 | 2.045 | 0.348 |
| code | 55.0 | 2.183 | 3.183 | 0.728 |
| structured | 48.4 | 1.824 | 2.824 | 0.608 |
| json | 48.3 | 1.735 | 2.735 | 0.578 |
| **all** | - | 1.662 | 2.662 | 0.554 |

Accepted tokens per draft position: 0: 1863, 1: 1312, 2: 989

_spec: ok (2026-10-17T21:23:40-0500)_
