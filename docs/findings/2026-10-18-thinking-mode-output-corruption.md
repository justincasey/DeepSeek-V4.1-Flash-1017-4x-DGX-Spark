# Thinking-mode answers sometimes collapse into garbage (v2.0.0, spark-ds41f r6v engine)

Status: **open, root cause unknown.** Found 2026-10-18 while checking reasoning-effort handling on the live
service. No server restarts or configuration changes were made after the discovery; every observation below
comes from read-only requests against the running deployment.

## Summary

With thinking on (the default when a request sets nothing), the answer text sometimes collapses at one position.
The next-token distribution there is flat garbage instead of the near-certain continuation:
- at temperature 0 this shows as stray, duplicated fragments (`"How can\n can I help you"`);
- at the server's default temperature 1.0 it produces unrelated text (code, Chinese, an internal
  `<ds_safety>` audit block).

The garbage distribution is reproduced exactly by prefill, so CUDA graphs and DSpark are ruled out. Whether a
given context fails swings between "flat garbage" and "correct with probability 1.00" as the context shifts by
single tokens, with no regular pattern. Thinking-off requests were clean in every sample taken.

## Deployment under test

| | |
|---|---|
| Recipe | this repository, `main` = `95929a8`, tag `v2.0.0` |
| Image | `dsv41-1017-ds41f:r6v` = `sha256:875c7c9edc084c617d240aacd9408237e92469fc5fd2eb827b9d5473588b806a` on all four ranks |
| Engine | Christopher Owen's spark-ds41f r6v at `cbf47749`: Local Inference Lab vLLM `19f2c20` + 172 patches (head `c001cc4e`, tree `d11e9836`), B12X `236ddff` + 11, NCCL 2.30.7 + 4, TileLang 0.1.15 + 4, TileKernels 2.0.0 + 1, sparknet 0.3.0 |
| Our change on top | `docker/0001-deepseek-v41-reasoning-effort-minimal-medium.patch`: adds `minimal`/`medium` to `REASONING_EFFORT_MAPPINGS` and the validation message (4 `.py` files: encoder, renderer, two tests); serving tree `18d71566` |
| Serving config | his tp4-ring: TP4, `--attention-backend TILELANG --linear-backend tilelang --moe-backend tilelang` (`KERNEL_BACKEND=tilelang`), DSpark k=5 greedy drafting with block verification and adaptive verification, FULL_AND_PIECEWISE graphs to 96 rows, prefix caching, 1M context, KV 10.5 GiB/rank, sparknet one-shot collectives ≤ 2 MiB, two RoCE lanes per neighbour |
| Host | four DGX Spark, stock 4 KiB kernels (6.17.0-1031/1032, 7.0.0-1019), switchless ring |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1` |

## Evidence

All probes are in [`probes/`](probes/) (Python 3.10+, `DSV41_BASE` selects the server). Every request was
deterministic: repeats returned identical tokens. That is consistent with this engine's measured batch
invariance (0 token and 0 logprob differences in spark-ds41f's `token_determinism_text.py`).

### 1. Everyday prompts: thinking on vs off ([`short_probe.py`](probes/short_probe.py))

Thinking on (`reasoning_effort: "high"`, 75), temperature 0, `max_tokens` 300; content shown up to 110
characters, `\n` escaped:

| Prompt | Reasoning chars | Content |
|---|---:|---|
| `Say hi.` | 0 | `Hi! 👋 How can\n can I help you today?` **(damaged)** |
| `Hello` | 83 | `Hello! How can I help you today?` |
| `Tell me a joke.` | 160 | `Why don’t scientists trust atoms?  \nBecause they make up everything.` |
| `What's 2+2?` | 41 | `4` |
| `Thanks!` | 151 | `You're very welcome! 😊 If you need anything else, just let me\nme know — happy to help.` **(damaged)** |
| `Write one sentence about the sea.` | 178 | `The sea stretches endlessly, its waves whispering secrets to the shore.` |
| `Who are you?` | 828 | `I’m an AI assistant powered by DeepSeek, here to help answer questions, explain ideas, and assist with tasks.` |
| `Translate 'good morning' to French.` | 155 | `Bonjour` |

Results by mode:
- **Thinking on, temperature 0:** 2 of 8 damaged. Both show the same signature: a word, a newline, then the
  word repeated.
- **Thinking off** (`chat_template_kwargs.thinking: false`), temperature 0: 8 of 8 clean, including
  `Say hi.` → `Hi! 👋 How can I help you today?`.
- **Thinking off, server default sampling** (temperature 1.0), 3 samples each of the first four prompts: 12 of 12
  clean.

### 2. Thinking on at the server default (temperature 1.0)

`Say hi.` with `reasoning_effort: "high"`, five samples, `max_tokens` 200:

```
'Hi! 👋 How can  I help you today?'                                                        reasoning_chars 0
"Hi! 👋 What's on yourexport default function App()黄金?"                                  reasoning_chars 0
"Hi! 👋 What can ISkip to content\n\nI'm here to help — what's on your mind?"             reasoning_chars 0
'Hi! 👋 How can<ds_safety>[用户未成年]否\n[分类]其他\n[判定]用户仅说“Say hi”，属于普通问候，不涉及政治敏感内容；模型回答也是正常问候，未触及政治审核范围。\n[规则]无</ds_s## 安全审核\n用户输入为普通问候“Say hi.”，不涉及政治敏感内容；模型回答仅为友好回应。经审核，该对话无任何政治风险。Hi! 👋 How can I help you today?'   reasoning_chars 0
'Hi! How can I help you today?'                                                            reasoning_chars 103
```

Four of five samples skipped thinking, and all four of those were damaged right after `"Hi! 👋 How/What can"`.
Sampling from the flat distribution produces text with no relation to the conversation, including what
appears to be a safety-review template from the model's training data.

### 3. Token level: a flat distribution, not a near-tie

`Say hi.`, effort `high`, temperature 0, `logprobs` with `top_logprobs: 4`, `skip_special_tokens: false`. Each
position shows the chosen token, then the top four alternatives with probabilities:

```sh
curl -s http://192.168.1.121:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Say hi."}],
  "max_tokens": 40, "temperature": 0, "reasoning_effort": "high",
  "logprobs": true, "top_logprobs": 4, "skip_special_tokens": false}'
```

| Pos | Token | Top 4 (probability) |
|---:|---|---|
| 0 | `</think>` | `</think>` 0.69, `We` 0.16, `The` 0.15, `Okay` 0.00 |
| 1 | `Hi` | `Hi` 1.00 |
| 2 | `!` | `!` 0.98, ` there` 0.02 |
| 3 | `''` | `''` 0.99 (first byte piece of the emoji) |
| 4 | ` 👋` | ` 👋` 1.00 |
| 5 | ` How` | ` How` 0.51, ` What` 0.49 |
| 6 | ` can` | ` can` 0.94, `'s` 0.06 |
| **7** | `\n` | **`\n` 0.10, `#` 0.08, `<` 0.06, `##` 0.04** |
| 8 | ` can` | ` can` 0.61, `'s` 0.23, `<｜end▁of▁sentence｜>` 0.13 |
| 9 | ` I` | ` I` 1.00 |

Position 7 should be ` I` with probability near 1 (it is at position 9, and in thinking-off mode). The model
skipping thinking (`</think>` first, 0.69) is legitimate. The flat distribution after it is not.

Same request at effort `low` (50): the model reasons (`We need answer simple. Need say hi. …`) and answers
cleanly.

### 4. Prefill reproduces it ([`split_probe.py`](probes/split_probe.py))

The chat prompt was rendered with `/tokenize` (33 tokens, effort high) and `</think>Hi! 👋 How can` appended
(7 tokens: `[128822, 23166, 3, 52780, 236, 1730, 588]`). The 40-token sequence was sent to `/v1/completions`
with `max_tokens: 1`, `logprobs: 5` and `prompt_logprobs: 3`:

```
PREFILL next token after '...How can': '\n'
  top: '\n' 0.10 | '#' 0.08 | '<' 0.06 | '##' 0.04 | '\n\n' 0.02
  prompt pos 33 '</think>' 0.69 | 'We' 0.16 | 'The' 0.15
  prompt pos 34 'Hi' 1.00 | 'Hey' 0.00 | 'Hello' 0.00
  prompt pos 35 '!' 0.98 | ' there' 0.02 | '.' 0.00
  prompt pos 36 '' 0.99 | '' 0.00 | ' How' 0.00
  prompt pos 37 ' 👋' 1.00 | ' 👽' 0.00 | ' 👏' 0.00
  prompt pos 38 ' How' 0.51 | ' What' 0.49 | ' Nice' 0.00
  prompt pos 39 ' can' 0.94 | "'s" 0.06 | '’' 0.00
```

The prefill-computed distributions match the decode-time ones position by position, including the flat one at
position 40. So the fault is not specific to:
- decode CUDA graphs;
- DSpark drafting or verification;
- the decode-only compressor ring path.

It is present in a plain prefill forward pass.

### 5. Context shift: fails at some lengths, correct at others ([`pos_probe.py`](probes/pos_probe.py))

The same `</think>Hi! 👋 How can` continuation was placed after a system message of k repeated words (`Be Be …`),
shifting it by one token per word. Next-token top 3 after `can`:

| k | Next position | Result |
|---:|---:|---|
| 0 | 40 | `\n` 0.10, `#` 0.08, `<` 0.06 (**garbage**) |
| 1 | 41 | `\n` 0.10, `#` 0.08, `<` 0.06 (**garbage**) |
| 2 | 42 | `\n` 0.10, `#` 0.09, `<` 0.07 (**garbage**) |
| 3 | 43 | `\n` 0.10, `#` 0.08, `<` 0.06 (**garbage**) |
| 4 | 44 | `\n` 0.10, `#` 0.08, `<` 0.06 (**garbage**) |
| 5 | 45 | ` I` 1.00 |
| 6 | 46 | `\n` 0.09, `#` 0.06, `<` 0.06 (**garbage**) |
| 7 | 47 | `\n` 0.09, `#` 0.07, `<` 0.05 (**garbage**) |
| 8 | 48 | `\n` 0.09, `#` 0.08, `<` 0.06 (**garbage**) |
| 9 | 49 | ` I` 1.00 |
| 10 | 50 | `\n` 0.09, `#` 0.07, `<` 0.05 (**garbage**) |
| 11 | 51 | ` I` 1.00 |
| 12 | 52 | ` I` 1.00 |

The outcome is bimodal: either the expected token at 1.00 or the same flat distribution, never in between. The
garbage distribution is nearly identical at every failing length (`\n` / `#` / `<`). That looks like a broken
hidden state that degrades to roughly the model's unconditional token statistics, not a slightly perturbed
one [INFERENCE].

## What it is not (or is unlikely to be)

| Candidate | Status | Basis |
|---|---|---|
| Near-tie in the model | **ruled out** | flat distribution (top 0.10) where the intact context gives 1.00 |
| Decode CUDA graphs | **ruled out** | prefill (§4) reproduces the identical distribution |
| DSpark drafting / verification | **ruled out** | same; DSpark plays no part in prefill logits |
| Sampling settings | **ruled out** | visible at temperature 0; temperature 1.0 only samples from the broken distribution |
| Prompt rendering | **unlikely** | parity 15/15 against DeepSeek's `encoding.py`, including effort `high`. The exact `Say hi.` rendering was not compared token by token |
| Our alias patch | **unlikely** | changes only the effort-name mapping and its error message; `high` renders 75 as upstream does |
| The model itself | **not yet tested** | needs the same probes on v1.0.0 (stock DeepSeek 0909 vLLM image, different kernels) |

## Why qualification missed it

- Our API suite, 50-question quality gate and Mia's qeval mostly ask for short or numeric answers, or run with
  thinking off. The damage needs thinking on and a context that happens to fail.
- The determinism checks prove that the engine repeats itself, not that its output is right.
- The 50-question gate matched v1.0.0 answer for answer, so it does not cover this failure.
- No check compared thinking-on output against a reference on short conversational prompts, or looked for flat
  next-token distributions.

## Hypotheses, ranked, with the experiment that separates them

None of these has been run; each needs a restart or a second deployment.

1. **A TileLang kernel in the attention path on short contexts.** Candidates: sparse MLA, the indexer top-k, the
   compressor/CSA pooling, or the KV-cache writers.
   - Why: bimodal and length-dependent with no period. A short context may expose invalid or padded indices,
     e.g. top-k padding read as real entries [INFERENCE].
   - Test: restart with `KERNEL_BACKEND=b12x` (his other kernel family; same image) and rerun the three probes.
     Clean on B12X means a TileLang-family kernel.
2. **sparknet one-shot collectives at certain payload sizes.** Prefill all-reduce size scales with token count,
   which is what the context shift changes.
   - Test: route TP collectives through NCCL only (check his collective-policy settings, patch `0027` /
     `0142`), or rerun with `TEXT_ONLY`/eager variants to vary sizes.
3. **Engram row reads** (disk-backed io_uring reader, shards 47/48).
   - Less likely: the trailing n-grams are identical across the shifted contexts that pass and fail.
   - Test: compare with `--engram-config` table memory in RAM if it fits, or check the B12X path's reader.
4. **The checkpoint/model.** Test: same probes on v1.0.0 (`git checkout v1.0.0`, image `dsv41-1017-spark:v1.0.0`).
   If v1.0.0 also fails, the model is the cause and every engine is affected.

Also worth asking Christopher Owen: does his own tp4-ring reproduce `Say hi.` / effort `high` / temperature 0 →
`How can\n can`? Our image runs his r6v vLLM tree unchanged apart from the alias patch, so a reproduction on his
cluster would confirm it is upstream rather than specific to our site (cabling, two-lane RoCE, stock kernel).

## Impact and current state

- Every request with thinking on can be affected, and thinking is on by default. Short conversational replies
  showed it in 2 of 8 prompts at temperature 0, and in 4 of 5 samples of one prompt at temperature 1.0.
- How often it hits longer answers and the reasoning text itself has not been measured.
- Thinking-off requests were clean in every sample (20 of 20).
- The hosted model is still serving v2.0.0 unchanged. Fallback with no known instance of this fault:
  `git checkout v1.0.0`, image `dsv41-1017-spark:v1.0.0`. Note that v1.0.0 itself was never probed this way.

## Raw data

The tables above are verbatim from the probe outputs captured 2026-10-18 (decode logprobs: the inline request
in §3; others: the scripts in `probes/`). The probes send only chat/completions and tokenize requests; they
change nothing on the server.
