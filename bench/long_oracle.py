#!/usr/bin/env python3
"""Long-prompt bit-identity oracle: prefill paths the short token-id oracle never reaches.

usage: long_oracle.py pin|check BASE_URL [--model NAME] [--reference PATH]

Three fixed prompts (~6K, ~46K, ~109K tokens: one, two and five 24,576-token prefill chunks), each sent with
a fresh cache_salt so neither the in-process nor the persisted prefix cache serves it, greedy, thinking off,
48 tokens with the top-5 logprobs of every position. Above 205 rows prefill runs sequence parallelism and the
rank-order reduce-scatter, whose exchange and NCCL settings are what fabric changes touch. `pin` writes the
reference; `check` requires every token id and every logprob to be identical (exit 1 otherwise).
"""
import argparse
import json
import random
import sys
import urllib.request
import uuid

p = argparse.ArgumentParser()
p.add_argument("mode", choices=("pin", "check"))
p.add_argument("base_url")
p.add_argument("--model", default="deepseek-v4.1-flash")
p.add_argument("--reference", default="bench/long_reference.json")
args = p.parse_args()
url = args.base_url.rstrip("/") + "/v1/chat/completions"


def prompt(words: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = [f"record {i}: id={rng.randint(0, 10**9)} tag={rng.choice('abcdefgh')}{rng.randint(0, 999)}"
             for i in range(words // 4)]
    return ("Below is a log. Report the tag of record 7 and the number of records, then stop.\n"
            + "\n".join(lines))


CASES = {"6k": prompt(1_800, 1), "46k": prompt(13_500, 2), "109k": prompt(31_500, 3)}


def run(text: str) -> dict:
    body = {"model": args.model, "messages": [{"role": "user", "content": text}], "max_tokens": 48,
            "temperature": 0, "logprobs": True, "top_logprobs": 5, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=900))
    content = out["choices"][0]["logprobs"]["content"]
    return {"prompt_tokens": out["usage"]["prompt_tokens"],
            "tokens": [c["token"] for c in content],
            "logprobs": [[c["logprob"]] + [t["logprob"] for t in c["top_logprobs"]] for c in content]}


got = {name: run(text) for name, text in CASES.items()}
if args.mode == "pin":
    json.dump(got, open(args.reference, "w"), indent=1)
    print("pinned", ", ".join(f"{n} ({g['prompt_tokens']} tokens)" for n, g in got.items()))
    sys.exit(0)
ref = json.load(open(args.reference))
bad = 0
for name, g in got.items():
    r = ref[name]
    same = g == r
    first = next((i for i, (a, b) in enumerate(zip(g["tokens"], r["tokens"])) if a != b), None)
    print(f"{name}: {'MATCH' if same else 'DIFF'} ({g['prompt_tokens']} tokens"
          + (f", first token divergence at {first}" if first is not None else
             "" if same else ", tokens equal, logprobs differ") + ")")
    bad += not same
print(f"long oracle {len(got) - bad}/{len(got)}")
sys.exit(1 if bad else 0)
