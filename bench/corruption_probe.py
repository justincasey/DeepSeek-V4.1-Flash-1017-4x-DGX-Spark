#!/usr/bin/env python3
"""Count contexts that hit the thinking-mode flat-distribution defect (docs/findings/2026-10-18-...).

The `Say hi.` chat prompt (thinking on, effort high) is rendered with /tokenize, `</think>Hi! 👋 How can` is appended,
and the context is shifted by a system message of k words (k = 0..12). For each shift the next-token distribution is
read with /v1/completions (max_tokens 1, temperature 0). An intact forward pass gives ` I` at ~1.00; the defect
gives a flat distribution (`\\n` 0.10, `#` 0.08, ...). Prints `corrupt N/13` (r6v at a651912: 8/13).

usage: corruption_probe.py BASE_URL [--model NAME]
"""
import argparse
import json
import math
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("base_url")
p.add_argument("--model", default="deepseek-v4.1-flash")
args = p.parse_args()
base = args.base_url.rstrip("/")


def post(path, body):
    r = urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())


cont = post("/tokenize", {"model": args.model, "prompt": "</think>Hi! 👋 How can", "add_special_tokens": False})["tokens"]
bad = 0
cells = []
for k in range(13):
    msgs = ([{"role": "system", "content": " ".join(["Be"] * k)}] if k else []) + [{"role": "user", "content": "Say hi."}]
    ids = post("/tokenize", {"model": args.model, "messages": msgs, "add_generation_prompt": True,
                             "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})["tokens"]
    c = post("/v1/completions", {"model": args.model, "prompt": ids + cont, "max_tokens": 1, "temperature": 0,
                                 "logprobs": 1, "skip_special_tokens": False})["choices"][0]
    top_tok, top_lp = max(c["logprobs"]["top_logprobs"][0].items(), key=lambda kv: kv[1])
    prob = math.exp(top_lp)
    corrupt = prob < 0.5
    bad += corrupt
    cells.append(f"{k}:{top_tok!r}:{prob:.2f}")
print(" ".join(cells))
print(f"corrupt {bad}/13")
