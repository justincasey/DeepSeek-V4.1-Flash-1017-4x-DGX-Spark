"""Shift the same continuation by 0-12 tokens (padding system message) and read the next-token distribution.

Usage: DSV41_BASE=http://host:8000 python3.13 pos_probe.py   (Python 3.10+; read-only requests)
"""
import json, math, os, urllib.request
BASE = os.environ.get("DSV41_BASE", "http://192.168.1.121:8000")
def post(path, body):
    r = urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())
cont = post("/tokenize", {"model": "deepseek-v4.1-flash", "prompt": "</think>Hi! 👋 How can", "add_special_tokens": False})["tokens"]
for k in range(0, 13):
    msgs = ([{"role": "system", "content": " ".join(["Be"] * k)}] if k else []) + [{"role": "user", "content": "Say hi."}]
    ids = post("/tokenize", {"model": "deepseek-v4.1-flash", "messages": msgs, "add_generation_prompt": True,
                             "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})["tokens"]
    full = ids + cont
    c = post("/v1/completions", {"model": "deepseek-v4.1-flash", "prompt": full, "max_tokens": 1, "temperature": 0,
                                 "logprobs": 3, "skip_special_tokens": False})["choices"][0]
    tl = c["logprobs"]["top_logprobs"][0]
    top = sorted(tl.items(), key=lambda kv: -kv[1])
    print("sys_words=%2d next_pos=%d  %s" % (k, len(full), " | ".join("%r %.2f" % (t, math.exp(v)) for t, v in top)))
