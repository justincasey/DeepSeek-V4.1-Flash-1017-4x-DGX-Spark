"""Prefill-path check: feed the damaged greedy sequence back as a prompt and read prefill logprobs.

Usage: DSV41_BASE=http://host:8000 python3.13 split_probe.py   (Python 3.10+; read-only requests)
"""
import json, math, os, urllib.request
BASE = os.environ.get("DSV41_BASE", "http://192.168.1.121:8000")
def post(path, body):
    r = urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())
msgs = [{"role": "user", "content": "Say hi."}]
# rendered prompt token ids exactly as chat would render them (effort high)
tok = post("/tokenize", {"model": "deepseek-v4.1-flash", "messages": msgs, "add_generation_prompt": True,
                         "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})
ids = tok["tokens"]
print("prompt tokens", len(ids))
cont = post("/tokenize", {"model": "deepseek-v4.1-flash", "prompt": "</think>Hi! 👋 How can", "add_special_tokens": False})["tokens"]
full = ids + cont
print("continuation ids", cont)
# 1) prefill path: prompt_logprobs over the full sequence + next token
out = post("/v1/completions", {"model": "deepseek-v4.1-flash", "prompt": full, "max_tokens": 1, "temperature": 0,
                               "logprobs": 5, "prompt_logprobs": 3, "skip_special_tokens": False})
c = out["choices"][0]
print("PREFILL next token after '...How can':", repr(c["text"]))
tl = c["logprobs"]["top_logprobs"][0]
print("  top:", " | ".join("%r %.2f" % (k, math.exp(v)) for k, v in sorted(tl.items(), key=lambda kv: -kv[1])))
pl = c.get("prompt_logprobs") or []
for pos in range(len(ids), len(full)):
    e = pl[pos] or {}
    best = sorted(e.values(), key=lambda x: -x["logprob"])[:3]
    print("  prompt pos", pos, " | ".join("%r %.2f" % (b.get("decoded_token"), math.exp(b["logprob"])) for b in best))
