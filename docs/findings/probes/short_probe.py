"""Eight everyday prompts, thinking on (effort high) and off, temperature 0, plus sampled thinking-off runs.

Usage: DSV41_BASE=http://host:8000 python3.13 short_probe.py   (Python 3.10+; read-only requests)
"""
import json, os, urllib.request
BASE = os.environ.get("DSV41_BASE", "http://192.168.1.121:8000") + "/v1/chat/completions"
def chat(content, kw, temp, extra=None):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": content}], "max_tokens": 300}
    if temp is not None: body["temperature"] = temp
    body.update(kw)
    d = json.loads(urllib.request.urlopen(urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read())
    m = d["choices"][0]["message"]
    return (m.get("content") or "").replace("\n", "\\n")[:110], len(m.get("reasoning_content") or m.get("reasoning") or "")
prompts = ["Say hi.", "Hello", "Tell me a joke.", "What's 2+2?", "Thanks!", "Write one sentence about the sea.", "Who are you?", "Translate 'good morning' to French."]
modes = {"think-high": {"reasoning_effort": "high"}, "think-off": {"chat_template_kwargs": {"thinking": False}}}
for mode, kw in modes.items():
    print("==", mode, "temperature 0")
    for p in prompts:
        c, r = chat(p, kw, 0)
        print(f"  {p!r:42} reasoning={r:4d}  {c!r}")
print("== think-off, server default sampling (temperature 1.0), 3 samples each")
for p in prompts[:4]:
    for i in range(3):
        c, r = chat(p, modes["think-off"], None)
        print(f"  {p!r:20} {c!r}")
