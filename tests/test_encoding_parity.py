#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Server prompt rendering must equal DeepSeek's DeepSeek-V4.1-Flash-1017 reference encoder.

    python3 tests/test_encoding_parity.py --base-url http://192.168.1.121:8000 [--model NAME] [--api-key K | env API_KEY=K]

Fetches encoding/encoding.py and its golden fixtures from the pinned checkpoint revision (MIT, DeepSeek),
sha256-checked against config/DeepSeek-V4.1-Flash-1017.sha256. Each case is rendered by the reference encoder;
the server renders the same OpenAI request through POST /tokenize. Token ids must match exactly. Covers the
golden fixtures, thinking on/off and default, every reasoning-effort tier and the OpenAI aliases, namespaced
tools, the 1017 tool-loop turn rule, and mid-conversation system messages. Image placement is exercised live
by api_conformance.py (image_input).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "config", "DeepSeek-V4.1-Flash-1017.sha256")
REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ARGS: argparse.Namespace

def fetch(path: str) -> bytes:
    digests = {}
    for line in open(MANIFEST):
        if line.strip():
            digest, name = line.split(None, 1)
            digests[name.strip()] = digest
    data = urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{ARGS.revision}/{path}", timeout=120).read()
    if hashlib.sha256(data).hexdigest() != digests.get(path):
        sys.exit(f"{path}: sha256 does not match {MANIFEST}")
    return data


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(ARGS.base_url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if ARGS.api_key:
        req.add_header("Authorization", f"Bearer {ARGS.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path} HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from None


def server_chat_tokens(messages, tools, kwargs) -> list[int]:
    body = {"model": ARGS.model, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": kwargs}
    if tools:
        body["tools"] = tools
    return post("/tokenize", body)["tokens"]


def server_text_tokens(text: str) -> list[int]:
    return post("/tokenize", {"model": ARGS.model, "prompt": text, "add_special_tokens": False})["tokens"]


def detok(tokens: list[int]) -> str:
    return post("/detokenize", {"model": ARGS.model, "tokens": tokens})["prompt"]


def with_tool_call_ids(messages):
    """Publisher fixtures omit tool-call ids, which the OpenAI schema requires. DSML never renders ids, so
    numbering them call_0, call_1, ... (the fixtures' tool_call_id convention) leaves the prompt unchanged."""
    msgs, n = copy.deepcopy(messages), 0
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if not tc.get("id"):
                tc["id"] = f"call_{n}"
            n += 1
    return msgs


def with_tools_on_system(messages, tools):
    """vLLM's V4.1 wrapper attaches request tools to the first system message, inserting an empty one."""
    msgs = copy.deepcopy(messages)
    if tools:
        system = next((m for m in msgs if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            msgs.insert(0, system)
        system["tools"] = tools
    return msgs


def compare(name: str, reference: str, messages, tools, kwargs) -> bool:
    want = server_text_tokens(reference)
    got = server_chat_tokens(messages, tools, kwargs)
    if got == want:
        print(f"PASS {name} ({len(got)} tokens)")
        return True
    i = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    lo = max(0, i - 12)
    print(f"FAIL {name}: first difference at token {i} (server {len(got)} tokens, reference {len(want)})")
    print(f"     server:    {detok(got[lo:i + 12])!r}")
    print(f"     reference: {detok(want[lo:i + 12])!r}")
    return False


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
NS_TOOLS = [{"type": "function", "function": {"name": "files::read", "description": "Read a file.",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "web::read", "description": "Read a URL.",
             "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}}]


def main() -> int:
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="server root, e.g. http://host:8000")
    p.add_argument("--revision", default="1bae0e1088993b9b82de08f59de5be5a34e42170")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="default: env API_KEY")
    ARGS = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        enc_path = os.path.join(tmp, "encoding.py")
        open(enc_path, "wb").write(fetch("encoding/encoding.py"))
        spec = importlib.util.spec_from_file_location("ref_encoding", enc_path)
        enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enc)
        ok = True

        # 1. golden fixtures reachable through the OpenAI API: exact publisher renders
        openai_roles = {"system", "developer", "user", "assistant", "tool"}
        for i in range(1, 6):
            raw = json.loads(fetch(f"encoding/tests/test_input_{i}.json"))
            case = raw if isinstance(raw, dict) else {"messages": raw}
            roles = {m["role"] for m in case["messages"]}
            if not roles <= openai_roles:
                print(f"SKIP golden_{i}: roles {sorted(roles - openai_roles)} are not expressible in the OpenAI API")
                continue
            if "image" in json.dumps(case["messages"]):
                print(f"SKIP golden_{i}: image case (covered live by api_conformance image_input)")
                continue
            path = os.path.join(tmp, f"in{i}.json")
            open(path, "w").write(json.dumps(raw))
            ref, _ = enc.encode_case(enc.load_cases(path)[0], thinking_mode="chat")
            assert ref == fetch(f"encoding/tests/test_output_{i}.txt").decode(), f"golden_{i} reference drifted"
            kw = {"thinking": case.get("thinking_mode") == "thinking"}
            if case.get("reasoning_effort") is not None:
                kw["reasoning_effort"] = case["reasoning_effort"]
            ok &= compare(f"golden_{i}", ref, with_tool_call_ids(case["messages"]), case.get("tools"), kw)

        base = [{"role": "system", "content": "You are a careful assistant."},
                {"role": "user", "content": "How many primes are below 60?"}]

        # 2. thinking off, thinking on with the default effort, every tier, integer budget, OpenAI aliases
        ok &= compare("chat_mode", enc.encode_messages(base, thinking_mode="chat"), base, None, {"thinking": False})
        ok &= compare("thinking_default_effort", enc.encode_messages(base, thinking_mode="thinking"), base, None, {})
        for effort in ("low", "high", "max", 30):
            ok &= compare(f"effort_{effort}", enc.encode_messages(base, thinking_mode="thinking", reasoning_effort=effort),
                          base, None, {"thinking": True, "reasoning_effort": effort})
        # xhigh keeps its pre-1017 budget of 75 (= high), as in the Local Inference Lab fork and spark-ds41f.
        for alias, tier in (("minimal", "low"), ("medium", "high"), ("xhigh", "high")):
            ok &= compare(f"effort_alias_{alias}", enc.encode_messages(base, thinking_mode="thinking", reasoning_effort=tier),
                          base, None, {"thinking": True, "reasoning_effort": alias})

        # 3. namespaced tools (qualified names)
        msgs = [{"role": "user", "content": "Read /etc/hostname and https://example.com."}]
        ok &= compare("namespaced_tools", enc.encode_messages(with_tools_on_system(msgs, NS_TOOLS), thinking_mode="chat"),
                      msgs, NS_TOOLS, {"thinking": False})

        # 4. agentic loop: reasoning before a tool call stays in the prompt after the tool result (1017 turn rule)
        loop = [{"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "I should call the weather tool.",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 21}"}]
        ok &= compare("tool_loop_thinking", enc.encode_messages(with_tools_on_system(loop, [WEATHER]), thinking_mode="thinking"),
                      loop, [WEATHER], {"thinking": True})

        # 5. multi-turn: earlier reasoning dropped at a new user turn; mid-conversation system message
        multi = [{"role": "user", "content": "Hi."},
                 {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
                 {"role": "system", "content": "Answer in French from now on."},
                 {"role": "user", "content": "How are you?"}]
        ok &= compare("multi_turn_mid_system", enc.encode_messages(multi, thinking_mode="thinking"), multi, None,
                      {"thinking": True})

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
