#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenAI API conformance suite for the live DeepSeek-V4.1-Flash-1017 endpoint.

stdlib only: runs from the control machine or any node.

    python3 tests/api_conformance.py --base-url http://192.168.1.121:8000/v1 [--api-key K | env API_KEY=K]
        [--model NAME] [--only case1,case2] [--long-tokens 16000]

Prints one PASS/FAIL line per case and exits non-zero if any case fails. Every case is a
production requirement: endpoints, sampling controls, thinking/reasoning, tool calling,
structured output, image input, long context, auth and error handling.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import time
import traceback
import urllib.error
import urllib.request
import zlib

ARGS: argparse.Namespace
MAX_MODEL_LEN = 1 << 20   # replaced from /v1/models at start


def budget(tokens: int) -> int:
    """Generation budget that fits the server's context window with room for the prompt."""
    return max(64, min(tokens, MAX_MODEL_LEN - 1024))


class Fail(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


# ---------------------------------------------------------------- HTTP
def _req(method: str, path: str, body: dict | None = None, key: str | None = "default", timeout: float = 600):
    url = ARGS.base_url.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    k = ARGS.api_key if key == "default" else key
    if k:
        req.add_header("Authorization", f"Bearer {k}")
    return urllib.request.urlopen(req, timeout=timeout)


def call(method: str, path: str, body: dict | None = None, key: str | None = "default") -> tuple[int, dict]:
    try:
        with _req(method, path, body, key) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw.decode(errors="replace")}


def chat(**body) -> dict:
    body.setdefault("model", ARGS.model)
    status, out = call("POST", "/chat/completions", body)
    check(status == 200, f"chat HTTP {status}: {json.dumps(out)[:400]}")
    return out


def stream(path: str, body: dict) -> list[dict]:
    body = dict(body, stream=True)
    body.setdefault("model", ARGS.model)
    events = []
    with _req("POST", path, body) as r:
        check(r.status == 200, f"stream HTTP {r.status}")
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                events.append({"_done": True})
                break
            events.append(json.loads(payload))
    check(events and events[-1].get("_done"), "stream did not end with [DONE]")
    return events[:-1]


def no_think(**extra) -> dict:
    return {"chat_template_kwargs": {"thinking": False}, "temperature": 0, **extra}


def reasoning_of(msg: dict) -> str:
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------- fixtures
def png(rgb: tuple[int, int, int], size: int = 64) -> str:
    """Solid-colour PNG as a data URL, built with zlib only."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    row = b"\x00" + bytes(rgb) * size
    raw = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(raw).decode()


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Current local time for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def tool_calls(msg: dict) -> list[tuple[str, dict]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        check(tc.get("type") == "function", f"tool call type {tc.get('type')}")
        check(tc.get("id"), "tool call without id")
        out.append((tc["function"]["name"], json.loads(tc["function"]["arguments"])))
    return out


# ---------------------------------------------------------------- cases
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def health_and_models():
    root = ARGS.base_url.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(root + "/health", timeout=30) as r:
        check(r.status == 200, f"/health {r.status}")
    status, out = call("GET", "/models")
    check(status == 200, f"/models {status}")
    ids = [m["id"] for m in out["data"]]
    check(ARGS.model in ids, f"{ARGS.model} not in {ids}")
    m = next(m for m in out["data"] if m["id"] == ARGS.model)
    print(f"      max_model_len={m.get('max_model_len')}")


@case
def chat_basic_and_usage():
    out = chat(messages=user("What is 17*23? Reply with the integer only."), max_tokens=16, **no_think())
    c = out["choices"][0]
    check("391" in c["message"]["content"], f"answer {c['message']['content']!r}")
    check(c["finish_reason"] == "stop", f"finish_reason {c['finish_reason']}")
    u = out["usage"]
    check(u["prompt_tokens"] > 0 and u["completion_tokens"] > 0, f"usage {u}")
    check(u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], f"usage sum {u}")


@case
def greedy_is_deterministic():
    body = dict(messages=user("Write one sentence about the ocean."), max_tokens=48, **no_think())
    a = chat(**body)["choices"][0]["message"]["content"]
    b = chat(**body)["choices"][0]["message"]["content"]
    check(a == b, f"greedy outputs differ:\n  {a!r}\n  {b!r}")


@case
def seed_is_reproducible():
    body = dict(messages=user("Invent a name for a cat."), max_tokens=24,
                chat_template_kwargs={"thinking": False}, temperature=0.9, seed=1234)
    a = chat(**body)["choices"][0]["message"]["content"]
    b = chat(**body)["choices"][0]["message"]["content"]
    check(a == b, f"seeded outputs differ: {a!r} vs {b!r}")


@case
def streaming_chat_with_usage():
    ev = stream("/chat/completions", dict(messages=user("Count from 1 to 5, comma separated."), max_tokens=32,
                                         stream_options={"include_usage": True}, **no_think()))
    text = "".join((e["choices"][0]["delta"].get("content") or "") for e in ev if e.get("choices"))
    check("1" in text and "5" in text, f"streamed text {text!r}")
    finishes = [e["choices"][0].get("finish_reason") for e in ev if e.get("choices")]
    check("stop" in finishes, f"finish reasons {finishes}")
    check(ev[-1].get("usage", {}).get("completion_tokens", 0) > 0, f"no usage in final chunk: {ev[-1]}")


@case
def max_tokens_truncates():
    out = chat(messages=user("Write a long story about a dragon."), max_tokens=5, **no_think())
    c = out["choices"][0]
    check(c["finish_reason"] == "length", f"finish_reason {c['finish_reason']}")
    check(out["usage"]["completion_tokens"] == 5, f"completion_tokens {out['usage']['completion_tokens']}")


@case
def stop_sequence():
    out = chat(messages=user("Count from 1 to 10, separated by commas and spaces."), max_tokens=64,
               stop=["6"], **no_think())
    c = out["choices"][0]
    check("6" not in c["message"]["content"] and "5" in c["message"]["content"], f"content {c['message']['content']!r}")
    check(c["finish_reason"] == "stop", f"finish_reason {c['finish_reason']}")


@case
def n_choices():
    out = chat(messages=user("Say a random fruit."), max_tokens=8, n=2,
               chat_template_kwargs={"thinking": False}, temperature=1.0)
    check(len(out["choices"]) == 2, f"{len(out['choices'])} choices")


@case
def logprobs():
    out = chat(messages=user("Say hello."), max_tokens=4, logprobs=True, top_logprobs=3, **no_think())
    lp = out["choices"][0]["logprobs"]["content"]
    check(lp and all(len(t["top_logprobs"]) == 3 for t in lp), f"logprobs {json.dumps(lp)[:300]}")


@case
def completions_endpoint():
    status, out = call("POST", "/completions", {"model": ARGS.model, "prompt": "The capital of France is",
                                                 "max_tokens": 8, "temperature": 0})
    check(status == 200, f"/completions {status}: {out}")
    check("Paris" in out["choices"][0]["text"], f"text {out['choices'][0]['text']!r}")
    ev = stream("/completions", {"prompt": "1, 2, 3, 4,", "max_tokens": 6, "temperature": 0})
    check("5" in "".join(e["choices"][0]["text"] for e in ev if e.get("choices")), "streamed completion lacks 5")


@case
def thinking_on_splits_reasoning():
    out = chat(messages=user("Is 221 prime? Answer yes or no."), max_tokens=budget(4096), temperature=0,
               chat_template_kwargs={"thinking": True})
    m = out["choices"][0]["message"]
    check(len(reasoning_of(m)) > 0, "thinking on but no reasoning content")
    check("no" in (m["content"] or "").lower(), f"answer {m['content']!r}")  # 221 = 13*17


@case
def thinking_off_has_no_reasoning():
    m = chat(messages=user("Is 221 prime? Answer yes or no."), max_tokens=16, **no_think())["choices"][0]["message"]
    check(not reasoning_of(m), f"thinking off but reasoning present: {reasoning_of(m)[:200]!r}")


@case
def thinking_streams_reasoning_then_content():
    ev = stream("/chat/completions", dict(messages=user("What is 12*12?"), max_tokens=budget(2048), temperature=0,
                                          chat_template_kwargs={"thinking": True}))
    kinds = []
    for e in ev:
        if not e.get("choices"):
            continue
        d = e["choices"][0]["delta"]
        if d.get("reasoning_content") or d.get("reasoning"):
            kinds.append("r")
        if d.get("content"):
            kinds.append("c")
    check("r" in kinds and "c" in kinds, f"delta kinds {set(kinds)}")
    check(kinds.index("c") > kinds.index("r"), "content streamed before reasoning")


@case
def reasoning_effort_tiers():
    lengths = {}
    for tier in ("low", "high", "max"):
        out = chat(messages=user("How many primes are below 60?"), max_tokens=budget(8192), temperature=0,
                   reasoning_effort=tier, chat_template_kwargs={"thinking": True})
        m = out["choices"][0]["message"]
        check(reasoning_of(m), f"no reasoning at effort {tier}")
        check("17" in (m["content"] or ""), f"effort {tier} answer {m['content']!r}")
        lengths[tier] = out["usage"]["completion_tokens"]
    print(f"      completion tokens by effort: {lengths}")


@case
def tool_call_single():
    out = chat(messages=user("What's the weather in Paris right now?"), tools=[WEATHER], tool_choice="auto",
               max_tokens=256, **no_think())
    c = out["choices"][0]
    calls = tool_calls(c["message"])
    check(c["finish_reason"] == "tool_calls", f"finish_reason {c['finish_reason']}")
    check(calls and calls[0][0] == "get_weather" and "paris" in calls[0][1]["city"].lower(), f"calls {calls}")


@case
def tool_call_parallel():
    m = chat(messages=user("Get the weather in Paris and in Tokyo."), tools=[WEATHER], max_tokens=512,
             **no_think())["choices"][0]["message"]
    cities = sorted(a["city"].lower() for n, a in tool_calls(m) if n == "get_weather")
    check(any("paris" in c for c in cities) and any("tokyo" in c for c in cities), f"cities {cities}")


@case
def tool_choice_none_required_named():
    m = chat(messages=user("What's the weather in Rome?"), tools=[WEATHER], tool_choice="none", max_tokens=128,
             **no_think())["choices"][0]["message"]
    check(not m.get("tool_calls"), f"tool_choice none produced calls {m.get('tool_calls')}")
    m = chat(messages=user("Hello there."), tools=[WEATHER], tool_choice="required", max_tokens=256,
             **no_think())["choices"][0]["message"]
    check(tool_calls(m), "tool_choice required produced no call")
    m = chat(messages=user("What's going on in Berlin?"), tools=[WEATHER, TIME],
             tool_choice={"type": "function", "function": {"name": "get_time"}}, max_tokens=256,
             **no_think())["choices"][0]["message"]
    calls = tool_calls(m)
    check(calls and all(n == "get_time" for n, _ in calls), f"named choice calls {calls}")


@case
def tool_multi_turn():
    first = chat(messages=user("What's the weather in Oslo?"), tools=[WEATHER], max_tokens=256,
                 **no_think())["choices"][0]["message"]
    tc = first["tool_calls"][0]
    msgs = user("What's the weather in Oslo?") + [
        {"role": "assistant", "content": first.get("content"), "tool_calls": first["tool_calls"]},
        {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"temp_c": -7, "sky": "snow"})}]
    final = chat(messages=msgs, tools=[WEATHER], max_tokens=256, **no_think())["choices"][0]["message"]
    # The reply must use the tool result (-7 C, snow), not just mention a 7.
    text = final["content"] or ""
    check(re.search(r"(-|\u2212|minus\s*)7\b", text, re.I) or "snow" in text.lower(), f"final {text!r}")


@case
def tool_call_streaming():
    ev = stream("/chat/completions", dict(messages=user("What's the weather in Lima?"), tools=[WEATHER],
                                          max_tokens=256, **no_think()))
    name, args = "", ""
    for e in ev:
        for tc in (e.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []:
            name += tc.get("function", {}).get("name") or ""
            args += tc.get("function", {}).get("arguments") or ""
    check(name == "get_weather", f"streamed name {name!r}")
    check("lima" in json.loads(args)["city"].lower(), f"streamed args {args!r}")


@case
def json_object_mode():
    m = chat(messages=user("Return a JSON object with keys city and country for the Eiffel Tower."),
             response_format={"type": "json_object"}, max_tokens=128, **no_think())["choices"][0]["message"]
    obj = json.loads(m["content"])
    check(isinstance(obj, dict) and "city" in obj, f"json {obj}")


@case
def json_schema_mode():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
              "required": ["name", "age"], "additionalProperties": False}
    m = chat(messages=user("Make up a person."), max_tokens=128,
             response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema, "strict": True}},
             **no_think())["choices"][0]["message"]
    obj = json.loads(m["content"])
    check(set(obj) == {"name", "age"} and isinstance(obj["name"], str) and isinstance(obj["age"], int), f"obj {obj}")


@case
def image_input():
    for rgb, word in (((220, 20, 20), "red"), ((20, 60, 220), "blue")):
        m = chat(messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": png(rgb)}},
            {"type": "text", "text": "What single colour fills this image? One word."}]}],
            max_tokens=16, **no_think())["choices"][0]["message"]
        check(word in (m["content"] or "").lower(), f"expected {word}, got {m['content']!r}")


@case
def long_context_needle():
    filler = "The sky was grey and the river ran slowly past the old mill. "
    root = ARGS.base_url.rstrip("/").removesuffix("/v1")
    req = urllib.request.Request(root + "/tokenize", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": ARGS.model, "prompt": filler, "add_special_tokens": False}).encode())
    if ARGS.api_key:
        req.add_header("Authorization", f"Bearer {ARGS.api_key}")
    with urllib.request.urlopen(req, timeout=60) as r:
        per = len(json.loads(r.read())["tokens"])  # exact tokens per filler sentence on this server
    reps = max(1, min(ARGS.long_tokens, MAX_MODEL_LEN - 512) // per)
    half = reps // 2
    text = filler * half + "The secret code is 48213. " + filler * (reps - half)
    out = chat(messages=user(text + "\n\nWhat is the secret code? Reply with the number only."), max_tokens=16,
               **no_think())
    check("48213" in out["choices"][0]["message"]["content"], f"needle answer {out['choices'][0]['message']['content']!r}")
    print(f"      prompt_tokens={out['usage']['prompt_tokens']}")


@case
def concurrent_requests():
    body = dict(messages=user("What is 17*23? Reply with the integer only."), max_tokens=16, **no_think())
    with cf.ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: chat(**body), range(8)))
    answers = [o["choices"][0]["message"]["content"] for o in outs]
    check(all("391" in a for a in answers), f"answers {answers}")


@case
def errors_are_clean():
    status, _ = call("POST", "/chat/completions", {"model": "no-such-model", "messages": user("hi")})
    check(status == 404, f"unknown model gave {status}")
    status, _ = call("POST", "/chat/completions", {"model": ARGS.model, "messages": user("hi"), "max_tokens": -1})
    check(status == 400, f"invalid max_tokens gave {status}")


@case
def api_key_enforced():
    if not ARGS.api_key:
        print("      no API key given: server runs unauthenticated")
        return
    status, _ = call("GET", "/models", key=None)
    check(status == 401, f"request without key gave {status}")
    status, _ = call("GET", "/models", key="wrong-key")
    check(status == 401, f"request with a wrong key gave {status}")


def main() -> int:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="default: env API_KEY")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--only", default="")
    p.add_argument("--long-tokens", type=int, default=16000)
    p.add_argument("--xfail", action="append", default=[], metavar="CASE:REASON",
                   help="documented known limit: the case still runs and is reported XFAIL (or XPASS), not FAIL")
    ARGS = p.parse_args()
    global MAX_MODEL_LEN
    status, models = call("GET", "/models")
    if status == 200:
        MAX_MODEL_LEN = next((m.get("max_model_len") for m in models["data"] if m["id"] == ARGS.model), None) or MAX_MODEL_LEN
    wanted = {s for s in ARGS.only.split(",") if s}
    xfail = dict(x.split(":", 1) if ":" in x else (x, "known limit") for x in ARGS.xfail)
    failed = xfailed = 0
    for fn in CASES:
        if wanted and fn.__name__ not in wanted:
            continue
        t = time.time()
        try:
            fn()
            tag = "XPASS" if fn.__name__ in xfail else "PASS"
            print(f"{tag} {fn.__name__} ({time.time() - t:.1f}s)")
        except Exception as e:  # noqa: BLE001 - every failure is reported, the suite keeps going
            detail = str(e) if isinstance(e, Fail) else traceback.format_exc(limit=2).strip()
            if fn.__name__ in xfail:
                xfailed += 1
                print(f"XFAIL {fn.__name__} ({time.time() - t:.1f}s) [{xfail[fn.__name__]}]: {detail}")
            else:
                failed += 1
                print(f"FAIL {fn.__name__} ({time.time() - t:.1f}s): {detail}")
    print(f"{'OK' if not failed else 'FAILED'}: {failed} failing case(s), {xfailed} expected failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
