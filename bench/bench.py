#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark and regression client for the DeepSeek-V4.1-Flash-1017 TP4 server (stdlib only).

Runs from the control machine against any OpenAI-compatible vLLM endpoint:

  python3 bench/bench.py [--base-url http://192.168.1.121:8000] [--model deepseek-v4.1-flash]
                         [--api-key K | env API_KEY=K] [--timeout S] <step> [step options] [--out DIR]

Steps (each writes results.json + summary.md to --out, default bench/runs/<YYYY-mm-dd-HHMMSS>-<step>/,
and prints the summary):

  decode   Concurrency sweep (--conc 1,2,4,8) x prompt type (prose, code, structured, json), thinking off,
           temperature 0, --max-tokens 512. Each cell is a fixed --seconds window (default 30) in which c workers
           issue back-to-back NON-streaming requests; no new request starts after the window closes and the cell
           ends when the last one returns. Aggregate tok/s = sum(usage.completion_tokens) / cell wall time.
           Per-stream tok/s = median over requests of usage.completion_tokens / request latency, so it includes the
           short prompt's prefill and any queueing (at c > max_num_seqs requests wait for a slot). TTFT is measured
           under load: once the c workers have run for 25% of the window, one extra streaming request is sent and
           timed to its first non-empty content (or reasoning) delta, i.e. the TTFT a new arrival sees while c
           streams decode (queueing included). Its tokens are not counted in the aggregate.
  prefill  One request per target length (--lengths 4k,...,1M; k=1024, M=1024*1024), sequential, max_tokens 1.
           Prompts are pseudo-random words seeded per length behind a per-run nonce, so nothing hits the prefix
           cache. Reports usage.prompt_tokens, wall time and prompt_tokens / wall time. Lengths above the server's
           max_model_len (GET /v1/models) are skipped; a length equal to it is trimmed to leave generation room.
  needle   A secret 5-digit code in the middle of unique filler at --tokens (default 1,000,000, capped at
           max_model_len - 256); PASS when the reply contains it. Reports prompt_tokens and time.
  spec     Scrapes GET /metrics before and after running each decode prompt at c=1 for --seconds and reports the
           DSpark acceptance: mean accepted tokens per draft step = delta(vllm:spec_decode_num_accepted_tokens_total)
           / delta(vllm:spec_decode_num_drafts_total), plus the draft-token acceptance rate and per-position counts
           when exported. Missing metrics are reported with the spec_decode names that do exist.
  pin      Token-id regression oracle: six fixed greedy prompts (thinking off, max_tokens 128, "return_token_ids":
           true) written with their exact request bodies to bench/reference.json (--reference).
  check    Replays reference.json's request bodies and reports the first divergence per prompt (token ids when both
           sides have them, else text). Exit 1 on any mismatch.
  all      decode, prefill, needle, spec, check in order into one run directory, continuing past failures; exit 1
           if any step failed.

Why tokens come from usage and never from counting SSE chunks: DSpark speculative decoding accepts several tokens
per verification step and vLLM emits all of them in one delta, so a chunk carries 1..k+1 tokens and chunk counts
undercount decode speed by roughly the acceptance length. usage.completion_tokens is the engine's own count.
Prompt lengths likewise come from usage.prompt_tokens, not from the word counts used to build the prompts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
THINK_OFF = {"chat_template_kwargs": {"thinking": False}}

DECODE_PROMPTS = {
    "prose": "Write a long, detailed essay on how the invention of the printing press changed European science, "
             "religion and politics over the following three centuries.",
    "code": "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, "
            "a decorator interface, statistics counters and an extensive unittest suite.",
    "structured": "Produce a detailed Markdown reference guide to TCP congestion control with headings, numbered "
                  "steps, bullet lists and a comparison table of Reno, CUBIC, BBR and Vegas.",
    "json": "Output a JSON array of 40 fictional library books, each an object with title, author, year, isbn, "
            "genres (array), page_count, rating and a two-sentence summary. Output only JSON.",
}

ORACLE_PROMPTS = [
    ("prose", "Describe a quiet harbour town at dawn in one vivid paragraph."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
    ("math", "Compute 37 * 43 step by step, then state the final answer."),
    ("json", "Return a JSON object with keys name, age and city for a fictional person. Output only JSON."),
    ("reasoning_off", "A farmer has 17 sheep; all but 9 run away. How many are left? Explain briefly."),
    ("summary", "Summarise the following in three sentences: The Industrial Revolution began in Britain in the "
                "late eighteenth century, driven by coal, iron and new textile machinery. Steam engines powered "
                "factories, railways and ships, reshaping where people lived and worked. Cities grew rapidly, "
                "bringing both prosperity and harsh working conditions, which in turn spurred labour movements, "
                "public health reforms and new political ideas across Europe and beyond."),
]

WORDS = ("amber basin cobalt delta ember fjord granite harbor island juniper kestrel lantern meadow nectar orchid "
         "pebble quartz river saffron thistle umber velvet willow xenon yarrow zephyr anchor bramble cinder dune "
         "eclipse falcon glacier heron indigo jasper kelp lagoon marble nimbus onyx prairie quiver rune sierra "
         "tundra upland vortex walnut yonder zenith arbor beacon canyon drift estuary fern grove hollow inlet "
         "jetty knoll ledge mesa nook oasis plume quay ridge shoal tide vale wharf").split()


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float):
        base = base_url.rstrip("/")
        self.root = base[:-3] if base.endswith("/v1") else base
        self.model, self.key, self.timeout = model, api_key, timeout

    def _open(self, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.root + path, data=data, method="GET" if body is None else "POST")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} {path}: {e.read()[:500].decode('utf-8', 'replace')}") from None

    def get_json(self, path: str) -> dict:
        with self._open(path) as r:
            return json.loads(r.read())

    def get_text(self, path: str) -> str:
        with self._open(path) as r:
            return r.read().decode("utf-8", "replace")

    def model_info(self) -> dict:
        data = self.get_json("/v1/models").get("data") or []
        for m in data:
            if m.get("id") == self.model:
                return m
        return data[0] if data else {}

    def chat(self, body: dict) -> tuple[dict, float]:
        """Non-streaming chat completion; returns (response, latency seconds)."""
        t0 = time.perf_counter()
        with self._open("/v1/chat/completions", dict(body, model=self.model, stream=False)) as r:
            resp = json.loads(r.read())
        return resp, time.perf_counter() - t0

    def ttft(self, body: dict) -> dict:
        """Streaming chat completion: time to the first non-empty content/reasoning delta, plus final usage."""
        body = dict(body, model=self.model, stream=True, stream_options={"include_usage": True})
        t0 = time.perf_counter()
        first, usage = None, None
        with self._open("/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                        first = time.perf_counter() - t0
        return {"ttft_s": first, "total_s": time.perf_counter() - t0, "usage": usage}

    def tokenize_count(self, text: str) -> int | None:
        """Prompt tokens of a single user message via vLLM's /tokenize, or None if unavailable."""
        try:
            with self._open("/tokenize", {"model": self.model, "messages": [{"role": "user", "content": text}],
                                          **THINK_OFF}) as r:
                return int(json.loads(r.read())["count"])
        except Exception:
            return None


def parse_count(s: str) -> int:
    s = s.strip().lower().replace("_", "")
    mult = {"k": 1024, "m": 1024 * 1024}.get(s[-1:], 1)
    return int(float(s[:-1] if mult > 1 else s) * mult)


def human(n: int) -> str:
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def content_of(resp: dict) -> str:
    return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


# --------------------------------------------------------------------------------------------- unique filler
def filler(n_words: int, seed: int, nonce: str) -> str:
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    for i in range(12, n_words, 13):          # sentence breaks keep the tokenizer in a natural regime
        words[i] += "."
    return f"[{nonce}] " + " ".join(words)


class Sizer:
    """Maps a target token count to a word count: /tokenize when the server has it, else one calibration request."""

    def __init__(self, cli: Client, nonce: str):
        self.cli, self.nonce, self.ratio, self.exact = cli, nonce, None, True

    def _calibrate(self) -> float:
        if self.ratio is None:
            n = 4000
            text = filler(n, seed=-1, nonce=self.nonce + "-cal")
            count = self.cli.tokenize_count(text)
            if count is None:
                self.exact = False
                resp, _ = self.cli.chat({"messages": [{"role": "user", "content": text}], "max_tokens": 1,
                                         "temperature": 0, **THINK_OFF})
                count = resp["usage"]["prompt_tokens"]
            self.ratio = count / n
        return self.ratio

    def build(self, target: int, seed: int, wrap) -> str:
        """Text whose wrap(filler) prompt is ~target tokens; refined once with /tokenize when available."""
        ratio = self._calibrate()
        safety = 1.0 if self.exact else 0.98
        words = max(16, int(target * safety / ratio))
        text = wrap(filler(words, seed, self.nonce))
        if self.exact:
            count = self.cli.tokenize_count(text)
            if count:
                words = max(16, int(words * target / count) - 8)
                text = wrap(filler(words, seed, self.nonce))
        return text


# --------------------------------------------------------------------------------------------- steps
def step_decode(cli: Client, a) -> tuple[dict, str, bool]:
    conc = [int(c) for c in a.conc.split(",")]
    types = [t for t in a.types.split(",")]
    cells, ok = [], True
    for ptype in types:
        for c in conc:
            body = {"messages": [{"role": "user", "content": DECODE_PROMPTS[ptype]}], "max_tokens": a.max_tokens,
                    "temperature": 0, **THINK_OFF}
            if a.ignore_eos:
                body["ignore_eos"] = True
            cell = run_window(cli, body, c, a.seconds, probe_at=0.25)
            tt = cell.pop("ttft")
            cell.update(prompt_type=ptype, concurrency=c, ttft=tt)
            ok &= cell["requests"] > 0 and cell["errors"] == 0
            cells.append(cell)
            print(f"  decode {ptype:<10} c={c:<3} agg {cell['agg_tok_s']:8.1f} tok/s  per-stream "
                  f"{cell['per_stream_tok_s'] or 0:7.1f}  ttft {fmt_s(tt.get('ttft_s'))}  "
                  f"reqs {cell['requests']} err {cell['errors']}", flush=True)
    res = {"settings": {"concurrency": conc, "prompt_types": types, "seconds": a.seconds,
                        "max_tokens": a.max_tokens, "ignore_eos": a.ignore_eos, "thinking": False,
                        "temperature": 0, "prompts": {t: DECODE_PROMPTS[t] for t in types}},
           "cells": cells}
    lines = ["## Decode", "", f"Aggregate tok/s (median per-stream tok/s = completion tokens / request latency, "
             f"prefill and queueing included); window {a.seconds}s per cell, "
             f"max_tokens {a.max_tokens}, thinking off, greedy. Tokens from usage.completion_tokens.", "",
             "| prompt | " + " | ".join(f"c={c}" for c in conc) + " |", "|---|" + "---:|" * len(conc)]
    for ptype in types:
        row = [cell for cell in cells if cell["prompt_type"] == ptype]
        lines.append(f"| {ptype} | " + " | ".join(
            f"{x['agg_tok_s']:.1f} ({x['per_stream_tok_s'] or 0:.1f})" for x in row) + " |")
    lines += ["", "TTFT under load (one extra streaming request sent 25% into each cell's window while the c "
              "workers run; seconds, queueing included):", "",
              "| prompt | " + " | ".join(f"c={c}" for c in conc) + " |", "|---|" + "---:|" * len(conc)]
    for ptype in types:
        row = [cell for cell in cells if cell["prompt_type"] == ptype]
        lines.append(f"| {ptype} | " + " | ".join(fmt_s(x["ttft"].get("ttft_s")) for x in row) + " |")
    return res, "\n".join(lines), ok


def run_window(cli: Client, body: dict, c: int, seconds: float, probe_at: float | None = None) -> dict:
    """c workers send back-to-back requests for `seconds`. With probe_at, one extra streaming request starts at
    that fraction of the window and its TTFT is returned under "ttft" (measured while the c streams decode)."""
    lock = threading.Lock()
    recs, errors = [], []
    t_start = time.perf_counter()
    deadline = t_start + seconds
    tt = {}

    def probe():
        time.sleep(max(0.0, t_start + probe_at * seconds - time.perf_counter()))
        try:
            tt.update(cli.ttft(body))
        except Exception as e:
            tt.update(ttft_s=None, error=str(e)[:300])

    prober = threading.Thread(target=probe, daemon=True) if probe_at is not None else None

    def worker():
        while time.perf_counter() < deadline:
            try:
                resp, lat = cli.chat(body)
                u = resp.get("usage") or {}
                with lock:
                    recs.append({"completion_tokens": u.get("completion_tokens", 0),
                                 "prompt_tokens": u.get("prompt_tokens", 0), "latency_s": lat,
                                 "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason")})
            except Exception as e:
                with lock:
                    errors.append(str(e)[:300])
                time.sleep(0.5)

    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        futures = [ex.submit(worker) for _ in range(c)]
        if prober:
            prober.start()
        for f in futures:
            f.result()
    elapsed = time.perf_counter() - t_start
    if prober:
        prober.join()
    toks = sum(r["completion_tokens"] for r in recs)
    rates = [r["completion_tokens"] / r["latency_s"] for r in recs if r["latency_s"] > 0]
    cell = {"requests": len(recs), "errors": len(errors), "error_samples": errors[:3], "elapsed_s": elapsed,
            "completion_tokens": toks, "agg_tok_s": toks / elapsed if elapsed else 0.0,
            "per_stream_tok_s": statistics.median(rates) if rates else None,
            "mean_completion_tokens": toks / len(recs) if recs else None, "records": recs}
    if prober:
        cell["ttft"] = tt
    return cell


def fmt_s(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def step_prefill(cli: Client, a) -> tuple[dict, str, bool]:
    info = cli.model_info()
    mml = info.get("max_model_len")
    nonce = uuid.uuid4().hex[:12]
    sizer = Sizer(cli, nonce)
    rows, ok = [], True
    for target in [parse_count(x) for x in a.lengths.split(",")]:
        row = {"target": target}
        if mml and target > mml:
            row["skipped"] = f"above max_model_len {mml}"
            rows.append(row)
            print(f"  prefill {human(target):>5} skipped ({row['skipped']})", flush=True)
            continue
        goal = min(target, mml - 512) if mml else target
        try:
            text = sizer.build(goal, seed=target, wrap=lambda f: f + "\n\nReply with the single word OK.")
            resp, wall = cli.chat({"messages": [{"role": "user", "content": text}], "max_tokens": 1,
                                   "temperature": 0, **THINK_OFF})
            pt = (resp.get("usage") or {}).get("prompt_tokens", 0)
            row.update(prompt_tokens=pt, seconds=wall, tok_s=pt / wall if wall else None)
            print(f"  prefill {human(target):>5} {pt:>9} tok {wall:9.2f}s {row['tok_s']:9.1f} tok/s", flush=True)
        except Exception as e:
            ok = False
            row["error"] = str(e)[:500]
            print(f"  prefill {human(target):>5} ERROR {row['error'][:200]}", flush=True)
        rows.append(row)
    res = {"settings": {"lengths": a.lengths, "max_model_len": mml, "nonce": nonce, "max_tokens": 1,
                        "tokens_per_word": sizer.ratio, "sized_with_tokenize": sizer.exact}, "rows": rows}
    lines = ["## Prefill", "", "Unique prompts (no prefix-cache hits), max_tokens 1, sequential. "
             "Tokens from usage.prompt_tokens.", "", "| target | prompt_tokens | seconds | tok/s |",
             "|---|---:|---:|---:|"]
    for r in rows:
        if "prompt_tokens" in r:
            lines.append(f"| {human(r['target'])} | {r['prompt_tokens']} | {r['seconds']:.2f} | {r['tok_s']:.1f} |")
        else:
            lines.append(f"| {human(r['target'])} | - | - | {r.get('skipped') or 'ERROR: ' + r['error'][:80]} |")
    return res, "\n".join(lines), ok


def step_needle(cli: Client, a) -> tuple[dict, str, bool]:
    mml = cli.model_info().get("max_model_len")
    target = min(a.tokens, mml - 256) if mml else a.tokens
    nonce = uuid.uuid4().hex[:12]
    code = f"{random.Random(nonce).randint(10000, 99999)}"
    needle = f" The secret code is {code}. Remember it. "
    question = "\n\nWhat is the secret code mentioned in the text above? Answer with only the 5-digit number."

    def wrap(f: str) -> str:
        mid = len(f) // 2
        mid = f.rfind(" ", 0, mid) if " " in f[:mid] else mid
        return f[:mid] + needle + f[mid:] + question

    res = {"settings": {"tokens": a.tokens, "target": target, "max_model_len": mml, "nonce": nonce,
                        "code": code, "max_tokens": 16}}
    try:
        text = Sizer(cli, nonce).build(target, seed=target + 7, wrap=wrap)
        resp, wall = cli.chat({"messages": [{"role": "user", "content": text}], "max_tokens": 16,
                               "temperature": 0, **THINK_OFF})
        answer = content_of(resp)
        res.update(prompt_tokens=(resp.get("usage") or {}).get("prompt_tokens"), seconds=wall,
                   answer=answer, passed=code in answer)
    except Exception as e:
        res.update(passed=False, error=str(e)[:500])
    verdict = "PASS" if res["passed"] else "FAIL"
    print(f"  needle {verdict} prompt_tokens={res.get('prompt_tokens')} {fmt_s(res.get('seconds'))}s "
          f"answer={res.get('answer', res.get('error'))!r}", flush=True)
    lines = ["## Needle", "", "| result | target | prompt_tokens | seconds | expected | answer |",
             "|---|---:|---:|---:|---|---|",
             f"| {verdict} | {human(target)} | {res.get('prompt_tokens', '-')} | {fmt_s(res.get('seconds'))} | "
             f"{code} | {str(res.get('answer', res.get('error', ''))).strip()[:60].replace('|', '/')} |"]
    return res, "\n".join(lines), res["passed"]


METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def scrape(cli: Client) -> dict[str, float]:
    """Prometheus text -> {name{labels}: value}; label sets kept so per-position series stay separate."""
    out = {}
    for line in cli.get_text("/metrics").splitlines():
        m = METRIC_RE.match(line)
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def metric_sum(snap: dict, name: str) -> float | None:
    vals = [v for k, v in snap.items() if k == name or k.startswith(name + "{")]
    return sum(vals) if vals else None


def acceptance(before: dict, after: dict) -> dict:
    d = {}
    for key in ("accepted", "drafts", "draft_tokens"):
        name = {"accepted": "vllm:spec_decode_num_accepted_tokens_total", "drafts": "vllm:spec_decode_num_drafts_total",
                "draft_tokens": "vllm:spec_decode_num_draft_tokens_total"}[key]
        b, e = metric_sum(before, name), metric_sum(after, name)
        d[key] = None if b is None or e is None else e - b
    d["mean_accepted_per_draft"] = d["accepted"] / d["drafts"] if d["accepted"] is not None and d["drafts"] else None
    d["acceptance_rate"] = (d["accepted"] / d["draft_tokens"]
                            if d["accepted"] is not None and d["draft_tokens"] else None)
    per_pos = {}
    for k, v in after.items():
        if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
            pos = re.search(r'position="(\d+)"', k)
            if pos:
                per_pos[int(pos.group(1))] = per_pos.get(int(pos.group(1)), 0.0) + v - before.get(k, 0.0)
    d["accepted_per_position"] = {str(p): per_pos[p] for p in sorted(per_pos)}
    return d


def step_spec(cli: Client, a) -> tuple[dict, str, bool]:
    types = a.types.split(",")
    before_all = scrape(cli)
    names = sorted({k.split("{")[0] for k in before_all if "spec_decode" in k})
    rows = []
    for ptype in types:
        body = {"messages": [{"role": "user", "content": DECODE_PROMPTS[ptype]}], "max_tokens": a.max_tokens,
                "temperature": 0, **THINK_OFF}
        b = scrape(cli)
        cell = run_window(cli, body, 1, a.seconds)
        acc = acceptance(b, scrape(cli))
        rows.append({"prompt_type": ptype, "tok_s": cell["agg_tok_s"], "requests": cell["requests"],
                     "errors": cell["errors"], **acc})
        print(f"  spec {ptype:<10} {cell['agg_tok_s']:7.1f} tok/s  accepted/draft "
              f"{fmt_s(acc['mean_accepted_per_draft'])}  rate {fmt_s(acc['acceptance_rate'])}", flush=True)
    total = acceptance(before_all, scrape(cli))
    ok = total["mean_accepted_per_draft"] is not None and all(r["errors"] == 0 for r in rows)
    res = {"settings": {"seconds": a.seconds, "max_tokens": a.max_tokens, "concurrency": 1, "prompt_types": types},
           "spec_metric_names": names, "rows": rows, "total": total}
    lines = ["## Speculative decoding (DSpark) acceptance", "",
             "c=1, thinking off, greedy. accepted/draft = delta(vllm:spec_decode_num_accepted_tokens_total) / "
             "delta(vllm:spec_decode_num_drafts_total); tokens per step = 1 + accepted/draft.", "",
             "| prompt | tok/s | accepted/draft | tokens/step | draft-token acceptance |", "|---|---:|---:|---:|---:|"]
    for r in rows + [dict(total, prompt_type="**all**", tok_s=None)]:
        m, tok_s = r["mean_accepted_per_draft"], r["tok_s"]
        lines.append(f"| {r['prompt_type']} | {'-' if tok_s is None else f'{tok_s:.1f}'} | "
                     f"{fmt_s(m)} | {fmt_s(None if m is None else 1 + m)} | {fmt_s(r['acceptance_rate'])} |")
    if total["accepted_per_position"]:
        lines += ["", "Accepted tokens per draft position: " + ", ".join(
            f"{p}: {v:.0f}" for p, v in total["accepted_per_position"].items())]
    if total["mean_accepted_per_draft"] is None:
        lines += ["", "Acceptance metrics missing. spec_decode metric names exposed: "
                  + (", ".join(names) or "none")]
    return res, "\n".join(lines), ok


# --------------------------------------------------------------------------------------------- oracle
def oracle_body(prompt: str) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0,
            "return_token_ids": True, **THINK_OFF}


def oracle_run(cli: Client, body: dict) -> dict:
    resp, wall = cli.chat(body)
    ch = (resp.get("choices") or [{}])[0]
    return {"token_ids": ch.get("token_ids"), "text": (ch.get("message") or {}).get("content") or "",
            "finish_reason": ch.get("finish_reason"), "usage": resp.get("usage"), "seconds": wall}


def step_pin(cli: Client, a) -> tuple[dict, str, bool]:
    info = cli.model_info()
    ref = {"model": cli.model, "served": {k: info.get(k) for k in ("id", "root", "revision", "max_model_len")
                                          if k in info},
           "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prompts": []}
    for name, prompt in ORACLE_PROMPTS:
        body = oracle_body(prompt)
        out = oracle_run(cli, body)
        ref["prompts"].append({"name": name, "request": body, **{k: out[k] for k in ("token_ids", "text",
                                                                                      "finish_reason")}})
        print(f"  pin {name:<14} {len(out['token_ids'] or [])} ids  {out['text'][:60]!r}", flush=True)
    with open(a.reference, "w") as f:
        json.dump(ref, f, indent=1)
        f.write("\n")
    lines = ["## Oracle pinned", "", f"Wrote {a.reference}", "", "| prompt | ids | finish |", "|---|---:|---|"]
    lines += [f"| {p['name']} | {len(p['token_ids']) if p['token_ids'] is not None else 'text only'} | "
              f"{p['finish_reason']} |" for p in ref["prompts"]]
    return {"reference": a.reference, "pinned": ref}, "\n".join(lines), True


def first_divergence(ref, got) -> int | None:
    for i, (x, y) in enumerate(zip(ref, got)):
        if x != y:
            return i
    return None if len(ref) == len(got) else min(len(ref), len(got))


def step_check(cli: Client, a) -> tuple[dict, str, bool]:
    with open(a.reference) as f:
        ref = json.load(f)
    rows = []
    for p in ref["prompts"]:
        row = {"name": p["name"]}
        try:
            out = oracle_run(cli, p["request"])
            if p.get("token_ids") is not None and out["token_ids"] is not None:
                row["compared"] = "token_ids"
                div = first_divergence(p["token_ids"], out["token_ids"])
                if div is not None:
                    row.update(ref_ids=p["token_ids"][div:div + 8], got_ids=out["token_ids"][div:div + 8])
            else:
                row["compared"] = "text"
                div = first_divergence(p["text"], out["text"])
                if div is not None:
                    row.update(ref_text=p["text"][max(0, div - 20):div + 40],
                               got_text=out["text"][max(0, div - 20):div + 40])
            row.update(divergence=div, match=div is None)
        except Exception as e:
            row.update(match=False, divergence=None, error=str(e)[:500])
        rows.append(row)
        print(f"  check {p['name']:<14} {'MATCH' if row['match'] else 'MISMATCH'} ({row.get('compared', 'error')})"
              + ("" if row["match"] else f" first divergence at {row['divergence']} {row.get('error', '')}"),
              flush=True)
    ok = all(r["match"] for r in rows)
    lines = ["## Token-id regression check", "", f"Reference: {a.reference} (pinned {ref.get('date')}, "
             f"served {ref.get('served', {}).get('root') or ref.get('model')})", "",
             "| prompt | compared | result | first divergence |", "|---|---|---|---:|"]
    lines += [f"| {r['name']} | {r.get('compared', '-')} | {'MATCH' if r['match'] else 'MISMATCH'} | "
              f"{'-' if r['divergence'] is None else r['divergence']} |" for r in rows]
    return {"reference": a.reference, "rows": rows}, "\n".join(lines), ok


# --------------------------------------------------------------------------------------------- driver
STEPS = {"decode": step_decode, "prefill": step_prefill, "needle": step_needle, "spec": step_spec,
         "pin": step_pin, "check": step_check}


def write_out(out: str, step: str, cli: Client, args, started: str, res: dict, summary: str, ok: bool) -> None:
    os.makedirs(out, exist_ok=True)
    meta = {"step": step, "ok": ok, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "base_url": cli.root, "model": cli.model,
            "args": {k: v for k, v in vars(args).items() if k not in ("api_key", "func")}}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({**meta, "results": res}, f, indent=1)
        f.write("\n")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(summary + "\n")


def run_step(step: str, cli: Client, args, out: str) -> bool:
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"== {step} -> {out}", flush=True)
    try:
        res, summary, ok = STEPS[step](cli, args)
    except Exception as e:
        res, summary, ok = {"error": str(e)[:1000]}, f"## {step}\n\nFAILED: {e}", False
    summary += f"\n\n_{step}: {'ok' if ok else 'FAILED'} ({started})_"
    write_out(out, step, cli, args, started, res, summary, ok)
    print("\n" + summary + "\n", flush=True)
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Run '<step> --help' for step options. Protocol details: module docstring.")
    p.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://192.168.1.121:8000"))
    p.add_argument("--model", default=os.environ.get("SERVED_NAME", "deepseek-v4.1-flash"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY") or None)
    p.add_argument("--timeout", type=float, default=7200.0, help="per-request timeout seconds (default 7200)")
    sub = p.add_subparsers(dest="step", required=True)

    def add(name, help_):
        s = sub.add_parser(name, help=help_, description=help_)
        s.add_argument("--out", help="output directory (default bench/runs/<stamp>-<step>/)")
        return s

    def decode_opts(s):
        s.add_argument("--seconds", type=float, default=30.0, help="window per cell (default 30)")
        s.add_argument("--max-tokens", type=int, default=512)
        s.add_argument("--types", default=",".join(DECODE_PROMPTS), help="prompt types (default all four)")

    def ref_opt(s):
        s.add_argument("--reference", default=os.path.join(HERE, "reference.json"))

    def prefill_opts(s):
        s.add_argument("--lengths", default="4k,16k,32k,64k,128k,256k,512k,1M",
                       help="target prompt tokens, k=1024 M=1048576 (default 4k..1M)")

    def needle_opts(s):
        s.add_argument("--tokens", type=parse_count, default=1_000_000,
                       help="needle prompt tokens (default 1000000, capped at max_model_len-256)")

    s = add("decode", "concurrency x prompt-type decode sweep with TTFT")
    decode_opts(s)
    s.add_argument("--conc", default="1,2,4,8", help="concurrency levels (default 1,2,4,8)")
    s.add_argument("--ignore-eos", action="store_true", help="vLLM ignore_eos: every request runs to max_tokens")
    prefill_opts(add("prefill", "sequential unique-prompt prefill sweep"))
    needle_opts(add("needle", "needle-in-a-haystack retrieval at long context"))
    decode_opts(add("spec", "DSpark acceptance from /metrics deltas at c=1"))
    ref_opt(add("pin", "write the token-id regression oracle"))
    ref_opt(add("check", "compare against the token-id regression oracle"))
    s = add("all", "decode, prefill, needle, spec, check; continue past failures")
    decode_opts(s)
    prefill_opts(s)
    needle_opts(s)
    ref_opt(s)
    s.add_argument("--conc", default="1,2,4,8")
    s.add_argument("--ignore-eos", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cli = Client(args.base_url, args.model, args.api_key, args.timeout)
    out = args.out or os.path.join(HERE, "runs", f"{time.strftime('%Y-%m-%d-%H%M%S')}-{args.step}")
    if args.step != "all":
        return 0 if run_step(args.step, cli, args, out) else 1
    results = {s: run_step(s, cli, args, os.path.join(out, s)) for s in ("decode", "prefill", "needle", "spec",
                                                                          "check")}
    parts = []
    for s in results:
        with open(os.path.join(out, s, "summary.md")) as f:
            parts.append(f.read().strip())
    status = " | ".join(f"{s}: {'ok' if v else 'FAILED'}" for s, v in results.items())
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("# Benchmark run\n\n" + status + "\n\n" + "\n\n".join(parts) + "\n")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"step": "all", "steps": results, "base_url": cli.root, "model": cli.model}, f, indent=1)
        f.write("\n")
    print("== all: " + status, flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
