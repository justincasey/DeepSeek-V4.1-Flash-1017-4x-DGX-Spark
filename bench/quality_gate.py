#!/usr/bin/env python3
"""Fixed 50-question exact-answer gate (40 seeded multiplications + 10 capitals), thinking off, temperature 0.

usage: quality_gate.py BASE_URL [--model NAME] [--out answers.json]
Prints `accuracy N/50`; v1.0.0 and r6v both score 49/50 with identical answers (question 29 is the shared miss).
"""
import argparse
import concurrent.futures as cf
import json
import random
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("base_url")
p.add_argument("--model", default="deepseek-v4.1-flash")
p.add_argument("--out")
args = p.parse_args()
url = args.base_url.rstrip("/") + "/v1/chat/completions"

rng = random.Random(1017)
items = []
for _ in range(40):
    a, b = rng.randint(12, 999), rng.randint(12, 99)
    items.append((f"What is {a}*{b}? Reply with the integer only.", str(a * b)))
for c, cap in [("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
               ("Brazil", "Brasília"), ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Egypt", "Cairo"), ("Peru", "Lima"),
               ("Vietnam", "Hanoi")]:
    items.append((f"What is the capital of {c}? Reply with the city name only.", cap))


def ask(q):
    body = {"model": args.model, "messages": [{"role": "user", "content": q}], "max_tokens": 24, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read())["choices"][0]["message"]["content"].strip()


with cf.ThreadPoolExecutor(4) as ex:
    answers = list(ex.map(lambda it: ask(it[0]), items))
ok = sum(1 for (q, a), got in zip(items, answers) if a.replace(",", "") in got.replace(",", ""))
print(f"accuracy {ok}/{len(items)}")
if args.out:
    json.dump(answers, open(args.out, "w"))
