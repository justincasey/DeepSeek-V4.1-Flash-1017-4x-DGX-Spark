#!/usr/bin/env bash
# Autoresearch benchmark: deploy the current tree on the four-Spark ring and measure it with Christopher Owen's
# servebench (spark-ds41f 2d2b2328, vendored unmodified in bench/servebench, sha256-pinned), so our numbers compare
# with his: distinct prompts per stream (bench.py decode sends one prompt to every concurrent worker).
#
#   1. .env := .env.example (experiments change tracked files only).
#   2. If docker/ differs from the image last built by this harness: stop, ./dsv41.sh build (~20 min).
#   3. ./dsv41.sh restart, timed (boot_s = stop + start + smoke wall time; engine_up_s = start -> /health).
#   4. The served MAX_NUM_BATCHED_TOKENS, long-prefill threshold and CED non-final skip are read from the running
#      head container (docker inspect), not from .env.
#   5. servebench suite bench/suites/dsv41-ladder.json, run on the head node against 127.0.0.1 with its system
#      python3 (the prefill documents are that interpreter's standard-library source, as on Chris's head node):
#      closed loops c16, c8, c4, c2, c1 (60 s warm-up, 300 s measured), cold prefill 32K-512K (3 repeats) and 8K-16K
#      (6 repeats: a 1.4 s 8K sample swung 8% in the first run).
#      About 42 minutes.
#   6. Quality gate (50 exact answers, must stay >= 48) and the thinking-mode corruption probe (reported only).
#
# Primary metric prefill_tps_gmean: geometric mean of servebench cold-prefill tok/s over the six sizes (the path
# MAX_NUM_BATCHED_TOKENS and the CED skip act on). ced_skipped_chunks_* are derived from the recorded prompt
# lengths with docker/0005's rule (chunk ends on a threshold multiple, >= 200 rows follow at k=6), not observed.
# vs_chris_r6v_* compare with Chris's r6v release run on his ring (experiments/2026-10-17-r6v, same suite args).
set -euo pipefail
cd "$(dirname "$0")"
HEAD=192.168.1.121
HEAD_HOST=spark@$HEAD
URL=http://$HEAD:8000
SERVEBENCH_REV=2d2b2328a92a9dca9a73ce7acd56e188741f30a8
REMOTE='.cache/dsv41-ds41f/servebench'
STATE='$HOME/.cache/dsv41-ds41f/autoresearch.docker.sha256'
ID=autoresearch-$(date +%Y%m%d-%H%M%S)
RUN=bench/runs/$ID
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
mkdir -p "$RUN"

cp .env.example .env
bash tests/test-recipe.sh > "$RUN/contract.log" 2>&1 || { tail -20 "$RUN/contract.log"; echo "contract tests failed" >&2; exit 1; }
shasum -a 256 -c --quiet bench/servebench.sha256 || { echo "bench/servebench differs from spark-ds41f@$SERVEBENCH_REV" >&2; exit 1; }

docker_hash=$(find docker -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-64)
built_hash=$("${SSH[@]}" "$HEAD_HOST" "cat $STATE 2>/dev/null || true")
if [[ $docker_hash != "$built_hash" ]]; then
  echo "docker/ changed ($built_hash -> $docker_hash): rebuilding the image"
  ./dsv41.sh stop > "$RUN/stop-for-build.log" 2>&1
  t=$SECONDS
  ./dsv41.sh build > "$RUN/build.log" 2>&1 || { tail -40 "$RUN/build.log"; echo "build failed" >&2; exit 1; }
  echo "METRIC build_s=$((SECONDS - t))"
  "${SSH[@]}" "$HEAD_HOST" "mkdir -p \$HOME/.cache/dsv41-ds41f && echo $docker_hash > $STATE"
fi

t=$SECONDS
./dsv41.sh restart > "$RUN/restart.log" 2>&1 || { tail -60 "$RUN/restart.log"; echo "restart failed" >&2; exit 1; }
boot_s=$((SECONDS - t))
engine_up_s=$(grep -o 'up after [0-9]*s' "$RUN/restart.log" | grep -o '[0-9]*' | tail -1)
grep -q 'smoke: OK' "$RUN/restart.log" || { tail -20 "$RUN/restart.log"; echo "smoke failed" >&2; exit 1; }

# What the head rank actually serves (every rank gets the same args and env from dsv41.sh).
"${SSH[@]}" "$HEAD_HOST" 'docker inspect dsv41-1017 --format "{{json .Args}}{{\"\n\"}}{{json .Config.Env}}"; python3 --version' \
  > "$RUN/served.txt"
python3 - "$RUN" "$docker_hash" "$SERVEBENCH_REV" "$(git rev-parse HEAD)" "$(git status --porcelain | wc -l | tr -d ' ')" <<'PY'
import json, sys
run, docker_hash, rev, commit, dirty = sys.argv[1:]
lines = open(f"{run}/served.txt").read().splitlines()
args, env = json.loads(lines[0]), dict(e.split("=", 1) for e in json.loads(lines[1]))
def arg(name):
    return args[args.index(name) + 1]
spec = json.loads(arg("--speculative-config"))
meta = {"recipe_commit": commit, "recipe_dirty_paths": int(dirty), "docker_sha256": docker_hash,
        "servebench_rev": rev, "client_python": lines[2].split()[-1],
        "max_num_batched_tokens": int(arg("--max-num-batched-tokens")),
        "long_prefill_token_threshold": int(arg("--long-prefill-token-threshold")),
        "max_num_seqs": int(arg("--max-num-seqs")),
        "num_speculative_tokens": int(spec["num_speculative_tokens"]),
        "ced_skip_nonfinal": int(env.get("DSV41_CED_SKIP_NONFINAL", "1") == "1")}
json.dump(meta, open(f"{run}/metadata.json", "w"), indent=1)
PY

rsync -a --delete --exclude __pycache__ bench/servebench bench/suites "$RUN/metadata.json" "$HEAD_HOST:$REMOTE/"
sb=0
"${SSH[@]}" "$HEAD_HOST" "cd $REMOTE && python3 -m servebench suite http://127.0.0.1:8000 --suite suites/dsv41-ladder.json \
  --out runs/$ID --label $ID --metadata metadata.json" > "$RUN/servebench.log" 2>&1 || sb=$?
rsync -a "$HEAD_HOST:$REMOTE/runs/$ID/" "$RUN/servebench/" || true
(( sb == 0 )) || { tail -30 "$RUN/servebench.log"; echo "servebench suite failed (exit $sb)" >&2; exit 1; }

mem=$(for h in 192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18; do
        "${SSH[@]}" spark@$h "awk '/MemAvailable/{printf \"%.1f\n\", \$2/1048576}' /proc/meminfo"; done | sort -n | head -1)
python3 bench/quality_gate.py "$URL" --out "$RUN/quality_answers.json" > "$RUN/quality.log" 2>&1 \
  || { cat "$RUN/quality.log"; echo "quality gate errored" >&2; exit 1; }
python3 bench/corruption_probe.py "$URL" > "$RUN/corruption.log" 2>&1 \
  || { cat "$RUN/corruption.log"; echo "corruption probe errored" >&2; exit 1; }

python3 - "$RUN" "$boot_s" "${engine_up_s:-0}" "$mem" <<'PY'
import json, math, re, sys
run, boot_s, up_s, mem = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
sb = f"{run}/servebench"
meta = json.load(open(f"{run}/metadata.json"))
# Chris's r6v release (experiments/2026-10-17-r6v, TP4 ring, 2026-10-17): release c16/c8, single-stream c1, prefill.
CHRIS_R6V = {"c16": 243.770, "c8": 185.443, "c1": 71.05}
CHRIS_R6V_PF = {32768: 5982.405, 131072: 5946.093, 262144: 5676.088, 524288: 5170.933}
MIN_REMAINING = 200  # docker/0005: max(CED window, DSpark SWA index width) + 1 + (1 + k) rows, at k=6

def metric(name, value, fmt=".2f"):
    print(f"METRIC {name}={value:{fmt}}" if value is not None else f"METRIC {name}=nan")

pf = {}
for step in ("prefill", "prefill-short"):
    for row in json.load(open(f"{sb}/{step}.json"))["prefill"]:
        pf[row["size"]] = row
gmean = math.exp(sum(math.log(r["tps"]["mean"]) for r in pf.values()) / len(pf))
metric("prefill_tps_gmean", gmean, ".1f")

threshold, ced = meta["long_prefill_token_threshold"], meta["ced_skip_nonfinal"]
for size in sorted(pf):
    r, tag = pf[size], f"{size // 1024}k"
    metric(f"prefill_{tag}", r["tps"]["mean"], ".1f")
    metric(f"prefill_{tag}_ci95_pct", r["tps"]["ci95_pct"])
    p = r["prompt_tokens"]
    chunks = -(-p // threshold)
    skipped = sum(1 for i in range(1, chunks) if p - i * threshold >= MIN_REMAINING) \
        if ced and meta["num_speculative_tokens"] == 6 else 0
    metric(f"prefill_chunks_{tag}", chunks, "d")
    metric(f"ced_skipped_chunks_{tag}", skipped, "d")

decode_errors = foreign = 0
for c in (1, 2, 4, 8, 16):
    s = json.load(open(f"{sb}/c{c}.json"))["summary"]
    server, lat = s.get("server", {}), s["latency"]
    decode_errors += s["requests"]["failed"]
    foreign += s["requests"]["foreign"] or 0
    metric(f"c{c}_tps", s["output_tps"]["mean"])
    metric(f"c{c}_ci95_pct", s["output_tps"]["ci95_pct"])
    metric(f"c{c}_steps_per_s", server.get("steps_per_s", {}).get("mean"))
    metric(f"c{c}_tokens_per_step", server.get("output_tokens_per_step", {}).get("mean"), ".3f")
    metric(f"c{c}_ttft_p50_ms", 1000 * lat["ttft_s"]["p50"], ".1f")
    metric(f"c{c}_tpot_p50_ms", lat["tpot_ms"]["p50"])
    if f"c{c}" in CHRIS_R6V:
        metric(f"vs_chris_r6v_c{c}_pct", 100 * (s["output_tps"]["mean"] / CHRIS_R6V[f"c{c}"] - 1))
for size, ref in CHRIS_R6V_PF.items():
    metric(f"vs_chris_r6v_prefill_{size // 1024}k_pct", 100 * (pf[size]["tps"]["mean"] / ref - 1))

q = int(re.search(r"accuracy (\d+)/50", open(f"{run}/quality.log").read()).group(1))
bad = int(re.search(r"corrupt (\d+)/13", open(f"{run}/corruption.log").read()).group(1))
metric("max_num_batched_tokens", meta["max_num_batched_tokens"], "d")
metric("long_prefill_threshold", threshold, "d")
metric("ced_skip", ced, "d")
metric("boot_s", boot_s, "d")
metric("engine_up_s", up_s, "d")
metric("mem_min_gib", mem, ".1f")
metric("quality_gate", q, "d")
metric("corrupt_offsets", bad, "d")
metric("decode_errors", decode_errors, "d")
if decode_errors or foreign or q < 48:
    sys.exit(f"gate failed: request errors {decode_errors}, foreign requests {foreign}, quality {q}/50 (need >= 48)")
PY
