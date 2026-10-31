#!/usr/bin/env bash
# Autoresearch benchmark: deploy the current tree on the four Sparks (the FABRIC in .env.example) and measure decode,
# prefill, boot and quality.
#
#   1. .env := .env.example (experiments change tracked files only).
#   2. If docker/ differs from the image last built by this harness: stop, ./dsv41.sh build (~20 min).
#   3. ./dsv41.sh restart, timed (boot_s = stop + start + smoke wall time; engine_up_s = start -> /health).
#   4. Decode: bench decode, four prompt types, c=1 and c=8, 30 s windows, greedy, thinking off.
#      Prefill: bench prefill at 32k, 128k, 512k (unique per-run nonce, no prefix-cache hits).
#   5. Quality gate (50 exact answers, must stay >= 48) and the thinking-mode corruption probe (reported only).
#
# Primary metric speedup_min_pct = min(decode gain, prefill gain) in % against the r6v record
# (bench/records/2026-10-18-r6v, v2.0.0 defaults): decode = mean over the four types of c=1 aggregate tok/s / ref,
# prefill = mean over the three lengths of tok/s / ref. Goal: >= 10 on both, i.e. speedup_min_pct >= 10.
set -euo pipefail
cd "$(dirname "$0")"
HEAD_HOST=spark@192.168.1.121
URL=http://192.168.1.121:8000
STATE='$HOME/.cache/dsv41-ds41f/autoresearch.docker.sha256'
RUN=bench/runs/autoresearch-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"

cp .env.example .env
bash tests/test-recipe.sh > "$RUN/contract.log" 2>&1 || { tail -20 "$RUN/contract.log"; echo "contract tests failed" >&2; exit 1; }

docker_hash=$(find docker -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-64)
built_hash=$(ssh -o BatchMode=yes "$HEAD_HOST" "cat $STATE 2>/dev/null || true")
if [[ $docker_hash != "$built_hash" ]]; then
  echo "docker/ changed ($built_hash -> $docker_hash): rebuilding the image"
  ./dsv41.sh stop > "$RUN/stop-for-build.log" 2>&1
  t=$SECONDS
  ./dsv41.sh build > "$RUN/build.log" 2>&1 || { tail -40 "$RUN/build.log"; echo "build failed" >&2; exit 1; }
  echo "METRIC build_s=$((SECONDS - t))"
  ssh -o BatchMode=yes "$HEAD_HOST" "mkdir -p \$HOME/.cache/dsv41-ds41f && echo $docker_hash > $STATE"
fi

t=$SECONDS
./dsv41.sh restart > "$RUN/restart.log" 2>&1 || { tail -60 "$RUN/restart.log"; echo "restart failed" >&2; exit 1; }
boot_s=$((SECONDS - t))
engine_up_s=$(grep -o 'up after [0-9]*s' "$RUN/restart.log" | grep -o '[0-9]*' | tail -1)
grep -q 'smoke: OK' "$RUN/restart.log" || { tail -20 "$RUN/restart.log"; echo "smoke failed" >&2; exit 1; }

# Warm-up, discarded: the first minutes after a boot decode ~1-2.5% slower than the rest of the boot (measured:
# +2.95% first pass vs +4.01/+4.07/+4.03% on three later passes of one boot), which is noise for these A/Bs.
./dsv41.sh bench decode --conc 1 --seconds 20 --out "$RUN/warmup-decode" > "$RUN/warmup.log" 2>&1 \
  && ./dsv41.sh bench prefill --lengths 32k --out "$RUN/warmup-prefill" >> "$RUN/warmup.log" 2>&1 \
  || { tail -30 "$RUN/warmup.log"; echo "warm-up failed" >&2; exit 1; }

./dsv41.sh bench decode --conc 1,8 --seconds 30 --out "$RUN/decode" > "$RUN/decode.log" 2>&1 \
  || { tail -30 "$RUN/decode.log"; echo "decode bench failed" >&2; exit 1; }
./dsv41.sh bench prefill --lengths 32k,128k,512k --out "$RUN/prefill" > "$RUN/prefill.log" 2>&1 \
  || { tail -30 "$RUN/prefill.log"; echo "prefill bench failed" >&2; exit 1; }
mem=$(for h in 192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18; do
        ssh -o BatchMode=yes spark@$h "awk '/MemAvailable/{printf \"%.1f\n\", \$2/1048576}' /proc/meminfo"; done | sort -n | head -1)
python3 bench/quality_gate.py "$URL" --out "$RUN/quality_answers.json" > "$RUN/quality.log" 2>&1 \
  || { cat "$RUN/quality.log"; echo "quality gate errored" >&2; exit 1; }
python3 bench/corruption_probe.py "$URL" > "$RUN/corruption.log" 2>&1 \
  || { cat "$RUN/corruption.log"; echo "corruption probe errored" >&2; exit 1; }

python3 - "$RUN" "$boot_s" "${engine_up_s:-0}" "$mem" <<'PY'
import json, re, sys
run, boot_s, up_s, mem = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
REF_DEC = {"prose": 61.39, "code": 87.23, "structured": 80.62, "json": 88.53}        # r6v c=1 aggregate tok/s
REF_DEC8 = {"prose": 274.8, "code": 422.7, "structured": 436.2, "json": 307.3}       # r6v c=8 aggregate tok/s
REF_PF = {32768: 6012.6, 131072: 5915.8, 524288: 5042.0}                            # r6v prefill tok/s
cells = json.load(open(f"{run}/decode/results.json"))["results"]["cells"]
errors = sum(c["errors"] for c in cells)
c1 = {c["prompt_type"]: c["agg_tok_s"] for c in cells if c["concurrency"] == 1}
c8 = {c["prompt_type"]: c["agg_tok_s"] for c in cells if c["concurrency"] == 8}
rows = {r["target"]: r["tok_s"] for r in json.load(open(f"{run}/prefill/results.json"))["results"]["rows"]}
dec = 100 * (sum(c1[t] / REF_DEC[t] for t in REF_DEC) / len(REF_DEC) - 1)
dec8 = 100 * (sum(c8[t] / REF_DEC8[t] for t in REF_DEC8) / len(REF_DEC8) - 1)
pf = 100 * (sum(rows[n] / REF_PF[n] for n in REF_PF) / len(REF_PF) - 1)
q = int(re.search(r"accuracy (\d+)/50", open(f"{run}/quality.log").read()).group(1))
bad = int(re.search(r"corrupt (\d+)/13", open(f"{run}/corruption.log").read()).group(1))
print(f"METRIC speedup_min_pct={min(dec, pf):.2f}")
print(f"METRIC decode_gain_pct={dec:.2f}")
print(f"METRIC prefill_gain_pct={pf:.2f}")
print(f"METRIC decode_c8_gain_pct={dec8:.2f}")
for t in REF_DEC:
    print(f"METRIC decode_c1_{t}={c1[t]:.1f}")
for n in REF_PF:
    print(f"METRIC prefill_{n // 1024}k={rows[n]:.0f}")
print(f"METRIC boot_s={boot_s}")
print(f"METRIC engine_up_s={up_s}")
print(f"METRIC mem_min_gib={mem:.1f}")
print(f"METRIC quality_gate={q}")
print(f"METRIC corrupt_offsets={bad}")
print(f"METRIC decode_errors={errors}")
if errors or q < 48:
    sys.exit(f"gate failed: decode errors {errors}, quality {q}/50 (need >= 48)")
PY
