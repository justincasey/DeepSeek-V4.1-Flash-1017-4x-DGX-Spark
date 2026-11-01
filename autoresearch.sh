#!/usr/bin/env bash
# Autoresearch benchmark, switched fabric: deploy the current tree on the four Sparks behind the CRS804 switch
# (FABRIC=switched from .env.example) and measure decode, prefill, fabric bytes, boot and exactness.
#
#   1. .env := .env.example (experiments change tracked files only); static contract tests.
#   2. If docker/ differs from the image this harness last built under IMAGE's tag: ./dsv41.sh stop, build.
#   3. ./dsv41.sh restart, timed.
#   4. Warm-up pass, discarded (decode is 1-2.5% slower in the first minutes after a boot).
#   5. Decode c=1, four prompt types, 30 s windows, greedy, thinking off (c=8 is boot-bimodal, TD-010).
#   6. Prefill 32k, 128k, 512k (unique per-run nonce), with the switch's per-port tx byte and drop counters
#      read around it: prefill_fabric_gb is what every port sent during the prefill bench.
#   7. Gates (exit 1 on any): decode errors 0, quality gate >= 48/50, token-id oracle 6/6 (short prompts,
#      bench/reference.json pinned on r6v), long-prompt oracle 3/3 (bench/long_oracle.py: ~6K/46K/109K-token
#      prompts, token ids and top-5 logprobs bit-identical to bench/long_reference.json, pinned on the relayed
#      switched build), switch drops 0. The thinking-mode corruption probe is reported only (TD-009).
#
# Primary metric switched_score_pct = 100 * (sqrt(decode ratio * prefill ratio) - 1) against the relayed
# switched record (bench/records/2026-10-31-switched): decode ratio = mean over the four types of c=1 aggregate
# tok/s / ref, prefill ratio = mean over the three lengths of tok/s / ref. Higher is better.
set -euo pipefail
cd "$(dirname "$0")"
HEAD_HOST=spark@192.168.1.121
SWITCH=admin@192.168.1.159
URL=http://192.168.1.121:8000
RUN=bench/runs/autoresearch-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"

cp .env.example .env
bash tests/test-recipe.sh > "$RUN/contract.log" 2>&1 || { tail -20 "$RUN/contract.log"; echo "contract tests failed" >&2; exit 1; }

image=$(awk -F= '/^IMAGE=/ {split($2, a, " "); print a[1]}' .env.example)
state="\$HOME/.cache/dsv41-ds41f/autoresearch.${image//[^A-Za-z0-9._-]/_}.docker.sha256"
docker_hash=$(find docker -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-64)
built_hash=$(ssh -o BatchMode=yes "$HEAD_HOST" "cat $state 2>/dev/null || true")
if [[ $docker_hash != "$built_hash" ]]; then
  echo "docker/ changed for $image (${built_hash:-never built} -> $docker_hash): rebuilding"
  ./dsv41.sh stop > "$RUN/stop-for-build.log" 2>&1
  t=$SECONDS
  ./dsv41.sh build > "$RUN/build.log" 2>&1 || { tail -40 "$RUN/build.log"; echo "build failed" >&2; exit 1; }
  echo "METRIC build_s=$((SECONDS - t))"
  ssh -o BatchMode=yes "$HEAD_HOST" "mkdir -p \$HOME/.cache/dsv41-ds41f && echo $docker_hash > $state"
fi

t=$SECONDS
./dsv41.sh restart > "$RUN/restart.log" 2>&1 || { tail -60 "$RUN/restart.log"; echo "restart failed" >&2; exit 1; }
boot_s=$((SECONDS - t))
engine_up_s=$(grep -o 'up after [0-9]*s' "$RUN/restart.log" | grep -o '[0-9]*' | tail -1)
grep -q 'smoke: OK' "$RUN/restart.log" || { tail -20 "$RUN/restart.log"; echo "smoke failed" >&2; exit 1; }

./dsv41.sh bench decode --conc 1 --seconds 20 --out "$RUN/warmup-decode" > "$RUN/warmup.log" 2>&1 \
  && ./dsv41.sh bench prefill --lengths 32k --out "$RUN/warmup-prefill" >> "$RUN/warmup.log" 2>&1 \
  || { tail -30 "$RUN/warmup.log"; echo "warm-up failed" >&2; exit 1; }

./dsv41.sh bench decode --conc 1 --seconds 30 --out "$RUN/decode" > "$RUN/decode.log" 2>&1 \
  || { tail -30 "$RUN/decode.log"; echo "decode bench failed" >&2; exit 1; }

# Per Spark port: name tx-bytes tx-drop-packet tx-pause
switch_counters() {
  ssh -o BatchMode=yes "$SWITCH" ':foreach i in=[/interface ethernet find where name~"qsfp56-dd-.-1"] do={:put ([/interface ethernet get $i name] . " " . [/interface ethernet get $i tx-bytes] . " " . [/interface ethernet get $i tx-drop-packet] . " " . [/interface ethernet get $i tx-pause])}' | tr -d '\r'
}
switch_counters > "$RUN/switch-before.txt"
./dsv41.sh bench prefill --lengths 32k,128k,512k --out "$RUN/prefill" > "$RUN/prefill.log" 2>&1 \
  || { tail -30 "$RUN/prefill.log"; echo "prefill bench failed" >&2; exit 1; }
switch_counters > "$RUN/switch-after.txt"

mem=$(for h in 192.168.1.121 192.168.1.122 192.168.1.123 192.168.1.18; do
        ssh -o BatchMode=yes spark@$h "awk '/MemAvailable/{printf \"%.1f\n\", \$2/1048576}' /proc/meminfo"; done | sort -n | head -1)
python3 bench/quality_gate.py "$URL" --out "$RUN/quality_answers.json" > "$RUN/quality.log" 2>&1 \
  || { cat "$RUN/quality.log"; echo "quality gate errored" >&2; exit 1; }
./dsv41.sh bench check --out "$RUN/check" > "$RUN/check.log" 2>&1 || true
python3 bench/long_oracle.py check "$URL" > "$RUN/long_oracle.log" 2>&1 || true
python3 bench/corruption_probe.py "$URL" > "$RUN/corruption.log" 2>&1 \
  || { cat "$RUN/corruption.log"; echo "corruption probe errored" >&2; exit 1; }

python3 - "$RUN" "$boot_s" "${engine_up_s:-0}" "$mem" <<'PY'
import json, math, re, sys
run, boot_s, up_s, mem = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
# The relayed switched build (bench/records/2026-10-31-switched) and the ring (2026-10-18-autoresearch).
REF_DEC = {"prose": 67.09, "code": 97.72, "structured": 87.64, "json": 97.20}
REF_PF = {32768: 6483.7, 131072: 6478.0, 524288: 5489.4}
RING_DEC = {"prose": 65.08, "code": 93.91, "structured": 84.49, "json": 93.75}
RING_PF = {32768: 6622.3, 131072: 6505.1, 524288: 5507.5}
cells = json.load(open(f"{run}/decode/results.json"))["results"]["cells"]
errors = sum(c["errors"] for c in cells)
c1 = {c["prompt_type"]: c["agg_tok_s"] for c in cells if c["concurrency"] == 1}
rows = {r["target"]: r["tok_s"] for r in json.load(open(f"{run}/prefill/results.json"))["results"]["rows"]}
dec = sum(c1[t] / REF_DEC[t] for t in REF_DEC) / len(REF_DEC)
pf = sum(rows[n] / REF_PF[n] for n in REF_PF) / len(REF_PF)
ring_dec = sum(c1[t] / RING_DEC[t] for t in RING_DEC) / len(RING_DEC)
ring_pf = sum(rows[n] / RING_PF[n] for n in RING_PF) / len(RING_PF)
q = int(re.search(r"accuracy (\d+)/50", open(f"{run}/quality.log").read()).group(1))
bad = int(re.search(r"corrupt (\d+)/13", open(f"{run}/corruption.log").read()).group(1))
check = json.load(open(f"{run}/check/results.json"))["results"]["rows"]
oracle = sum(r["match"] for r in check)
m = re.search(r"long oracle (\d+)/(\d+)", open(f"{run}/long_oracle.log").read())
long_ok, long_n = (int(m.group(1)), int(m.group(2))) if m else (0, 3)
def counters(path):
    return {f[0]: [int(x) for x in f[1:]] for f in (l.split() for l in open(path)) if len(f) == 4}
before, after = counters(f"{run}/switch-before.txt"), counters(f"{run}/switch-after.txt")
tx_gb = sum(after[p][0] - before[p][0] for p in after) / 1e9
drops = sum(after[p][1] - before[p][1] for p in after)
pauses = sum(after[p][2] - before[p][2] for p in after)
print(f"METRIC switched_score_pct={100 * (math.sqrt(dec * pf) - 1):.2f}")
print(f"METRIC decode_gain_pct={100 * (dec - 1):.2f}")
print(f"METRIC prefill_gain_pct={100 * (pf - 1):.2f}")
print(f"METRIC vs_ring_decode_pct={100 * (ring_dec - 1):.2f}")
print(f"METRIC vs_ring_prefill_pct={100 * (ring_pf - 1):.2f}")
for t in REF_DEC:
    print(f"METRIC decode_c1_{t}={c1[t]:.1f}")
for n in REF_PF:
    print(f"METRIC prefill_{n // 1024}k={rows[n]:.0f}")
print(f"METRIC prefill_fabric_gb={tx_gb:.1f}")
print(f"METRIC switch_drops={drops}")
print(f"METRIC switch_pauses={pauses}")
print(f"METRIC boot_s={boot_s}")
print(f"METRIC engine_up_s={up_s}")
print(f"METRIC mem_min_gib={mem:.1f}")
print(f"METRIC quality_gate={q}")
print(f"METRIC token_oracle={oracle}")
print(f"METRIC long_oracle={long_ok}")
print(f"METRIC corrupt_offsets={bad}")
print(f"METRIC decode_errors={errors}")
problems = [msg for ok, msg in ((errors == 0, f"decode errors {errors}"), (q >= 48, f"quality {q}/50"),
            (oracle == len(check), f"token oracle {oracle}/{len(check)}"),
            (long_ok == long_n, f"long oracle {long_ok}/{long_n}"), (drops == 0, f"switch drops {drops}")) if not ok]
if problems:
    sys.exit("gate failed: " + ", ".join(problems))
PY
