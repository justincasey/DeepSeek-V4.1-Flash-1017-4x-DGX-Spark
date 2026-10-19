# Draft budget and depth on servebench (2026-10-18/19), follow-up to the MNBT x CED sweep

For manual review. The recipe's defaults are unchanged (`ADAPTIVE_VERIFICATION_ROW_COST=0.01`,
`NUM_SPECULATIVE_TOKENS=6`).

Same harness, suite and method as [2026-10-18-servebench-sweep](../2026-10-18-servebench-sweep/summary.md)
(servebench `2d2b2328`, suite `dsv41-ladder` sha `b7fc74aeaea4`, one boot per config, 2,200 MHz lock).
`MAX_NUM_BATCHED_TOKENS=24576` and the CED skip on throughout. These are autoresearch runs #37 (`k6-rowcost0.035/`)
and #38 (`k5-rowcost0.035/`); the reference is run #27 (the sweep's `mnbt24576-cedon/`). The row cost served by
each run is recorded in its `served.txt`.

Question: the sweep left c8 at -3.9% and c16 at -6.0% against Christopher Owen's r6v release, with neither knob
moving them. Our row cost 0.01 was picked on `bench.py`'s same-prompt c=1 screen; Chris serves 0.035 at k=5.

## Results (output tok/s, distinct prompts, closed loop)

| config | c1 | c2 | c4 | c8 | c16 | c8 TTFT p50 ms | c16 TTFT p50 ms | c16 steps/s | c16 tok/step | prefill gmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k=6, row cost 0.01 (defaults, #27) | 72.65 | 103.70 | 138.85 | 178.19 | 229.12 | 321 | 483 | 6.54 | 35.04 | 6,125 |
| k=6, row cost 0.035 (#37) | 72.48 | 104.66 | 140.87 | 183.26 | 227.65 | 301 | 436 | 7.39 | 30.83 | 6,108 |
| k=5, row cost 0.035 (#38) | 72.89 | 104.55 | 141.18 | 183.55 | 228.53 | 300 | 433 | 7.43 | 30.77 | 6,103 |
| 95% interval (typical) | ±3.4% | ±2.1% | ±1.2% | ±0.5-1.0% | ±0.6-0.8% | | | | | |

Against Chris's r6v (his ring and session): c1 +2.3 / +2.0 / +2.6%, c8 -3.9 / -1.2 / -1.0%,
c16 -6.0 / -6.6 / -6.3% for the three rows.

## Findings

1. **Row cost 0.035 trades tokens per step for steps per second**: at c16, 12% fewer tokens per step and 13% more
   steps per second. Throughput is +1.5% at c4 and **+2.8% at c8** (outside c8's ±0.5% interval), level at c1, c2
   and c16. First-token latency improves at load: c8 p50 321 → 301 ms, c16 483 → 436 ms (-10%).
2. **k=5 vs k=6 at row cost 0.035 makes no difference** at any concurrency (all within interval); prefill level.
   At this budget the sixth draft position is rarely verified.
3. **The c16 gap to Chris (about -6%) is not the draft budget or the draft depth**, and the sweep showed it is not
   MAX_NUM_BATCHED_TOKENS or the CED skip. What's left is cross-site: his ring and clocks (ours locked at 2,200 MHz)
   and session. Not tested.
4. Prefill is unaffected by either change (gmean within 0.4% of the defaults).

## Recommendations (none applied)

1. **Consider `ADAPTIVE_VERIFICATION_ROW_COST=0.035`**: +2.8% at c8, +1.5% at c4, -10% TTFT at c8/c16, level at
   c1/c16 and in prefill. It is one boot per config: confirm with a repeat boot of each before switching the
   default. Greedy output is unaffected (verification is exact), so this is a speed-only change.
2. **Keep k=6** unless the graph-ladder memory matters: k=5 is level at 0.035 and captures fewer graph rows
   (96 vs 112; mem_min 13.2 vs 13.5 GiB here, within noise).
3. **For the c16 gap**, run Chris's r6x image on this ring in the same session as ours, and check whether the
   clock lock accounts for it. Only that comparison separates site from recipe.

## Harness note

`autoresearch.sh` derives `ced_skipped_chunks_*` only at k=6 (the 200-row rule). Run #38 reports 0 for them although
the skip was active (`DSV41_CED_SKIP_NONFINAL` unset; the minimum remaining at k=5 is 199 rows).

## State left behind

The cluster is serving **k=5, row cost 0.035** (run #38) until the next `./dsv41.sh restart`. `.env` was reset to
`.env.example`, so a restart returns to the defaults.
