# MAX_NUM_BATCHED_TOKENS x CED non-final skip, measured with servebench (2026-10-18)

For manual review. Nothing in the recipe's defaults was changed by this sweep: `.env.example` is still
`MAX_NUM_BATCHED_TOKENS=24576`, CED skip on (`DSV41_CED_SKIP_NONFINAL` unset = 1).

## Method

- Client: Christopher Owen's servebench (spark-ds41f `2d2b2328`, vendored unmodified in `bench/servebench`,
  sha256-pinned in `bench/servebench.sha256`), run on the head node (192.168.1.121, system Python 3.12.3)
  against 127.0.0.1:8000. Suite `bench/suites/dsv41-ladder.json` (sha `b7fc74aeaea4`):
  - closed loops at c16, c8, c4, c2, c1: 256 distinct prompts, a fresh cache salt per request, exactly 256
    output tokens (`ignore_eos`), greedy, thinking off, 60 s warm-up, 300 s measured in 10 batches
    (his release suite's c16/c8 and single-stream c1 arguments);
  - cold real-text prefill (stdlib source, his seeds) at 32K / 128K / 256K / 512K, 3 repeats, and 8K / 16K,
    6 repeats. Rate = prompt tokens / TTFT.
- One restart per config (`autoresearch.sh`, runs #27-#36 of autoresearch session 2, commit `048113e`); each
  config changed only `.env.example` (`MAX_NUM_BATCHED_TOKENS`, and `EXTRA_DOCKER_ARGS="-e
  DSV41_CED_SKIP_NONFINAL=0"` for skip off). Same image throughout (no `docker/` change, no rebuild).
  Served args/env were read back from the head container into each run's `metadata.json`.
- GPU clocks at the site's 2,200 MHz lock. Quality gate 49/50 and corruption probe 9/13 on every run; zero
  failed or foreign requests.
- `chunks / skipped` are derived from the recorded prompt lengths with docker/0005's rule (a non-final chunk
  ends on a threshold multiple and >= 200 rows follow it at k=6); they are not engine counters.
- One boot per config: differences under ~1% (and anything where the size's CI is wide; 32K reached 5.4%)
  are not separable from boot-to-boot spread.

Each config's full run directory is kept beside this file (`mnbt<N>-ced<on|off>/`, servebench JSON in
`servebench/`).

## Prefill (tok/s, servebench cold real text)

| MNBT (threshold) | CED skip | gmean | 8K | 16K | 32K | 128K | 256K | 512K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8192 (8080) | on | 5,609 | 5,257 | 5,782 | 5,982 | 5,926 | 5,654 | 5,111 |
| 8192 (8080) | off | 5,387 | 5,178 | 5,548 | 5,718 | 5,661 | 5,383 | 4,880 |
| 12288 (12176) | on | 5,918 | 5,837 | 6,087 | 6,305 | 6,189 | 5,849 | 5,299 |
| 12288 (12176) | off | 5,776 | 5,827 | 5,886 | 6,171 | 6,006 | 5,700 | 5,125 |
| 16384 (16272) | on | 5,988 | 5,773 | 6,088 | 6,386 | 6,337 | 5,999 | 5,404 |
| 16384 (16272) | off | 5,868 | 5,831 | 6,076 | 6,153 | 6,122 | 5,813 | 5,262 |
| 20480 (20368) | on | 6,095 | 5,833 | 6,468 | 6,409 | 6,397 | 6,090 | 5,444 |
| 20480 (20368) | off | 6,052 | 5,774 | 6,478 | 6,448 | 6,325 | 5,995 | 5,370 |
| **24576 (24464)** | **on** | **6,125** | 5,815 | 6,463 | 6,509 | 6,436 | 6,113 | 5,486 |
| 24576 (24464) | off | 6,061 | 5,806 | 6,477 | 6,382 | 6,369 | 5,980 | 5,423 |

Prompt tokens: 8,255 / 16,520 / 32,761 / 131,900 / 262,551 / 522,785.

## What the skip acts on, and what it is worth

| MNBT | chunks / skipped: 8K | 16K | 32K | 128K | 256K | 512K | skip gain: gmean | 16K | 32K | 128K | 256K | 512K | ms saved per skipped chunk (128K / 512K) |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 2/0 | 3/2 | 5/4 | 17/16 | 33/32 | 65/64 | +4.1% | +4.2% | +4.6% | +4.7% | +5.0% | +4.7% | 65 / 76 |
| 12288 | 1/0 | 2/1 | 3/2 | 11/10 | 22/21 | 43/42 | +2.5% | +3.4% | +2.2% | +3.0% | +2.6% | +3.4% | 65 / 80 |
| 16384 | 1/0 | 2/1 | 3/2 | 9/8 | 17/16 | 33/32 | +2.1% | +0.2% | +3.8% | +3.5% | +3.2% | +2.7% | 92 / 82 |
| 20480 | 1/0 | 1/0 | 2/1 | 7/6 | 13/12 | 26/25 | +0.7% | -0.2% | -0.6% | +1.1% | +1.6% | +1.4% | 39 / 53 |
| 24576 | 1/0 | 1/0 | 2/1 | 6/5 | 11/10 | 22/21 | +1.1% | -0.2% | +2.0% | +1.0% | +2.2% | +1.2% | 43 / 53 |

(8K at 8192: 2 chunks, 0 skipped, the 175-row tail is under 200. 16K at 16384: 1 skipped, but the second chunk is
only 248 rows, so almost nothing is saved.)

## Decode ladder (output tok/s, servebench closed loop, distinct prompts)

| MNBT | CED | c1 | c2 | c4 | c8 | c16 | c16 TTFT p50 ms | c16 TPOT p50 ms | c16 steps/s | c16 tok/step | mem_min GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | on | 72.9 | 104.3 | 139.1 | 179.6 | 229.6 | 485 | 69.4 | 6.55 | 35.09 | 19.2 |
| 8192 | off | 72.8 | 104.1 | 138.9 | 179.1 | 229.3 | 482 | 69.4 | 6.55 | 35.05 | 19.2 |
| 12288 | on | 72.6 | 103.9 | 138.8 | 179.7 | 228.8 | 480 | 69.7 | 6.54 | 35.05 | 17.9 |
| 12288 | off | 72.7 | 104.0 | 139.3 | 178.7 | 228.1 | 483 | 69.8 | 6.51 | 35.05 | 17.8 |
| 16384 | on | 72.4 | 104.1 | 138.1 | 179.4 | 229.1 | 482 | 69.5 | 6.54 | 35.05 | 16.0 |
| 16384 | off | 72.6 | 104.0 | 139.0 | 179.1 | 228.9 | 478 | 69.4 | 6.54 | 35.05 | 16.0 |
| 20480 | on | 72.7 | 103.6 | 138.9 | 179.1 | 229.1 | 485 | 69.3 | 6.54 | 35.07 | 14.9 |
| 20480 | off | 72.7 | 104.2 | 139.2 | 178.9 | 229.2 | 484 | 69.4 | 6.54 | 35.05 | 14.8 |
| **24576** | **on** | 72.7 | 103.7 | 138.8 | 178.2 | 229.1 | 483 | 69.5 | 6.54 | 35.04 | 13.4 |
| 24576 | off | 72.5 | 103.7 | 139.1 | 179.5 | 229.0 | 484 | 69.7 | 6.54 | 35.05 | 13.4 |

Spread over all ten configs: c1 0.6%, c2 0.6%, c4 0.9%, c8 0.8%, c16 0.65%, each within its own 95% interval
(c1 ±3.5%, c2 ±2%, c4 ±1.2%, c8 ±0.6%, c16 ±0.8%). `mem_min` is the lowest MemAvailable over the four nodes after
the benchmarks.

## Findings

1. **Decode is independent of both knobs** at every concurrency from 1 to 16 on distinct prompts: the ladder,
   engine steps/s and tokens per step are flat across all ten configs. The closed loops' prefills are short chat
   prompts (one chunk at any MNBT), so neither the batch budget nor the skip reaches them.
2. **Prefill rises monotonically with MAX_NUM_BATCHED_TOKENS** (skip on, gmean): 5,609 → 5,918 → 5,988 → 6,095 →
   6,125. The step from 8192 to 12288 is the big one (+5.5%); 20480 → 24576 is +0.5%, inside one boot's noise.
3. **The skip's gain shrinks as MNBT grows**: +4.1% at 8192, +2.5% at 12288, +2.1% at 16384, +0.7% at 20480,
   +1.1% at 24576. The number of skipped chunks is ~prompt / threshold, and each saves a roughly fixed amount
   (about 40-90 ms here, the CED decoder on 127 discarded rows), so the relative gain goes as 1/MNBT. The two
   knobs are partial substitutes: the skip removes part of the per-chunk cost that a larger MNBT removes by
   having fewer chunks.
4. **The skip never replaces a larger budget**: 8192 with skip (5,609) is below 12288 without (5,776); 16384
   with skip (5,988) is below 20480 without (6,052). Batch size buys more than the skip at every step.
5. **Single-chunk prompts are untouched** (8K, and 16K at MNBT >= 20480): skip on/off within ±1%, as the patch
   intends (exactness was established when docker/0005 landed; this sweep only measured speed).
6. **Chunk boundaries show directly**: the 16,520-token prompt costs 2 chunks at MNBT <= 16384 and 1 above, which
   is the 6,088 → 6,468 jump (+6%) at 16K.
7. **Memory**: mem_min falls ~0.6-0.7 GiB per +4096 MNBT (19.2 GiB at 8192 → 13.4 at 24576); the skip has no
   memory effect.
8. **Against Christopher Owen's r6v release** (his ring and session, `experiments/2026-10-17-r6v`, same servebench
   arguments; recorded as `vs_chris_r6v_*` per run): at our defaults c1 +2.3% (72.65 vs 71.05), c8 -3.9% (178.2 vs
   185.4), c16 -6.0% (229.1 vs 243.8); prefill +8.8% / +8.2% / +7.7% / +6.1% at 32K / 128K / 256K / 512K. The
   c8/c16 deficit is the same at every MNBT and skip setting, so it is not caused by either. Cross-site,
   cross-session comparison: his own method says never to judge against a reference from another session, so
   treat this as a pointer, not a verdict. His tp4-ring config differs in k (5 vs our 6), draft budget (r6v's
   timed cost scale 2.0; r6x row cost 0.035 vs our `ADAPTIVE_VERIFICATION_ROW_COST=0.01`), MNBT 8192 vs 24576, and
   clocks (ours locked at 2,200 MHz).
9. **bench.py vs servebench**: our earlier `bench decode` c=8 numbers (321-506 tok/s per cell, one prompt copied
   to every worker, shared prefix cached) are not comparable with servebench's c8 (~179 tok/s, distinct prompts,
   fresh salts, 256 fixed output tokens). Concurrency claims should come from servebench.

## Recommendations (for manual review; none applied)

1. **Keep `MAX_NUM_BATCHED_TOKENS=24576` and the CED skip on.** Best prefill in the sweep, decode unaffected,
   the skip adds ~+1% at this budget for free.
2. **20480 is the memory-headroom alternative**: -0.5% prefill (within noise), decode identical, +1.5 GiB
   MemAvailable. Worth it only if the 1M-needle floor (13.1 GiB on the head at 24576) becomes a problem. Do not go
   below 16384 for prefill-heavy use: 12288 costs -3.4% and 8192 -8.4%.
3. **Keep docker/0005 even though its gain is small at 24576**: it is the cheap insurance if MNBT ever has to
   drop for memory (worth +2 to +4% at 8192-16384).
4. **Next investigation for c8/c16 (not run)**: our row cost 0.01 was picked on `bench.py`'s same-prompt c=1
   screen; Chris found 0.035 kept r6v's c16 operating point while 0.02 verified 8-12% more for the same output.
   A servebench c16/c8/c1 A/B of `ADAPTIVE_VERIFICATION_ROW_COST` 0.01 vs 0.02 vs 0.035 (and k=5 vs 6) is the
   likely place to recover the -6% at c16. A same-session run of his r6x image on this ring would settle the
   cross-site question.
5. **Harness**: use `autoresearch.sh` / servebench for concurrency results going forward and stop quoting
   `bench decode` c>1 numbers. For sub-1% decisions, repeat boots (one boot per config here).

## State left behind

- The cluster is serving the last sweep point, **MNBT 20480 with the CED skip off**, until the next
  `./dsv41.sh restart` (`.env` was reset to `.env.example`, so a restart returns to the defaults).
- Branch `autoresearch/session-20261018-2`: the servebench harness (`048113e`) and this record; not pushed.
