# Benchmark records

Curated benchmark results, committed to git.

- `bench/runs/` is scratch. Every `bench/bench.py` invocation writes a
  `<YYYY-mm-dd-HHMMSS>-<step>/` directory there containing `results.json`
  and `summary.md`. The directory is gitignored.
- `bench/records/<YYYY-mm-dd>-<label>/` is a run directory worth keeping,
  copied here unchanged and committed. `<label>` names the configuration
  being recorded, for example `2026-10-17-tp4-dspark-baseline`. Use the
  date the run was taken, not the date it was copied.

Copy the whole run directory (`results.json` holds the raw numbers, the
request settings and the timestamps; `summary.md` holds the tables).
Never edit a record after it is committed. If a number needs to change,
take a new run and add a new record.

To promote a run:

```bash
cp -R bench/runs/2026-10-17-142233-all bench/records/2026-10-17-tp4-dspark-baseline
```
