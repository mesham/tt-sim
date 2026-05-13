# Canned queries for tt-sim traces

Two query sets:

- **Perfetto** (below) — SQL run in the **Query (SQL)** tab of
  `ui.perfetto.dev` against a loaded `*.json.gz` trace.
- **DuckDB / Parquet** — see [`counters.sql`](counters.sql) for queries
  against the Parquet dataset produced by
  `TT_SIM_TRACE_COUNTERS=<dir>`. Run with
  `duckdb -c ".read tt_sim/trace/queries/counters.sql"` or copy into
  any DuckDB / pandas / Polars session.

## Perfetto SQL queries for tt-sim traces

After loading a `*.json.gz` trace into [ui.perfetto.dev](https://ui.perfetto.dev),
open the **Query (SQL)** tab and run any of the queries below. The
shape of the slice table is documented at
<https://perfetto.dev/docs/analysis/sql-tables>.

## 1. Top slices by duration

Surfaces the longest-running architectural events — useful for spotting
stall hotspots once duration data becomes meaningful (durations are
synthetic 1-cycle today; see §I cycle accuracy in ROADMAP.md).

```sql
SELECT name, COUNT(*) AS n, SUM(dur) AS total_dur, AVG(dur) AS avg_dur
FROM slice
GROUP BY name
ORDER BY total_dur DESC
LIMIT 10;
```

## 2. Per-unit instruction count

Which core / Tensix unit retired the most events.

```sql
SELECT thread.name AS unit, COUNT(*) AS events
FROM slice
JOIN thread USING (utid)
GROUP BY unit
ORDER BY events DESC;
```

## 3. NoC transaction latency

Pair each `noc:request:*` slice with its `noc:response:*` partner via
the flow events and report round-trip cycles.

```sql
SELECT
  req.name AS request,
  resp.name AS response,
  resp.ts - req.ts AS roundtrip_cycles
FROM slice req
JOIN flow ON flow.slice_out = req.id
JOIN slice resp ON flow.slice_in = resp.id
WHERE req.name LIKE 'noc:request:%'
ORDER BY roundtrip_cycles DESC
LIMIT 20;
```
