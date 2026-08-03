-- Canned DuckDB queries for the Parquet counter dataset produced by
-- TT_SIM_TRACE_COUNTERS=<dir>.
--
-- The dataset is Hive-partitioned by `chip` and `kernel_id`. Read via:
--   SELECT * FROM read_parquet('dir/**/*.parquet', hive_partitioning=true);
--
-- Run any of these in the DuckDB shell:
--   duckdb -c ".read tt_sim/trace/queries/counters.sql"

-- -----------------------------------------------------------------------
-- 1) Top counters by total value across the run
-- -----------------------------------------------------------------------
SELECT counter_name,
       COUNT(*)  AS samples,
       SUM(value) AS total
FROM   read_parquet('counters/**/*.parquet', hive_partitioning=true)
GROUP  BY counter_name
ORDER  BY total DESC
LIMIT  20;

-- -----------------------------------------------------------------------
-- 2) Per-unit instruction count (final value of `instr_retired`)
--    by joining each unit's last snapshot.
-- -----------------------------------------------------------------------
WITH last_sample AS (
    SELECT unit, MAX(cycle) AS max_cycle
    FROM   read_parquet('counters/**/*.parquet', hive_partitioning=true)
    WHERE  counter_name = 'instr_retired'
    GROUP  BY unit
)
SELECT  c.unit, c.value AS retired
FROM    read_parquet('counters/**/*.parquet', hive_partitioning=true) c
JOIN    last_sample l ON c.unit = l.unit AND c.cycle = l.max_cycle
WHERE   c.counter_name = 'instr_retired'
ORDER   BY retired DESC;

-- -----------------------------------------------------------------------
-- 3) Kernel-to-kernel comparison: how counter totals differ between
--    two kernels in the same run.
-- -----------------------------------------------------------------------
WITH kernel_totals AS (
    SELECT  kernel_id, counter_name, MAX(value) AS final_value
    FROM    read_parquet('counters/**/*.parquet', hive_partitioning=true)
    GROUP   BY kernel_id, counter_name
)
SELECT  a.counter_name,
        a.final_value AS k0,
        b.final_value AS k1,
        b.final_value - a.final_value AS delta
FROM    kernel_totals a
JOIN    kernel_totals b USING (counter_name)
WHERE   a.kernel_id = 0 AND b.kernel_id = 1
ORDER   BY ABS(delta) DESC
LIMIT   20;

-- -----------------------------------------------------------------------
-- 4) NoC hotspot detection: total bytes per (unit, txn_type)
-- -----------------------------------------------------------------------
SELECT  unit,
       counter_name,
       SUM(value) AS bytes
FROM   read_parquet('counters/**/*.parquet', hive_partitioning=true)
WHERE  counter_name = 'noc_bytes_total'
GROUP  BY unit, counter_name
ORDER  BY bytes DESC;

-- -----------------------------------------------------------------------
-- 5) Where the RISC-V time went: stall cycles per core, split by reason.
--    Rows only exist with TT_SIM_COST_MODEL set -- with the model off no
--    instruction can stall, so an empty result is the correct answer and
--    not a missing measurement.
-- -----------------------------------------------------------------------
SELECT  unit,
        SUM(value) FILTER (WHERE counter_name = 'stall_cycles')       AS stalled,
        SUM(value) FILTER (WHERE counter_name = 'stall_load_use')     AS load_use,
        SUM(value) FILTER (WHERE counter_name = 'stall_store_rate')   AS store_rate,
        SUM(value) FILTER (WHERE counter_name = 'stall_integer_unit') AS integer_unit,
        SUM(value) FILTER (WHERE counter_name = 'instr_retired')      AS retired
FROM    read_parquet('counters/**/*.parquet', hive_partitioning=true)
WHERE   counter_name IN ('stall_cycles', 'stall_load_use', 'stall_store_rate',
                         'stall_integer_unit', 'instr_retired')
GROUP   BY unit
ORDER   BY stalled DESC NULLS LAST;

-- -----------------------------------------------------------------------
-- 6) Tensix backend unit utilisation: modelled occupancy against the
--    length of the run. `busy_cycles` counts only opcodes the cost tables
--    have an opinion about, so a unit the tables leave uncosted (the
--    unpacker, the config unit, the mover) reports 0 rather than a guess.
-- -----------------------------------------------------------------------
SELECT  unit,
        SUM(value) FILTER (WHERE counter_name = 'busy_cycles') AS busy,
        SUM(value) FILTER (WHERE counter_name = 'compute_ops') AS ops,
        ROUND(100.0 * SUM(value) FILTER (WHERE counter_name = 'busy_cycles')
              / MAX(cycle), 2)                                 AS pct_of_run
FROM    read_parquet('counters/**/*.parquet', hive_partitioning=true)
GROUP   BY unit
HAVING  busy IS NOT NULL
ORDER   BY busy DESC;

-- -----------------------------------------------------------------------
-- 7) NoC flight time per NIU. Emitted in both regimes -- with the cost
--    model off every flight is one cycle, which is a real observation
--    about an un-modelled NoC rather than an estimate of a hop count.
-- -----------------------------------------------------------------------
SELECT  unit,
        SUM(value) FILTER (WHERE counter_name = 'noc_flight_cycles') AS flight,
        SUM(value) FILTER (WHERE counter_name = 'noc_txns_timed')    AS txns,
        ROUND(SUM(value) FILTER (WHERE counter_name = 'noc_flight_cycles')
              / NULLIF(SUM(value) FILTER (WHERE counter_name = 'noc_txns_timed'), 0),
              1)                                                     AS mean_cycles
FROM    read_parquet('counters/**/*.parquet', hive_partitioning=true)
GROUP   BY unit
HAVING  txns IS NOT NULL
ORDER   BY flight DESC;
