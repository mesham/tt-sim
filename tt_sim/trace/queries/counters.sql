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
