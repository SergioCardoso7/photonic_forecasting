-- silver_pv: one row per region per hour, cleaned and conformed.
--
-- Bronze is half-hourly (PV_Live's only resolution). Open-Meteo's Previous Runs
-- archive is hourly, so hourly is the finest granularity at which features
-- genuinely exist.
--
-- generation_mw is *average power over the period*, so averaging two half-hours
-- gives average power over the hour. Summing would be wrong.

CREATE OR REPLACE TABLE silver_pv AS
WITH half_hourly AS (
    SELECT
        region_id,
        time,
        generation_mw,
        capacity_mwp,
        installedcapacity_mwp
    FROM read_parquet('{bronze_pv}')
    WHERE generation_mw IS NOT NULL
      AND capacity_mwp IS NOT NULL
),

-- The same half-hour can arrive twice when chunked date ranges overlap at a
-- month boundary. Deduplicate on the natural key before aggregating, or the
-- hourly mean is silently weighted towards the duplicated period.
deduplicated AS (
    SELECT *
    FROM half_hourly
    QUALIFY row_number() OVER (PARTITION BY region_id, time) = 1
),

hourly AS (
    SELECT
        region_id,
        time_bucket(INTERVAL '1 hour', time) AS time,
        avg(generation_mw)         AS generation_mw,
        avg(capacity_mwp)          AS capacity_mwp,
        avg(installedcapacity_mwp) AS installedcapacity_mwp,
        count(*)                   AS n_periods
    FROM deduplicated
    GROUP BY region_id, time_bucket(INTERVAL '1 hour', time)
)

SELECT
    region_id,
    time,
    generation_mw,
    capacity_mwp,
    installedcapacity_mwp,
    -- Load factor is the modelling target: normalising by capacity makes
    -- regions of wildly different size directly comparable, which is what
    -- allows one global model to be trained across all of them.
    generation_mw / nullif(capacity_mwp, 0) AS load_factor
FROM hourly
-- An hour built from one half-period is not an hourly average. Dropping it rather
-- than let a partial hour masquerade as a complete one.
WHERE n_periods = 2
ORDER BY region_id, time;
