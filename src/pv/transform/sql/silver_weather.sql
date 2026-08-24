-- silver_weather: one row per region per hour per weather source.
--
-- All three sources (ERA5, historical forecast, previous runs day-1) live in
-- the same table, distinguished by `source` and `lead_days`.
--
-- Timestamps are already period-beginning UTC: open_meteo.to_frame relabels
-- the backward-looking accumulation variables at ingestion

CREATE OR REPLACE TABLE silver_weather AS
WITH raw AS (
    SELECT
        region_id,
        time,
        source,
        lead_days,
        shortwave_radiation,
        direct_radiation,
        diffuse_radiation,
        direct_normal_irradiance,
        temperature_2m,
        """Adding this constraint since open-meteo can return RH slightly above as derivation artefact
        of dew point and temperature"""
        least(greatest(relative_humidity_2m, 0.0),100.0) AS relative_humidity_2m,
        dew_point_2m,
        cloud_cover,
        cloud_cover_low,
        cloud_cover_mid,
        cloud_cover_high,
        wind_speed_10m,
        surface_pressure,
        precipitation,
        is_day
    FROM read_parquet('{bronze_weather}', union_by_name = true)
    -- The relabelling shift leaves the final hour of each fetch NULL, because
    -- that hour's accumulation genuinely is not known yet. Dropping beats
    -- imputing: a fabricated irradiance value would be indistinguishable from
    -- a real one downstream.
    WHERE shortwave_radiation IS NOT NULL
)

SELECT *
FROM raw
-- Overlapping fetches can duplicate an hour. The natural key includes source
-- and lead_days, since the same hour legitimately appears once per source.
QUALIFY row_number() OVER (PARTITION BY region_id, time, source, lead_days) = 1
ORDER BY region_id, source, lead_days, time;
