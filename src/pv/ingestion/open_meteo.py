"""Client for the Open-meteo weather APIs.

Four sources are supported.

- ERA5                      reanalysis, reconstructed after the fact. Ground truth only.


- HISTORICAL_FORECAST       archived operational model output,
                            stitched from the first hours of each run. Effective lead time is only
                            a few hours, so it is realistic for nowcasting but not for day-ahead.

- PREVIOUS_RUNS             forecast values at a fixed lead-time offset of 1-7 days.
                            This is what a day-ahead model would actually have seen.
                            Archived from January 2024.

- LIVE_FORECAST             the real forecast, for serving.

"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import httpx
import pandas as pd
import structlog

from pv.config import Layer, settings
from pv.ingestion._http_utils import IngestionError, get_json_cached

log = structlog.get_logger(__name__)

NAMESPACE: Final[str] = "open_meteo"
MAX_LEAD_DAYS: Final[int] = 7

HOURLY_VARIABLES: Final[tuple[str, ...]] = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "is_day",
)

BACKWARD_LOOKING: Final[frozenset[str]] = frozenset(
    {
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "terrestrial_radiation",
        "precipitation",
        "rain",
        "snowfall",
    }
)


class OpenMeteoError(IngestionError):
    """Raised when the Open-Meteo API cannot be queried successfully."""


class WeatherSource(StrEnum):
    """Which Open-Meteo dataset a set of weather values came from."""

    ERA5 = "era5"
    HISTORICAL_FORECAST = "historical_forecast"
    PREVIOUS_RUNS = "previous_runs"
    LIVE_FORECAST = "live_forecast"

    @property
    def endpoint(self) -> str:
        match self:
            case WeatherSource.ERA5:
                return settings.open_meteo_archive_url
            case WeatherSource.HISTORICAL_FORECAST:
                return settings.open_meteo_historical_forecast_url
            case WeatherSource.PREVIOUS_RUNS:
                return settings.open_meteo_previous_runs_url
            case WeatherSource.LIVE_FORECAST:
                return settings.open_meteo_forecast_url

    @property
    def is_operational(self) -> bool:
        """True when the values would have been available before valid time."""
        return self is not WeatherSource.ERA5

    @property
    def is_cacheable(self) -> bool:
        """False for live forecasts, whose whole purpose is being current."""
        return self is not WeatherSource.LIVE_FORECAST


def _suffix_variables_for_previous_runs(variables: Sequence[str], lead_days: int) -> list[str]:
    """Append the previous runs lead-time suffix to each variable name
    temperature_2m at a one-day offset becomes temperature_2m_previous_day1.
    """
    return [f"{name}_previous_day{lead_days}" for name in variables]


def _strip_lead_suffix_from_previous_runs(column: str) -> str:
    """Remove a ``_previous_dayN`` suffix so all sources share one schema."""
    marker = "_previous_day"
    head, sep, tail = column.rpartition(marker)
    if sep and tail.isdigit():
        return head
    return column


def build_params(
    latitude: float,
    longitude: float,
    source: WeatherSource,
    start: date | None = None,
    end: date | None = None,
    variables: Sequence[str] = HOURLY_VARIABLES,
    lead_days: int | None = None,
    forecast_days: int = 3,
) -> dict[str, Any]:
    """Assemble query parameters for a single Open-Meteo request.

    Kept pure and separate from the HTTP call so it can be unit tested without
    a network, and so a failing request can be reproduced from its params.
    """
    if source is WeatherSource.PREVIOUS_RUNS:
        if lead_days is None:
            raise ValueError("lead_days is required for the previous runs API")
        if not 1 <= lead_days <= MAX_LEAD_DAYS:
            raise ValueError(f"lead_days must be between 1 and {MAX_LEAD_DAYS}, got {lead_days}")
        hourly = _suffix_variables_for_previous_runs(variables, lead_days)
    else:
        if lead_days is not None:
            raise ValueError(f"lead_days is only meaningful for previous runs, not {source}")
        hourly = list(variables)

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    }

    if source is WeatherSource.LIVE_FORECAST:
        params["forecast_days"] = forecast_days
    else:
        if start is None or end is None:
            raise ValueError(f"start and end are required for {source}")
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        params["start_date"] = start.isoformat()
        params["end_date"] = end.isoformat()

    return params


def to_frame(payload: dict[str, Any], source: WeatherSource, lead_days: int | None) -> pd.DataFrame:
    """Convert an Open-Meteo JSON payload into UTC-indexed frame.

    Column names are normalised across sources: a previous-runs response with
    ``temperature_2m_previous_day1`` becomes plain ``temperature_2m``, so
    downstream code never branches on which source the data came from.
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise OpenMeteoError(f"response has no hourly block: {list(payload)}")

    frame = pd.DataFrame(hourly)
    frame = frame.rename(
        columns={c: _strip_lead_suffix_from_previous_runs(c) for c in frame.columns}
    )
    frame["time"] = pd.to_datetime(frame["time"], utc=True)

    frame["source"] = str(source)
    frame["lead_days"] = lead_days if lead_days is not None else 0
    frame["latitude"] = payload.get("latitude")
    frame["longitude"] = payload.get("longitude")

    frame = frame.sort_values("time").reset_index(drop=True)
    shiftable = [c for c in frame.columns if c in BACKWARD_LOOKING]
    if shiftable:
        frame[shiftable] = frame[shiftable].shift(-1)
    return frame


def fetch_hourly(
    latitude: float,
    longitude: float,
    source: WeatherSource,
    start: date | None = None,
    end: date | None = None,
    variables: Sequence[str] = HOURLY_VARIABLES,
    lead_days: int | None = None,
    forecast_days: int = 3,
    client: httpx.Client | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch hourly weather for one location from one Open-Meteo source."""
    params = build_params(
        latitude=latitude,
        longitude=longitude,
        source=source,
        start=start,
        end=end,
        variables=variables,
        lead_days=lead_days,
        forecast_days=forecast_days,
    )

    log.info(
        "open_meteo.fetch",
        source=str(source),
        lat=latitude,
        lon=longitude,
        start=str(start),
        end=str(end),
        lead_days=lead_days,
    )
    payload = get_json_cached(
        NAMESPACE,
        source.endpoint,
        params,
        client=client,
        use_cache=use_cache and source.is_cacheable,
    )

    return to_frame(payload, source=source, lead_days=lead_days)


def bronze_path(region_id: int, source: WeatherSource, lead_days: int | None) -> Path:
    """Return the bronze partition directory for one region and source.

    Keyed on ``region_id`` to match the PV_Live target table, so the silver
    join is a straight equality on region and time.
    """
    lead = lead_days if lead_days is not None else 0
    return (
        settings.layer_path(Layer.BRONZE, "weather")
        / f"source={source}"
        / f"lead_days={lead}"
        / f"region_id={region_id}"
    )


def write_bronze(
    frame: pd.DataFrame, region_id: int, source: WeatherSource, lead_days: int | None
) -> list[Path]:
    """Write a weather frame to bronze as one Parquet file per month.

    Monthly rather than daily partitions.
    Whole months are overwritten rather than appended, which makes
    re-running ingestion idempotent.
    """
    if frame.empty:
        return []

    directory = bronze_path(region_id, source, lead_days)
    directory.mkdir(parents=True, exist_ok=True)

    frame = frame.assign(region_id=region_id)
    written: list[Path] = []
    for month, group in frame.groupby(frame["time"].dt.strftime("%Y-%m")):
        target = directory / f"month={month}.parquet"
        group.to_parquet(target, index=False)
        written.append(target)
        log.info("bronze.write", path=str(target), rows=len(group))

    return written
