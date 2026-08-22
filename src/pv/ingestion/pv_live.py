"""Client for the Sheffield Solar PV_Live API.

PV_Live provides estimated PV generation for Great Britain, aggregated
nationally, by DNO Licence Area ("PES", 14 regions) or by Grid Supply Point
("GSP", ~330 regions), at 30-minute resolution from 2010 to now.

Two properties of this API drive the design here:

1. ``datetime_gmt`` labels the END of each half-hour interval, whereas
   Open-Meteo labels the START of each hour. Left unhandled that is a silent
   30-minute misalignment between features and target. Normalisation to
   period-beginning happens once, here, at the ingestion boundary.

2. Outturn estimates are revised retrospectively: every 5 minutes for the
   first 3 hours, and again on day+1 as late sample data arrives. A historical
   query therefore returns the *settled* value, not the value that was visible
   at the time. Training on settled values while serving against live ones is
   target leakage. ``fetch_vintage`` captures what is published right now so
   the two can be compared -- and it only works going forward, because past
   vintages are unrecoverable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import httpx
import pandas as pd
import structlog

from pv.config import Layer, settings
from pv.ingestion._http_utils import IngestionError, get_json, get_json_cached

log = structlog.get_logger(__name__)

BASE_URL: Final[str] = "https://api.pvlive.uk/pvlive/api/v4"
NAMESPACE: Final[str] = "pv_live"

# The API limits a request to 366 days and times out after 30s; 365 is safe.
MAX_RANGE_DAYS: Final[int] = 365
PERIOD_MINUTES: Final[int] = 30

EXTRA_FIELDS: Final[tuple[str, ...]] = (
    "capacity_mwp",
    "installedcapacity_mwp",
    "lcl_mw",
    "ucl_mw",
    "site_count",
    "updated_gmt",
)

NUMERIC_FIELDS: Final[tuple[str, ...]] = (
    "generation_mw",
    "capacity_mwp",
    "installedcapacity_mwp",
    "lcl_mw",
    "ucl_mw",
    "site_count",
)


class PVLiveError(IngestionError):
    """Raised when the PV_Live API returns something unusable."""


class EntityType(StrEnum):
    """Spatial aggregation level of a PV_Live series."""

    PES = "pes"
    GSP = "gsp"


def _endpoint(entity_type: EntityType, entity_id: int) -> str:
    return f"{BASE_URL}/{entity_type}/{entity_id}"


def _parse_response(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert PV_Live's column-oriented JSON into a DataFrame.

    The response is ``{"meta": [names...], "data": [[values...], ...]}``. The
    published docs show the key as both "meta" and "Meta", so both are
    accepted rather than trusting either.
    """
    meta = payload.get("meta", payload.get("Meta"))
    data = payload.get("data")
    if meta is None or data is None:
        raise PVLiveError(f"unexpected PV_Live response shape: {list(payload)}")
    return pd.DataFrame(data, columns=list(meta))


def _normalise(frame: pd.DataFrame, entity_type: EntityType) -> pd.DataFrame:
    """Rename to project conventions and shift to period-beginning UTC.

    ``datetime_gmt`` marks the end of the interval. Everything downstream --
    Open-Meteo weather, pvlib solar position, the feature builder -- assumes
    period-beginning, so the shift happens here and nowhere else.
    """
    if frame.empty:
        return frame

    out = frame.rename(columns={f"{entity_type}_id": "region_id"})

    end = pd.to_datetime(out["datetime_gmt"], utc=True)
    out["period_end_utc"] = end
    out["time"] = end - pd.Timedelta(minutes=PERIOD_MINUTES)
    out = out.drop(columns=["datetime_gmt"])

    out["region_id"] = out["region_id"].astype(int)
    out["entity_type"] = str(entity_type)

    if "updated_gmt" in out.columns:
        out["updated_gmt"] = pd.to_datetime(out["updated_gmt"], utc=True, errors="coerce")

    for column in NUMERIC_FIELDS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out.sort_values("time").reset_index(drop=True)


def _chunk_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into contiguous chunks the API will accept."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=MAX_RANGE_DAYS - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def list_regions(entity_type: EntityType, client: httpx.Client | None = None) -> pd.DataFrame:
    """Fetch the PES or GSP region list, with names and parent PES ids."""
    url = f"{BASE_URL}/{entity_type}_list"
    payload = get_json_cached(NAMESPACE, url, {}, client=client)
    return _parse_response(payload)


def fetch_outturn(
    entity_id: int,
    start: date,
    end: date,
    entity_type: EntityType = EntityType.PES,
    extra_fields: tuple[str, ...] = EXTRA_FIELDS,
    client: httpx.Client | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch settled historical outturn for one region over a date range.

    These are *revised* values. For what was actually visible at the time, see
    ``fetch_vintage``.
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    url = _endpoint(entity_type, entity_id)
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in _chunk_ranges(start, end):
        params: dict[str, Any] = {
            "start": f"{chunk_start.isoformat()}T00:00:00Z",
            "end": f"{chunk_end.isoformat()}T23:59:59Z",
        }
        if extra_fields:
            params["extra_fields"] = ",".join(extra_fields)

        log.info(
            "pv_live.fetch",
            entity=str(entity_type),
            region=entity_id,
            start=str(chunk_start),
            end=str(chunk_end),
        )
        payload = get_json_cached(NAMESPACE, url, params, client=client, use_cache=use_cache)
        frames.append(_parse_response(payload))

    if not frames:
        return pd.DataFrame()

    return _normalise(pd.concat(frames, ignore_index=True), entity_type)


def fetch_vintage(
    entity_id: int,
    entity_type: EntityType = EntityType.PES,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Capture the currently-published outturn for the last 24 hours.

    Run on a schedule, append-only. Never cached: the entire value of these
    rows is that they record what a live system would have seen at a specific
    moment, before revision.
    """
    now = datetime.now(UTC)
    params: dict[str, Any] = {
        "start": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extra_fields": ",".join(EXTRA_FIELDS),
    }
    payload = get_json(_endpoint(entity_type, entity_id), params, client=client)

    frame = _normalise(_parse_response(payload), entity_type)
    if not frame.empty:
        frame["captured_at_utc"] = now
    return frame


def bronze_path(entity_type: EntityType, region_id: int, dataset: str = "pv_live") -> Path:
    """Return the bronze partition directory for one region."""
    return (
        settings.layer_path(Layer.BRONZE, dataset)
        / f"entity_type={entity_type}"
        / f"region_id={region_id}"
    )


def write_bronze(
    frame: pd.DataFrame,
    entity_type: EntityType,
    region_id: int,
    dataset: str = "pv_live",
) -> list[Path]:
    """Write outturn to bronze as one Parquet file per month, overwriting."""
    if frame.empty:
        return []

    directory = bronze_path(entity_type, region_id, dataset)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for month, group in frame.groupby(frame["time"].dt.strftime("%Y-%m")):
        target = directory / f"month={month}.parquet"
        group.to_parquet(target, index=False)
        written.append(target)
        log.info("bronze.write", path=str(target), rows=len(group))

    return written


def append_vintage(frame: pd.DataFrame, dataset: str = "pv_live_vintage") -> Path | None:
    """Write a vintage capture to its own timestamped Parquet file.

    Append-only rather than overwrite: each capture is a distinct observation
    of what was published at a distinct moment, not a correction of the last.
    """
    if frame.empty:
        return None

    directory = settings.layer_path(Layer.BRONZE, dataset)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    target = directory / f"captured={stamp}.parquet"
    frame.to_parquet(target, index=False)
    log.info("vintage.write", path=str(target), rows=len(frame))
    return target
