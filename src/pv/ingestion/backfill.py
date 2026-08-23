"""Backfill bronze from PV_Live and Open-Meteo.

Run as a module:

    uv run python -m pv.ingestion.backfill --verify-regions
    uv run python -m pv.ingestion.backfill --start 2024-01-01 --end 2026-08-01

Both upstream sources are cached by request hash, so re-running after an
interruption re-fetches nothing already on disk. That makes this safe to stop
and restart, and safe to run repeatedly while developing.

Weather is fetched from three sources deliberately: ERA5 as ground truth,
previous-runs day-1 as the honest day-ahead feature set, and the historical
forecast archive as the middle rung. The three-way comparison between them is
the project's headline experiment, so all three are ingested together and the
choice of which to train on is deferred to the modelling layer.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, date, datetime, timedelta

import httpx
import structlog

from pv.config import settings
from pv.ingestion import open_meteo, pv_live, regions
from pv.ingestion.open_meteo import WeatherSource
from pv.ingestion.pv_live import EntityType

log = structlog.get_logger(__name__)

# Open-Meteo previous runs are archived from January 2024; earlier dates would
# return empty series and quietly produce a shorter training set than expected.
PREVIOUS_RUNS_START: date = date(2024, 1, 1)

# Weather sources to ingest, with the lead time each is fetched at.
WEATHER_PLAN: tuple[tuple[WeatherSource, int | None], ...] = (
    (WeatherSource.ERA5, None),
    (WeatherSource.HISTORICAL_FORECAST, None),
    (WeatherSource.PREVIOUS_RUNS, 1),
)

# Politeness delay between requests.
REQUEST_DELAY_S: float = 1.0


def backfill_outturn(
    start: date,
    end: date,
    client: httpx.Client,
    entity_type: EntityType = EntityType.PES,
) -> int:
    """Fetch and write settled outturn for every region. Returns rows written."""
    total = 0
    for region in regions.PES_REGIONS:
        frame = pv_live.fetch_outturn(
            entity_id=region.region_id,
            start=start,
            end=end,
            entity_type=entity_type,
            client=client,
        )
        pv_live.write_bronze(frame, entity_type, region.region_id)
        total += len(frame)
        log.info("backfill.outturn", region=region.region_id, rows=len(frame))
        time.sleep(REQUEST_DELAY_S)
    return total


def backfill_weather(start: date, end: date, client: httpx.Client) -> int:
    """Fetch and write all three weather sources for every region."""
    total = 0
    for region in regions.PES_REGIONS:
        for source, lead_days in WEATHER_PLAN:
            source_start = max(start, PREVIOUS_RUNS_START) if lead_days else start
            if source_start > end:
                log.warning(
                    "backfill.weather_skipped",
                    region=region.region_id,
                    source=str(source),
                    reason="requested range predates archive coverage",
                )
                continue

            frame = open_meteo.fetch_hourly(
                latitude=region.latitude,
                longitude=region.longitude,
                source=source,
                start=source_start,
                end=end,
                lead_days=lead_days,
                client=client,
            )
            open_meteo.write_bronze(frame, region.region_id, source, lead_days)
            total += len(frame)
            log.info(
                "backfill.weather",
                region=region.region_id,
                source=str(source),
                rows=len(frame),
            )
            time.sleep(REQUEST_DELAY_S)
    return total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=PREVIOUS_RUNS_START)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=(datetime.now(UTC) - timedelta(days=1)).date(),
    )
    parser.add_argument("--verify-regions", action="store_true", help="check metadata and exit")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-outturn", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.verify_regions:
        regions.verify_against_api()
        return 0

    with httpx.Client(timeout=settings.request_timeout_s) as client:
        if args.start > args.end:
            print(f"start {args.start} is after end {args.end}", file=sys.stderr)
            return 1

        log.info("backfill.start", start=str(args.start), end=str(args.end))

        if not args.skip_outturn:
            rows = backfill_outturn(args.start, args.end, client)
            log.info("backfill.outturn_complete", rows=rows)

        if not args.skip_weather:
            rows = backfill_weather(args.start, args.end, client)
            log.info("backfill.weather_complete", rows=rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
