"""Tests for the PV_Live client.

The period-end normalisation tests are the important ones: a silent 30-minute
misalignment between target and features is exactly the kind of bug that
survives to production and is nearly impossible to find afterwards.
"""

from __future__ import annotations

import itertools
from datetime import date
from typing import Any

import httpx
import pandas as pd
import pytest

from pv.ingestion._http_utils import IngestionError
from pv.ingestion.pv_live import (
    MAX_RANGE_DAYS,
    EntityType,
    PVLiveError,
    _chunk_ranges,
    _normalise,
    _parse_response,
    fetch_outturn,
    write_bronze,
)


def _response(rows: list[list[Any]], meta: list[str]) -> dict[str, Any]:
    return {"meta": meta, "data": rows}


def _mock_client(payload: dict[str, Any]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestParseResponse:
    def test_maps_meta_to_columns(self):
        payload = _response(
            [[10, "2026-06-01T13:00:00Z", 1508.5]], ["pes_id", "datetime_gmt", "generation_mw"]
        )
        assert list(_parse_response(payload).columns) == [
            "pes_id",
            "datetime_gmt",
            "generation_mw",
        ]

    def test_accepts_capitalised_meta_key(self):
        payload = {"Meta": ["pes_id"], "data": [[10]]}
        assert _parse_response(payload).iloc[0]["pes_id"] == 10

    def test_rejects_unexpected_shape(self):
        with pytest.raises(PVLiveError, match="unexpected PV_Live response"):
            _parse_response({"something": "else"})

    def test_error_is_catchable_as_ingestion_error(self):
        with pytest.raises(IngestionError):
            _parse_response({})


class TestNormalise:
    def test_shifts_period_end_to_period_beginning(self):
        """13:30 period-ending must become 13:00 period-beginning."""
        frame = _parse_response(
            _response(
                [[10, "2026-06-01T13:30:00Z", 1508.5]], ["pes_id", "datetime_gmt", "generation_mw"]
            )
        )
        out = _normalise(frame, EntityType.PES)
        assert out.iloc[0]["time"] == pd.Timestamp("2026-06-01T13:00:00Z")
        assert out.iloc[0]["period_end_utc"] == pd.Timestamp("2026-06-01T13:30:00Z")

    def test_renames_entity_id_to_region_id(self):
        frame = _parse_response(
            _response(
                [[7, "2026-06-01T13:30:00Z", 1.0]], ["gsp_id", "datetime_gmt", "generation_mw"]
            )
        )
        out = _normalise(frame, EntityType.GSP)
        assert out.iloc[0]["region_id"] == 7
        assert out.iloc[0]["entity_type"] == "gsp"

    def test_coerces_numeric_columns(self):
        frame = _parse_response(
            _response(
                [[10, "2026-06-01T13:30:00Z", "1508.5", "13000.0"]],
                ["pes_id", "datetime_gmt", "generation_mw", "capacity_mwp"],
            )
        )
        out = _normalise(frame, EntityType.PES)
        assert out["generation_mw"].dtype.kind == "f"
        assert out["capacity_mwp"].dtype.kind == "f"

    def test_nulls_survive_darkness(self):
        """lcl_mw and site_count are NULL at night; that must not raise."""
        frame = _parse_response(
            _response(
                [[10, "2026-06-01T01:30:00Z", 0.0, None]],
                ["pes_id", "datetime_gmt", "generation_mw", "lcl_mw"],
            )
        )
        out = _normalise(frame, EntityType.PES)
        assert pd.isna(out.iloc[0]["lcl_mw"])

    def test_parses_updated_gmt_for_revision_tracking(self):
        frame = _parse_response(
            _response(
                [[10, "2026-06-01T13:30:00Z", 1.0, "2026-06-02T10:32:00Z"]],
                ["pes_id", "datetime_gmt", "generation_mw", "updated_gmt"],
            )
        )
        out = _normalise(frame, EntityType.PES)
        assert out.iloc[0]["updated_gmt"] == pd.Timestamp("2026-06-02T10:32:00Z")

    def test_empty_frame_passes_through(self):
        assert _normalise(pd.DataFrame(), EntityType.PES).empty


class TestChunkRanges:
    def test_short_range_is_one_chunk(self):
        assert _chunk_ranges(date(2025, 1, 1), date(2025, 3, 1)) == [
            (date(2025, 1, 1), date(2025, 3, 1))
        ]

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (date(2023, 1, 1), date(2025, 12, 31)),  # spans a leap year
            (date(2024, 2, 29), date(2024, 2, 29)),  # single leap day
            (date(2010, 1, 1), date(2026, 1, 1)),  # full PV_Live history
            (date(2025, 6, 1), date(2025, 6, 1)),  # single day
        ],
    )
    def test_chunks_tile_the_range_exactly(self, start, end):
        chunks = _chunk_ranges(start, end)

        assert chunks, "a valid range must produce at least one chunk"
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

        for chunk_start, chunk_end in chunks:
            assert chunk_start <= chunk_end
            assert (chunk_end - chunk_start).days < MAX_RANGE_DAYS

        for (_, prev_end), (next_start, _) in itertools.pairwise(chunks):
            assert (next_start - prev_end).days == 1


class TestFetchOutturn:
    def test_rejects_reversed_range(self):
        with pytest.raises(ValueError, match="is after end"):
            fetch_outturn(10, date(2025, 5, 2), date(2025, 5, 1))

    def test_returns_normalised_frame(self):
        payload = _response(
            [[10, "2026-06-01T13:30:00Z", 1508.5], [10, "2026-06-01T14:00:00Z", 1400.0]],
            ["pes_id", "datetime_gmt", "generation_mw"],
        )
        frame = fetch_outturn(
            10,
            date(2026, 6, 1),
            date(2026, 6, 1),
            extra_fields=(),
            client=_mock_client(payload),
        )
        assert len(frame) == 2
        assert frame["time"].is_monotonic_increasing


class TestWriteBronze:
    def test_partitions_by_month_and_is_idempotent(self):
        times = pd.date_range("2025-05-30", "2025-06-02", freq="30min", tz="UTC")
        frame = pd.DataFrame({"time": times, "generation_mw": 1.0, "region_id": 10})

        first = write_bronze(frame, EntityType.PES, 10)
        second = write_bronze(frame, EntityType.PES, 10)

        assert {p.name for p in first} == {"month=2025-05.parquet", "month=2025-06.parquet"}
        assert first == second

    def test_empty_frame_writes_nothing(self):
        assert write_bronze(pd.DataFrame(), EntityType.PES, 10) == []


@pytest.mark.integration
def test_live_api_returns_capacity_and_generation():
    """Hits the real PV_Live API. Excluded from CI."""
    frame = fetch_outturn(
        entity_id=10,
        start=date(2025, 6, 1),
        end=date(2025, 6, 2),
        entity_type=EntityType.PES,
        use_cache=False,
    )
    assert not frame.empty
    assert {"generation_mw", "capacity_mwp", "time"} <= set(frame.columns)
    assert (frame["generation_mw"] >= 0).all()
