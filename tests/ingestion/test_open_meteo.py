"""Tests for the Open-Meteo client.

Unit tests use httpx.MockTransport, so they are deterministic and never touch
the network. The single integration test is marked and excluded from CI.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pandas as pd
import pytest

from pv.ingestion.open_meteo import (
    OpenMeteoError,
    WeatherSource,
    build_params,
    fetch_hourly,
    to_frame,
    write_bronze,
)


def _payload(times: list[str], **series: list[Any]) -> dict[str, Any]:
    return {
        "latitude": 22.3,
        "longitude": 114.2,
        "hourly": {"time": times, **series},
    }


def _mock_client(payload: dict[str, Any], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestBuildParams:
    def test_era5_requires_dates(self):
        with pytest.raises(ValueError, match="start and end are required"):
            build_params(22.3, 114.2, WeatherSource.ERA5)

    def test_rejects_reversed_date_range(self):
        with pytest.raises(ValueError, match="is after end"):
            build_params(
                22.3, 114.2, WeatherSource.ERA5, start=date(2024, 5, 2), end=date(2024, 5, 1)
            )

    def test_previous_runs_requires_lead_days(self):
        with pytest.raises(ValueError, match="lead_days is required"):
            build_params(
                22.3,
                114.2,
                WeatherSource.PREVIOUS_RUNS,
                start=date(2024, 5, 1),
                end=date(2024, 5, 2),
            )

    @pytest.mark.parametrize("lead", [0, 8, -1])
    def test_previous_runs_rejects_out_of_range_lead(self, lead):
        with pytest.raises(ValueError, match="between 1 and 7"):
            build_params(
                22.3,
                114.2,
                WeatherSource.PREVIOUS_RUNS,
                start=date(2024, 5, 1),
                end=date(2024, 5, 2),
                lead_days=lead,
            )

    def test_previous_runs_suffixes_every_variable(self):
        params = build_params(
            22.3,
            114.2,
            WeatherSource.PREVIOUS_RUNS,
            start=date(2024, 5, 1),
            end=date(2024, 5, 2),
            variables=["temperature_2m", "shortwave_radiation"],
            lead_days=1,
        )
        assert params["hourly"] == (
            "temperature_2m_previous_day1,shortwave_radiation_previous_day1"
        )

    def test_lead_days_rejected_for_non_previous_runs(self):
        with pytest.raises(ValueError, match="only meaningful for previous runs"):
            build_params(
                22.3,
                114.2,
                WeatherSource.ERA5,
                start=date(2024, 5, 1),
                end=date(2024, 5, 2),
                lead_days=1,
            )

    def test_live_forecast_uses_forecast_days_not_dates(self):
        params = build_params(22.3, 114.2, WeatherSource.LIVE_FORECAST, forecast_days=2)
        assert params["forecast_days"] == 2
        assert "start_date" not in params

    def test_always_requests_utc(self):
        params = build_params(
            22.3, 114.2, WeatherSource.ERA5, start=date(2024, 5, 1), end=date(2024, 5, 1)
        )
        assert params["timezone"] == "UTC"


class TestToFrame:
    def test_normalises_previous_run_column_names(self):
        payload = _payload(
            ["2024-05-01T00:00", "2024-05-01T01:00"],
            temperature_2m_previous_day1=[20.0, 21.0],
        )
        frame = to_frame(payload, WeatherSource.PREVIOUS_RUNS, lead_days=1)
        assert "temperature_2m" in frame.columns
        assert not any("previous_day" in c for c in frame.columns)

    def test_records_provenance(self):
        payload = _payload(["2024-05-01T00:00"], temperature_2m=[20.0])
        frame = to_frame(payload, WeatherSource.PREVIOUS_RUNS, lead_days=2)
        assert frame["source"].unique().tolist() == ["previous_runs"]
        assert frame["lead_days"].unique().tolist() == [2]

    def test_timestamps_are_utc_aware(self):
        payload = _payload(["2024-05-01T00:00"], temperature_2m=[20.0])
        frame = to_frame(payload, WeatherSource.ERA5, lead_days=None)
        assert str(frame["time"].dt.tz) == "UTC"

    def test_all_sources_produce_identical_columns(self):
        era5 = to_frame(
            _payload(["2024-05-01T00:00"], temperature_2m=[20.0]),
            WeatherSource.ERA5,
            lead_days=None,
        )
        prev = to_frame(
            _payload(["2024-05-01T00:00"], temperature_2m_previous_day1=[20.0]),
            WeatherSource.PREVIOUS_RUNS,
            lead_days=1,
        )
        assert sorted(era5.columns) == sorted(prev.columns)

    def test_rejects_payload_without_hourly_block(self):
        with pytest.raises(OpenMeteoError, match="no hourly block"):
            to_frame({"latitude": 22.3}, WeatherSource.ERA5, lead_days=None)


class TestFetchHourly:
    def test_returns_frame_from_mocked_response(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pv.ingestion.open_meteo.settings.data_root", tmp_path)
        payload = _payload(["2024-05-01T00:00", "2024-05-01T01:00"], temperature_2m=[20.0, 21.0])
        frame = fetch_hourly(
            22.3,
            114.2,
            WeatherSource.ERA5,
            start=date(2024, 5, 1),
            end=date(2024, 5, 1),
            variables=["temperature_2m"],
            client=_mock_client(payload),
        )
        assert len(frame) == 2

    def test_second_call_is_served_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pv.ingestion.open_meteo.settings.data_root", tmp_path)
        payload = _payload(["2024-05-01T00:00"], temperature_2m=[20.0])
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=payload)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        kwargs = {
            "latitude": 22.3,
            "longitude": 114.2,
            "source": WeatherSource.ERA5,
            "start": date(2024, 5, 1),
            "end": date(2024, 5, 1),
            "variables": ["temperature_2m"],
            "client": client,
        }
        fetch_hourly(**kwargs)
        fetch_hourly(**kwargs)
        assert calls["n"] == 1

    def test_raises_after_exhausting_retries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pv.ingestion.open_meteo.settings.data_root", tmp_path)
        monkeypatch.setattr("pv.ingestion.open_meteo.time.sleep", lambda _: None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(OpenMeteoError, match="failed after"):
            fetch_hourly(
                22.3,
                114.2,
                WeatherSource.ERA5,
                start=date(2024, 5, 1),
                end=date(2024, 5, 1),
                variables=["temperature_2m"],
                client=client,
            )


class TestWriteBronze:
    def test_splits_by_month_and_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pv.ingestion.open_meteo.settings.data_root", tmp_path)
        times = pd.date_range("2024-05-30", "2024-06-02", freq="h", tz="UTC")
        frame = pd.DataFrame(
            {"time": times, "temperature_2m": range(len(times)), "source": "era5", "lead_days": 0}
        )

        first = write_bronze(frame, "HK07", WeatherSource.ERA5, lead_days=None)
        second = write_bronze(frame, "HK07", WeatherSource.ERA5, lead_days=None)

        assert len(first) == 2
        assert {p.name for p in first} == {"month=2024-05.parquet", "month=2024-06.parquet"}
        assert first == second

    def test_empty_frame_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pv.ingestion.open_meteo.settings.data_root", tmp_path)
        assert write_bronze(pd.DataFrame(), "HK07", WeatherSource.ERA5, None) == []


@pytest.mark.integration
def test_live_api_returns_expected_variables():
    """Hits the real API. Run with `make test-all`, excluded from CI."""
    frame = fetch_hourly(
        latitude=22.3,
        longitude=114.2,
        source=WeatherSource.PREVIOUS_RUNS,
        start=date(2024, 6, 1),
        end=date(2024, 6, 2),
        variables=["temperature_2m", "shortwave_radiation"],
        lead_days=1,
        use_cache=False,
    )
    assert not frame.empty
    assert {"temperature_2m", "shortwave_radiation"} <= set(frame.columns)
    assert frame["time"].is_monotonic_increasing
