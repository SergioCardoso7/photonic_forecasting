"""Tests for the silver contracts.

Contracts are code, so they get tested like code. Each test constructs a frame
that is valid, then breaks exactly one thing and asserts the contract catches
it. A contract that never rejects anything is decoration.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from pv.contracts.silver import SILVER_PV, SILVER_WEATHER
from pv.transform.build_silver import _validate


def _pv_frame(days: int = 3, region_id: int = 20) -> pd.DataFrame:
    """A synthetic but physically plausible hourly generation series.

    A raised sine over daylight hours, zero at night, peaking near 12:00 UTC --
    close enough to real GB summer output for the contracts to accept it.
    """
    time = pd.date_range("2025-06-01", periods=24 * days, freq="h", tz="UTC")
    hour = time.hour.to_numpy(dtype=float)
    shape = np.sin(np.pi * (hour - 5) / 14)
    load_factor = np.where((hour >= 5) & (hour <= 19), np.clip(shape, 0, None) * 0.6, 0.0)
    capacity = 1000.0

    return pd.DataFrame(
        {
            "region_id": region_id,
            "time": time,
            "generation_mw": load_factor * capacity,
            "capacity_mwp": capacity,
            "installedcapacity_mwp": capacity * 1.1,
            "load_factor": load_factor,
        }
    )


def _weather_frame(rows: int = 48) -> pd.DataFrame:
    time = pd.date_range("2025-06-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "region_id": 20,
            "time": time,
            "source": "previous_runs",
            "lead_days": 1,
            "shortwave_radiation": 300.0,
            "direct_radiation": 200.0,
            "diffuse_radiation": 100.0,
            "direct_normal_irradiance": 400.0,
            "temperature_2m": 18.0,
            "relative_humidity_2m": 60.0,
            "dew_point_2m": 10.0,
            "cloud_cover": 40.0,
            "cloud_cover_low": 20.0,
            "cloud_cover_mid": 10.0,
            "cloud_cover_high": 10.0,
            "wind_speed_10m": 12.0,
            "surface_pressure": 1013.0,
            "precipitation": 0.0,
            "is_day": 1.0,
        }
    )


class TestSilverPV:
    def test_plausible_data_passes(self):
        assert not SILVER_PV.validate(_pv_frame()).empty

    def test_rejects_negative_generation(self):
        frame = _pv_frame()
        frame.loc[12, "generation_mw"] = -5.0
        with pytest.raises(pa.errors.SchemaError):
            SILVER_PV.validate(frame)

    def test_rejects_impossible_load_factor(self):
        frame = _pv_frame()
        frame.loc[12, "load_factor"] = 1.8
        with pytest.raises(pa.errors.SchemaError):
            SILVER_PV.validate(frame)

    def test_rejects_unknown_region(self):
        frame = _pv_frame(region_id=999)
        with pytest.raises(pa.errors.SchemaError):
            SILVER_PV.validate(frame)

    def test_rejects_duplicate_region_hour(self):
        frame = pd.concat([_pv_frame(days=1), _pv_frame(days=1)], ignore_index=True)
        with pytest.raises(pa.errors.SchemaError):
            SILVER_PV.validate(frame)

    def test_rejects_generation_at_night(self):
        """The check that catches every timestamp-convention bug."""
        frame = _pv_frame()
        frame.loc[2, "load_factor"] = 0.4
        frame.loc[2, "generation_mw"] = 400.0
        with pytest.raises(pa.errors.SchemaError):
            SILVER_PV.validate(frame)

    def test_rejects_a_six_hour_timestamp_shift(self):
        """Simulates the class of bug the night check exists for."""
        frame = _pv_frame()
        frame["time"] = frame["time"] + pd.Timedelta(hours=6)
        with pytest.raises(pa.errors.SchemaError):
            SILVER_PV.validate(frame)

    def test_rejects_unexpected_column(self):
        """strict=True: silver's shape is fixed, not whatever SQL emitted."""
        frame = _pv_frame().assign(mystery=1.0)
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            SILVER_PV.validate(frame)


class TestSilverWeather:
    def test_plausible_data_passes(self):
        assert not SILVER_WEATHER.validate(_weather_frame()).empty

    def test_rejects_impossible_irradiance(self):
        frame = _weather_frame()
        frame.loc[5, "shortwave_radiation"] = 5000.0
        with pytest.raises(pa.errors.SchemaError):
            SILVER_WEATHER.validate(frame)

    def test_rejects_null_irradiance(self):
        """The relabelling shift leaves a trailing NULL; SQL must drop it."""
        frame = _weather_frame()
        frame.loc[5, "shortwave_radiation"] = None
        with pytest.raises(pa.errors.SchemaError):
            SILVER_WEATHER.validate(frame)

    def test_rejects_unknown_source(self):
        frame = _weather_frame().assign(source="made_up")
        with pytest.raises(pa.errors.SchemaError):
            SILVER_WEATHER.validate(frame)

    def test_same_hour_allowed_once_per_source(self):
        """Three sources describing one hour is correct, not a duplicate."""
        era5 = _weather_frame(rows=24).assign(source="era5", lead_days=0)
        previous = _weather_frame(rows=24)
        combined = pd.concat([era5, previous], ignore_index=True)
        assert len(SILVER_WEATHER.validate(combined)) == 48

    def test_rejects_duplicate_within_one_source(self):
        frame = pd.concat([_weather_frame(rows=24)] * 2, ignore_index=True)
        with pytest.raises(pa.errors.SchemaError):
            SILVER_WEATHER.validate(frame)


class TestValidateGuard:
    def test_rejects_naive_timestamps(self):
        """coerce=True would localise silently, so the guard runs first."""
        frame = _pv_frame()
        frame["time"] = frame["time"].dt.tz_localize(None)
        with pytest.raises(ValueError, match="timezone-naive"):
            _validate(frame, SILVER_PV)

    def test_accepts_aware_timestamps(self):
        days = inspect.signature(_pv_frame).parameters["days"].default
        assert len(_validate(_pv_frame(), SILVER_PV)) == days * 24
