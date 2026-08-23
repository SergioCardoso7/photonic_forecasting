"""Data contracts for the silver layer.

A contract is a machine-checkable statement about what the data must look like,
enforced at a boundary. Violations fail the build rather than propagating into
a model that trains happily on nonsense.

Each check here corresponds to something known about the domain:

- Generation cannot be negative, and cannot exceed installed capacity by more
  than a small margin (could happen under certain specific conditions only,
  1.25x is generous headroom).
- The sun does not shine at night. A region reporting output while the sun is
  below the horizon means a timezone regression, a labelling bug, or a sensor
  fault, all of which are worth failing the build over.
- Every (region, hour) appears exactly once. Duplicates silently double-weight
  training rows.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pvlib

from pv.ingestion.regions import BY_ID

VALID_REGION_IDS: Final[frozenset[int]] = frozenset(BY_ID)

# Peak terrestrial GHI is ~1100 W/m2; allow headroom for cloud enhancement events.
MAX_GHI_W_M2: Final[float] = 1400.0

# Sun considered up above this elevation. Slightly negative to allow for
# refraction and for the fact that an hourly average straddles sunrise.
NIGHT_ELEVATION_DEG: Final[float] = -6.0

# Tolerance for "zero" generation at night, as a fraction of capacity.
NIGHT_LOAD_FACTOR_TOLERANCE: Final[float] = 0.01

# Generation may briefly exceed effective capacity; beyond this it is a fault.
MAX_LOAD_FACTOR: Final[float] = 1.25


def _solar_elevation(frame: pd.DataFrame) -> pd.Series:
    """Solar elevation in degrees for each (region, time) row.

    Computed per region because pvlib takes a single location at a time.
    """
    elevation = pd.Series(np.nan, index=frame.index, dtype="float64")
    for region_id in sorted({int(value) for value in frame["region_id"]}):
        mask = frame["region_id"] == region_id
        region = BY_ID[region_id]
        position = pvlib.solarposition.get_solarposition(
            pd.DatetimeIndex(frame.loc[mask, "time"]),
            latitude=region.latitude,
            longitude=region.longitude,
        )
        elevation.loc[mask] = position["apparent_elevation"].to_numpy()

    return elevation


def _no_generation_at_night(frame: pd.DataFrame) -> bool:
    """Assert load factor is ~zero whenever the sun is below the horizon.
    With some headroom in order to reach absolute nightime and the effects of refraction of the
    last rays of the sun
    """
    night = _solar_elevation(frame) < NIGHT_ELEVATION_DEG
    if not night.any():
        return True
    return bool((frame.loc[night, "load_factor"].abs() <= NIGHT_LOAD_FACTOR_TOLERANCE).all())


SILVER_PV: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns={
        "region_id": pa.Column(int, pa.Check.isin(VALID_REGION_IDS)),
        "time": pa.Column("datetime64[ns, UTC]", nullable=False),
        "generation_mw": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "capacity_mwp": pa.Column(float, pa.Check.gt(0.0), nullable=False),
        "installedcapacity_mwp": pa.Column(float, pa.Check.gt(0.0), nullable=True),
        "load_factor": pa.Column(
            float,
            [pa.Check.ge(0.0), pa.Check.le(MAX_LOAD_FACTOR)],
            nullable=False,
        ),
    },
    checks=[
        pa.Check(_no_generation_at_night, error="generation reported while sun below horizon"),
    ],
    unique=["region_id", "time"],
    strict=True,
    coerce=True,
    name="silver_pv",
)


SILVER_WEATHER: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns={
        "region_id": pa.Column(int, pa.Check.isin(VALID_REGION_IDS)),
        "time": pa.Column("datetime64[ns, UTC]", nullable=False),
        "source": pa.Column(
            str,
            pa.Check.isin({"era5", "historical_forecast", "previous_runs", "live_forecast"}),
        ),
        "lead_days": pa.Column(int, pa.Check.in_range(0, 7)),
        "shortwave_radiation": pa.Column(
            float, pa.Check.in_range(0.0, MAX_GHI_W_M2), nullable=False
        ),
        "direct_radiation": pa.Column(float, pa.Check.in_range(0.0, MAX_GHI_W_M2), nullable=True),
        "diffuse_radiation": pa.Column(float, pa.Check.in_range(0.0, MAX_GHI_W_M2), nullable=True),
        "direct_normal_irradiance": pa.Column(
            float, pa.Check.in_range(0.0, MAX_GHI_W_M2), nullable=True
        ),
        "temperature_2m": pa.Column(float, pa.Check.in_range(-55.0, 55.0), nullable=True),
        "relative_humidity_2m": pa.Column(float, pa.Check.in_range(0.0, 100.0), nullable=True),
        "dew_point_2m": pa.Column(float, pa.Check.in_range(-40.0, 40.0), nullable=True),
        "cloud_cover": pa.Column(float, pa.Check.in_range(0.0, 100.0), nullable=True),
        "cloud_cover_low": pa.Column(float, pa.Check.in_range(0.0, 100.0), nullable=True),
        "cloud_cover_mid": pa.Column(float, pa.Check.in_range(0.0, 100.0), nullable=True),
        "cloud_cover_high": pa.Column(float, pa.Check.in_range(0.0, 100.0), nullable=True),
        "wind_speed_10m": pa.Column(float, pa.Check.in_range(0.0, 200.0), nullable=True),
        "surface_pressure": pa.Column(float, pa.Check.in_range(870.0, 1085.0), nullable=True),
        "precipitation": pa.Column(float, pa.Check.ge(0.0), nullable=True),
        "is_day": pa.Column(float, pa.Check.isin({0.0, 1.0}), nullable=True),
    },
    unique=["region_id", "time", "source", "lead_days"],
    strict=True,
    coerce=True,
    name="silver_weather",
)


__all__ = ["SILVER_PV", "SILVER_WEATHER"]
