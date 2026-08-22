"""Central configuration and data lake layout"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Layer(StrEnum):
    """Medallion layer of the data lake"""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class Settings(BaseSettings):
    """Runtime configuration"""

    model_config = SettingsConfigDict(env_prefix="PV_", env_file=".env")

    data_root: Path = Field(default=Path("data"))
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_historical_forecast_url: str = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_previous_runs_url: str = "https://previous-runs-api.open-meteo.com/v1/forecast"
    request_timeout_s: float = 30.0

    def layer_path(self, layer: Layer, dataset: str) -> Path:
        """Return the directory for a dataset within a lake layer"""
        return self.data_root / layer.value / dataset

    def partition_path(self, layer: Layer, dataset: str, site_id: str, date: str) -> Path:
        """Return a Hive-partitioned path: <layer>/<dataset>/site_id=X/date=Y."""
        return self.layer_path(layer, dataset) / f"site_id={site_id}" / f"date={date}"

    def duckdb_path(self) -> Path:
        """Database file co-located with the lake"""
        return self.data_root / "pv.duckdb"


settings = Settings()
