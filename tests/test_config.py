from pathlib import Path

from pv.config import Layer, Settings


def test_layer_path_composes_layer_and_dataset():
    s = Settings(data_root=Path("/tmp/lake"))
    assert s.layer_path(Layer.BRONZE, "pv") == Path("/tmp/lake/bronze/pv")


def test_partition_path_uses_hive_style():
    s = Settings(data_root=Path("/tmp/lake"))
    got = s.partition_path(Layer.SILVER, "weather", site_id="HK07", date="2024-03-11")
    assert got == Path("/tmp/lake/silver/weather/site_id=HK07/date=2024-03-11")


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("PV_DATA_ROOT", "/custom")
    assert Settings().data_root == Path("/custom")
