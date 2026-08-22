"""Shared test fixtures.

Two things every test in this suite needs, applied automatically so no test
has to remember them:

- ``settings.data_root`` points at a temporary directory, so the raw-response
  cache and any Parquet writes never touch the real ``data/`` folder.
- Retry backoff is a no-op, so a test that exercises the retry path finishes
  in milliseconds instead of fifteen seconds.

Patching ``settings`` works from any import path because every module imports
the same singleton instance; setting an attribute on it is visible everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pv.config import settings


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all filesystem writes into a per-test temporary directory."""
    root = tmp_path / "data"
    monkeypatch.setattr(settings, "data_root", root)
    return root


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make exponential backoff instant during tests."""
    monkeypatch.setattr("pv.ingestion._http_utils.time.sleep", lambda _: None)
