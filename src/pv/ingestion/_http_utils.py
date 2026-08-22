"""Shared HTTP plumbing for ingestion clients.

Extracted from the Open-Meteo client once PV_Live needed the same behaviour:
bounded retries with exponential backoff, and raw responses cached on disk by
request hash so re-running a backfill costs nothing and hits no API.

Both upstream APIs throttle with HTTP 429, so every client goes through
``get_json`` rather than calling httpx directly.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Final

import httpx
import structlog

from pv.config import settings

log = structlog.get_logger(__name__)

DEFAULT_HEADERS: Final[dict[str, str]] = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}


class IngestionError(RuntimeError):
    """Base class for every failure to retrieve data from an upstream API.

    Client modules subclass this so callers can catch either a specific source
    (``OpenMeteoError``) or any ingestion failure at all.
    """


def cache_key(url: str, params: dict[str, Any]) -> str:
    """Return a stable short hash identifying a request."""
    payload = json.dumps({"url": url, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(namespace: str, url: str, params: dict[str, Any]) -> Path:
    """Return the on-disk location for a cached raw response.

    Resolved at call time rather than import time so tests can redirect
    ``settings.data_root`` to a temporary directory.
    """
    return settings.data_root / "bronze" / "_raw" / namespace / f"{cache_key(url, params)}.json"


def get_json(
    url: str,
    params: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    max_attempts: int = 5,
    initial_backoff_s: float = 1.0,
) -> dict[str, Any]:
    """GET a JSON payload, retrying on rate limits and transient failures.

    Backing off rather than failing is what allows an unattended multi-year,
    multi-region backfill to complete without supervision.
    """
    backoff = initial_backoff_s
    last_error: Exception | None = None
    owns_client = client is None
    active = client or httpx.Client(timeout=settings.request_timeout_s, headers=DEFAULT_HEADERS)

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = active.get(url, params=params)
                if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                    raise IngestionError("rate limited")
                if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
                    raise IngestionError(f"server error {response.status_code}")
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                return payload
            except (IngestionError, httpx.HTTPError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                log.warning("http.retry", attempt=attempt, wait_s=backoff, error=str(exc), url=url)
                time.sleep(backoff)
                backoff *= 2
    finally:
        if owns_client:
            active.close()

    raise IngestionError(f"failed after {max_attempts} attempts: {last_error}") from last_error


def get_json_cached(
    namespace: str,
    url: str,
    params: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Fetch JSON, reading from and writing to the raw-response cache.

    Pass ``use_cache=False`` for anything whose value depends on when it was
    fetched: live forecasts and point-in-time vintage captures.
    """
    path = cache_path(namespace, url, params)

    if use_cache and path.exists():
        log.debug("http.cache_hit", path=str(path))
        cached: dict[str, Any] = json.loads(path.read_text())
        return cached

    payload = get_json(url, params, client=client)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    return payload
