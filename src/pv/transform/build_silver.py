"""Build the silver layer: run SQL against bronze, then enforce contracts.

Order matters. The SQL runs first and materialises a table; validation runs
second and *fails the build* if the result violates a contract. A silver table
that exists but is wrong is worse than no silver table, because everything
downstream will trust it.

Run as a module:

    uv run python -m pv.transform.silver
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import duckdb
import pandas as pd
import pandera.pandas as pa
import structlog

from pv.config import Layer, settings
from pv.contracts.silver import SILVER_PV, SILVER_WEATHER

log = structlog.get_logger(__name__)

SQL_DIR: Final[Path] = Path(__file__).parent / "sql"


def _bronze_globs() -> dict[str, str]:
    """Glob patterns for each bronze dataset, resolved from settings.

    Absolute so the build works regardless of the working directory.
    """
    bronze = settings.data_root / Layer.BRONZE.value
    return {
        "bronze_pv": str(bronze / "pv_live" / "**" / "*.parquet"),
        "bronze_weather": str(bronze / "weather" / "**" / "*.parquet"),
    }


def _run_sql(connection: duckdb.DuckDBPyConnection, name: str) -> None:
    """Execute one SQL file with bronze paths substituted in."""
    statement = (SQL_DIR / f"{name}.sql").read_text().format(**_bronze_globs())
    log.info("silver.build", table=name)
    connection.execute(statement)


def _validate(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate a table, reporting every failure rather than only the first.

    ``lazy=True`` matters: without it Pandera raises on the first bad column,
    so fixing data becomes a slow game of whack-a-mole.
    """
    if "time" in frame.columns and frame["time"].dt.tz is None:
        raise ValueError(f"{schema.name}: 'time' is timezone-naive, expected UTC")

    try:
        validated: pd.DataFrame = schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        log.error("contract.failed", table=schema.name, failures=len(exc.failure_cases))
        print(f"\n--- {schema.name} contract violations ---", file=sys.stderr)
        print(exc.failure_cases.to_string(max_rows=40), file=sys.stderr)
        raise
    log.info("contract.passed", table=schema.name, rows=len(validated))
    return validated


def _export(connection: duckdb.DuckDBPyConnection, name: str) -> Path:
    """Write a silver table out as Parquet alongside the database.

    The DuckDB table is what queries hit; the Parquet copy is what makes silver
    a layer of the lake rather than a private detail of one database file.
    """
    target = settings.layer_path(Layer.SILVER, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = target.with_suffix(".parquet")
    connection.execute(f"COPY {name} TO '{destination}' (FORMAT PARQUET)")
    return destination


def build_silver(validate: bool = True) -> dict[str, int]:
    """Rebuild the whole silver layer from bronze. Returns row counts."""
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with duckdb.connect(str(settings.duckdb_path)) as connection:
        for name, schema in (("silver_pv", SILVER_PV), ("silver_weather", SILVER_WEATHER)):
            _run_sql(connection, name)
            frame = connection.execute(f"SELECT * FROM {name}").df()

            if frame.empty:
                raise ValueError(f"{name} is empty -- has the backfill run?")

            if validate:
                _validate(frame, schema)

            destination = _export(connection, name)
            counts[name] = len(frame)
            log.info("silver.written", table=name, rows=len(frame), path=str(destination))

    return counts


def main() -> int:
    counts = build_silver()
    for name, rows in counts.items():
        print(f"{name:<16} {rows:>10,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
