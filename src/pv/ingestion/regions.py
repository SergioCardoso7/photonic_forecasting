"""PES region metadata: identifiers, names and representative coordinates.

PV_Live returns generation for a *region*, but Open-Meteo needs a point. Each
region is therefore represented by a single coordinate, which is an
approximation: PV capacity within a DNO Licence Area is not uniformly
distributed, and a proper treatment would compute a capacity-weighted centroid
from the NESO GIS boundaries and Sheffield Solar's LLSOA capacity data.

That approximation is deliberate and scoped (see ADR 003). ``verify_against_api``
exists so it fails loudly rather than silently pairing the wrong coordinates
with the wrong region if PV_Live ever renumbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import structlog

from pv.ingestion.pv_live import EntityType, list_regions

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Region:
    """One PV_Live PES region and the point used to fetch its weather."""

    region_id: int
    code: str
    name: str
    latitude: float
    longitude: float


NATIONAL_ID: Final[int] = 0
# NOTE: verify these against `list_regions(EntityType.PES)` before the backfill.
# The ids and names below are a starting point, not an authority.
PES_REGIONS: Final[tuple[Region, ...]] = (
    Region(10, "_A", "UKPN (East)", 52.20, 0.50),
    Region(11, "_B", "WPD (East Midlands)", 52.95, -1.15),
    Region(12, "_C", "UKPN (London)", 51.51, -0.13),
    Region(13, "_D", "SPEN (SP MANWEB)", 53.19, -2.89),
    Region(14, "_E", "WPD (Midlands)", 52.48, -1.90),
    Region(15, "_F", "NPG (Northern Electric)", 54.90, -1.70),
    Region(16, "_G", "ENWL", 53.80, -2.60),
    Region(17, "_P", "SSE", 57.20, -3.80),
    Region(18, "_N", "SPEN (SP Distribution)", 55.85, -3.90),
    Region(19, "_J", "UKPN (South)", 51.15, 0.50),
    Region(20, "_H", "SSE (Southern)", 51.30, -1.20),
    Region(21, "_K", "WPD (South Wales)", 51.65, -3.60),
    Region(22, "_L", "WPD (South West)", 50.90, -3.50),
    Region(23, "_M", "NPG (Yorkshire Electric)", 53.85, -1.30),
)

BY_ID: Final[dict[int, Region]] = {region.region_id: region for region in PES_REGIONS}


def verify_against_api() -> None:
    """Check the local region table matches what PV_Live reports.

    Run once before the first backfill, and again whenever Sheffield Solar
    publish a boundary update. Raises if a region is missing or if an id maps
    to a different name than expected -- silently pairing Scottish weather
    with Cornish generation would train perfectly and be invisible.
    """
    api = list_regions(EntityType.PES)
    records = api[["pes_id", "pes_longname"]].to_dict("records")

    api_names: dict[int, str] = {
        int(record["pes_id"]): str(record["pes_longname"])
        for record in records
        if int(record["pes_id"]) != NATIONAL_ID
    }

    missing = api_names.keys() - BY_ID.keys()
    if missing:
        raise ValueError(f"PES regions with no local metadata: {sorted(missing)}")

    stale = BY_ID.keys() - api_names.keys()
    if stale:
        log.warning("regions.stale_local_entries", region_ids=sorted(stale))

    mismatched = {
        region_id: (name, BY_ID[region_id].name)
        for region_id, name in api_names.items()
        if BY_ID[region_id].name != name
    }
    if mismatched:
        raise ValueError(f"region name mismatches (api, local): {mismatched}")

    print(f"{'id':>4}  {'name':<40}  lat, lon")
    for region_id, name in sorted(api_names.items()):
        local = BY_ID[region_id]
        print(f"{region_id:>4}  {name:<40}  {local.latitude:.2f}, {local.longitude:.2f}")
