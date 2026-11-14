"""Named areas of interest used for development and the hand-verified M1 check."""

from __future__ import annotations

from geo_evidence_mcp.models import BBox

# Tozeur palm oasis, Tunisia: palms next to bare desert (about 2 x 2 km).
# The box was proposed from a Google Places reference point, not from imagery: m1_check reports
# whether it really contains both high- and low-NDVI ground.
TOZEUR = BBox(west=8.137, south=33.893, east=8.159, north=33.911)

# Named places the agent can resolve without a geocoder (ADR-005: geocoding is deferred).
GAZETTEER: dict[str, BBox] = {"tozeur oasis": TOZEUR}
_ALIASES = {
    "tozeur": "tozeur oasis",
    "tozeur_oasis": "tozeur oasis",
    "oasis of tozeur": "tozeur oasis",
}


def resolve_place(name: str) -> BBox | None:
    key = " ".join(name.lower().replace("-", " ").split())
    return GAZETTEER.get(_ALIASES.get(key, key))
