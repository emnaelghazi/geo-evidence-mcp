"""Small builders shared by provenance, validator and store tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.provenance import (
    Computation,
    Output,
    ProvenanceBlock,
    Spatial,
    Unit,
    bbox_polygon,
    build_block,
)

AOI = BBox(west=8.137, south=33.893, east=8.159, north=33.911)


def make_block(
    outputs: list[tuple[str, float, Unit]],
    *,
    tool: str = "band_math",
    args: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ProvenanceBlock:
    return build_block(
        tool=tool,
        args=args or {"item_id": "X", "index": "ndvi", "bbox": list(AOI.as_tuple)},
        sources=[],
        spatial=Spatial(crs="EPSG:4326", bbox=AOI.as_tuple, aoi_geometry=bbox_polygon(AOI)),
        computation=Computation(operation="ndvi", nodata_policy="test"),
        outputs=[Output(name=n, value=v, unit=u) for n, v, u in outputs],
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        run_id=run_id,
    )
