"""FastMCP server (R1, M2): validation and error mapping only; logic lives in service.py."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, Literal
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field
from pystac_client.exceptions import APIError

from geo_evidence_mcp import __version__, service
from geo_evidence_mcp.config import get_settings
from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.replay import replay_block
from geo_evidence_mcp.results import (
    IndexResult,
    ProvenanceLookup,
    SearchResult,
    WindowResult,
)
from geo_evidence_mcp.stac import StacClientLike
from geo_evidence_mcp.store import ProvenanceStore, make_store

BBoxArg = Annotated[
    list[float],
    Field(min_length=4, max_length=4, description="[west, south, east, north], EPSG:4326 degrees"),
]
IndexArg = Annotated[Literal["ndvi", "ndwi"], Field(description="Spectral index to compute")]
ItemIdArg = Annotated[str, Field(description="STAC item id, e.g. from search_stac")]
RunIdArg = Annotated[
    str | None, Field(description="Optional agent run id, stored in the trace to correlate calls")
]
GeometryArg = Annotated[
    dict[str, Any], Field(description="GeoJSON Polygon or MultiPolygon geometry in EPSG:4326")
]

INSTRUCTIONS = (
    "Earth-observation tools over Sentinel-2 L2A. Every numeric result carries a provenance "
    "block (scene, bands, CRS, parameters, trace) and is stored under its block_id. Quote "
    "numbers only as returned and cite the block; if a tool refuses or returns no data, say so "
    "instead of estimating."
)


def _guard[T](fn: Callable[[], T]) -> T:
    """Turn expected failures into clean tool errors the agent can read and act on."""
    try:
        return fn()
    except (ValueError, LookupError, NotImplementedError, OSError, APIError) as exc:
        raise ToolError(str(exc)) from exc


def _bbox(values: list[float]) -> BBox:
    return BBox(west=values[0], south=values[1], east=values[2], north=values[3])


def build_server(
    client: StacClientLike | None = None, store: ProvenanceStore | None = None
) -> FastMCP:
    """Create the server. `client` and `store` let tests inject a fake STAC API and storage."""
    mcp = FastMCP("geo-evidence-mcp", instructions=INSTRUCTIONS)
    blocks = store if store is not None else make_store(get_settings())

    @mcp.tool
    def health() -> dict[str, str]:
        """Report server status and configuration (no network access, no secrets)."""
        return {
            "status": "ok",
            "version": __version__,
            "stac_endpoint": get_settings().stac_endpoint,
        }

    @mcp.tool
    def search_stac(
        bbox: BBoxArg,
        start: date,
        end: date,
        max_cloud: Annotated[float, Field(ge=0, le=100, description="Max cloud cover, %")] = 20.0,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        collection: str = service.DEFAULT_COLLECTION,
        run_id: RunIdArg = None,
    ) -> SearchResult:
        """Find Sentinel-2 scenes over an area and period, least cloudy first."""
        return _guard(
            lambda: service.search(
                _bbox(bbox),
                start,
                end,
                max_cloud=max_cloud,
                limit=limit,
                collection=collection,
                client=client,
                store=blocks,
                run_id=run_id,
            )
        )

    @mcp.tool
    def load_window(
        item_id: ItemIdArg,
        asset_key: Annotated[str, Field(description="Asset key such as red, nir, green, scl")],
        bbox: BBoxArg,
        collection: str = service.DEFAULT_COLLECTION,
        run_id: RunIdArg = None,
    ) -> WindowResult:
        """Read one band over an area; returns grid info and summary statistics, not pixels."""
        return _guard(
            lambda: service.window(
                item_id, asset_key, _bbox(bbox), collection=collection, client=client
            )
        )

    @mcp.tool
    def band_math(
        item_id: ItemIdArg,
        index: IndexArg,
        bbox: BBoxArg,
        collection: str = service.DEFAULT_COLLECTION,
        run_id: RunIdArg = None,
    ) -> IndexResult:
        """Compute NDVI or NDWI for a scene over a bounding box, with summary statistics."""
        return _guard(
            lambda: service.index_stats(
                item_id,
                index,
                bbox=_bbox(bbox),
                collection=collection,
                client=client,
                store=blocks,
                run_id=run_id,
            )
        )

    @mcp.tool
    def zonal_stats(
        item_id: ItemIdArg,
        index: IndexArg,
        geometry: GeometryArg,
        collection: str = service.DEFAULT_COLLECTION,
        run_id: RunIdArg = None,
    ) -> IndexResult:
        """Compute NDVI or NDWI statistics inside a polygon (pixel centres inside the polygon)."""
        return _guard(
            lambda: service.index_stats(
                item_id,
                index,
                geometry=geometry,
                collection=collection,
                client=client,
                store=blocks,
                run_id=run_id,
            )
        )

    @mcp.tool
    def get_provenance(
        block_id: Annotated[str, Field(description="Full block_id (UUID) from a tool result")],
        replay: Annotated[
            bool, Field(description="Re-run the recorded call and compare every number")
        ] = False,
    ) -> ProvenanceLookup:
        """Fetch a stored provenance block; optionally replay it to verify the numbers."""

        def lookup() -> ProvenanceLookup:
            block = blocks.get(UUID(block_id))
            if block is None:
                raise LookupError(f"no provenance block {block_id!r}")
            report = replay_block(block, client=client) if replay else None
            return ProvenanceLookup(block=block, replay=report)

        return _guard(lookup)

    return mcp


def main() -> int:
    build_server().run()  # stdio transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
