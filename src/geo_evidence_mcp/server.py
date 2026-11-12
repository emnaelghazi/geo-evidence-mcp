"""FastMCP server (R1, M2): validation and error mapping only; logic lives in service.py."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field
from pystac_client.exceptions import APIError

from geo_evidence_mcp import __version__, service
from geo_evidence_mcp.config import get_settings
from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.results import IndexResult, SearchResult, WindowResult
from geo_evidence_mcp.stac import StacClientLike

BBoxArg = Annotated[
    list[float],
    Field(min_length=4, max_length=4, description="[west, south, east, north], EPSG:4326 degrees"),
]
IndexArg = Annotated[Literal["ndvi", "ndwi"], Field(description="Spectral index to compute")]
ItemIdArg = Annotated[str, Field(description="STAC item id, e.g. from search_stac")]
GeometryArg = Annotated[
    dict[str, Any], Field(description="GeoJSON Polygon or MultiPolygon geometry in EPSG:4326")
]

INSTRUCTIONS = (
    "Earth-observation tools over Sentinel-2 L2A. Every numeric result lists the exact scene, "
    "bands, scale/offset and pixel grid it was computed from. Quote numbers only as returned; "
    "if a tool refuses or returns no data, say so instead of estimating."
)


def _guard[T](fn: Callable[[], T]) -> T:
    """Turn expected failures into clean tool errors the agent can read and act on."""
    try:
        return fn()
    except (ValueError, LookupError, NotImplementedError, OSError, APIError) as exc:
        raise ToolError(str(exc)) from exc


def _bbox(values: list[float]) -> BBox:
    return BBox(west=values[0], south=values[1], east=values[2], north=values[3])


def build_server(client: StacClientLike | None = None) -> FastMCP:
    """Create the server. `client` lets tests inject a fake STAC API."""
    mcp = FastMCP("geo-evidence-mcp", instructions=INSTRUCTIONS)

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
            )
        )

    @mcp.tool
    def load_window(
        item_id: ItemIdArg,
        asset_key: Annotated[str, Field(description="Asset key such as red, nir, green, scl")],
        bbox: BBoxArg,
        collection: str = service.DEFAULT_COLLECTION,
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
    ) -> IndexResult:
        """Compute NDVI or NDWI for a scene over a bounding box, with summary statistics."""
        return _guard(
            lambda: service.index_stats(
                item_id, index, bbox=_bbox(bbox), collection=collection, client=client
            )
        )

    @mcp.tool
    def zonal_stats(
        item_id: ItemIdArg,
        index: IndexArg,
        geometry: GeometryArg,
        collection: str = service.DEFAULT_COLLECTION,
    ) -> IndexResult:
        """Compute NDVI or NDWI statistics inside a polygon (pixel centres inside the polygon)."""
        return _guard(
            lambda: service.index_stats(
                item_id, index, geometry=geometry, collection=collection, client=client
            )
        )

    return mcp


def main() -> int:
    build_server().run()  # stdio transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
