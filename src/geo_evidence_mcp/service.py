"""Tool logic as plain functions (R1). The MCP layer in server.py only wraps these."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import numpy as np
from shapely.geometry import shape

from geo_evidence_mcp.indices import normalized_difference
from geo_evidence_mcp.models import BBox, SceneSummary
from geo_evidence_mcp.raster import WindowData, load_window
from geo_evidence_mcp.results import (
    GridInfo,
    IndexResult,
    SceneBrief,
    SearchResult,
    SourceRef,
    WindowResult,
)
from geo_evidence_mcp.stac import (
    S2_BAND_TO_ASSET,
    StacClientLike,
    fetch_scene,
)
from geo_evidence_mcp.stac import search_stac as _search_stac
from geo_evidence_mcp.zonal import ZonalStats, zonal_stats, zone_mask

DEFAULT_COLLECTION = "sentinel-2-l2a"
MAX_WINDOW_PIXELS = 4_000_000  # about 20 x 20 km at 10 m; larger requests are refused

CLIP_MIN = 0.0
# index -> (formula, first asset, second asset): (first - second) / (first + second)
INDICES: dict[str, tuple[str, str, str]] = {
    "ndvi": ("(nir - red) / (nir + red)", "nir", "red"),
    "ndwi": ("(green - nir) / (green + nir)", "green", "nir"),
}
_ASSET_TO_BAND = {asset: band for band, asset in S2_BAND_TO_ASSET.items()}


def _source(scene: SceneSummary, asset_key: str) -> SourceRef:
    ref = scene.assets[asset_key]
    return SourceRef(
        item_id=scene.item_id,
        collection=scene.collection,
        asset_key=asset_key,
        band=_ASSET_TO_BAND.get(asset_key),
        href=ref.href,
        datetime=scene.datetime,
        processing_baseline=scene.processing_baseline,
        cloud_cover=scene.cloud_cover,
        scale=ref.scale,
        offset=ref.offset,
        nodata=ref.nodata,
    )


def _grid(window: WindowData) -> GridInfo:
    height, width = window.values.shape
    return GridInfo(
        crs=window.crs,
        bounds=window.bounds,
        resolution=window.resolution[0],
        height=height,
        width=width,
        clipped=window.clipped,
    )


def search(
    bbox: BBox,
    start: date,
    end: date,
    *,
    max_cloud: float = 20.0,
    limit: int = 20,
    collection: str = DEFAULT_COLLECTION,
    client: StacClientLike | None = None,
) -> SearchResult:
    scenes = _search_stac(
        bbox, start, end, collection=collection, max_cloud=max_cloud, limit=limit, client=client
    )
    return SearchResult(
        count=len(scenes),
        max_cloud=max_cloud,
        scenes=[
            SceneBrief(
                item_id=s.item_id,
                collection=s.collection,
                datetime=s.datetime,
                cloud_cover=s.cloud_cover,
                processing_baseline=s.processing_baseline,
                epsg=s.epsg,
                assets=sorted(s.assets),
            )
            for s in scenes
        ],
    )


def window(
    item_id: str,
    asset_key: str,
    bbox: BBox,
    *,
    collection: str = DEFAULT_COLLECTION,
    client: StacClientLike | None = None,
) -> WindowResult:
    scene = fetch_scene(item_id, collection=collection, client=client)
    data = load_window(scene, asset_key, bbox, max_pixels=MAX_WINDOW_PIXELS)
    return WindowResult(
        sources=[_source(scene, asset_key)],
        grid=_grid(data),
        stats=zonal_stats(data.values, data.transform, data.crs),
    )


def _geometry_bbox(geometry: dict[str, Any]) -> BBox:
    geom = shape(geometry)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"geometry must be a Polygon or MultiPolygon, got {geom.geom_type}")
    if geom.is_empty or not geom.is_valid:
        raise ValueError("geometry is empty or invalid")
    west, south, east, north = geom.bounds
    return BBox(west=west, south=south, east=east, north=north)


def index_stats(
    item_id: str,
    index: Literal["ndvi", "ndwi"],
    *,
    bbox: BBox | None = None,
    geometry: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
    client: StacClientLike | None = None,
) -> IndexResult:
    """Spectral-index statistics for a scene over a bbox, or inside a polygon (EPSG:4326)."""
    if (bbox is None) == (geometry is None):
        raise ValueError("provide exactly one of bbox or geometry")
    if index not in INDICES:
        raise ValueError(f"unknown index {index!r}; choose from {sorted(INDICES)}")
    formula, first_key, second_key = INDICES[index]
    aoi = bbox if bbox is not None else _geometry_bbox(geometry or {})

    scene = fetch_scene(item_id, collection=collection, client=client)
    first = load_window(scene, first_key, aoi, max_pixels=MAX_WINDOW_PIXELS)
    second = load_window(scene, second_key, aoi, max_pixels=MAX_WINDOW_PIXELS)
    if first.transform != second.transform or first.values.shape != second.values.shape:
        raise ValueError(f"{first_key} and {second_key} are on different pixel grids")

    values = normalized_difference(first.values, second.values, clip_min=CLIP_MIN)
    inside = zone_mask(values.shape, first.transform, first.crs, geometry)
    counted = inside & ~np.ma.getmaskarray(values)
    negative = (np.ma.getdata(first.values) < 0) | (np.ma.getdata(second.values) < 0)
    stats: ZonalStats = zonal_stats(values, first.transform, first.crs, geometry)
    return IndexResult(
        index=index,
        formula=formula,
        clip_min=CLIP_MIN,
        zone="polygon" if geometry is not None else "bbox",
        sources=[_source(scene, first_key), _source(scene, second_key)],
        grid=_grid(first),
        negative_reflectance_fraction=float(negative[counted].mean()) if counted.any() else 0.0,
        stats=stats,
    )
