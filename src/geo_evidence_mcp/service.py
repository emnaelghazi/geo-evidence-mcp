"""Tool logic as plain functions (R1, R4). The MCP layer in server.py only wraps these."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
from shapely.geometry import shape

from geo_evidence_mcp.config import get_settings
from geo_evidence_mcp.indices import normalized_difference
from geo_evidence_mcp.models import BBox, SceneSummary
from geo_evidence_mcp.provenance import (
    Computation,
    Output,
    ProvenanceBlock,
    Source,
    Spatial,
    Unit,
    bbox_polygon,
    build_block,
)
from geo_evidence_mcp.raster import WindowData, load_window
from geo_evidence_mcp.results import (
    GridInfo,
    IndexResult,
    SceneBrief,
    SearchResult,
    WindowResult,
)
from geo_evidence_mcp.stac import (
    S2_BAND_TO_ASSET,
    StacClientLike,
    fetch_scene,
)
from geo_evidence_mcp.stac import search_stac as _search_stac
from geo_evidence_mcp.store import ProvenanceStore
from geo_evidence_mcp.zonal import ZonalStats, zonal_stats, zone_mask

DEFAULT_COLLECTION = "sentinel-2-l2a"
MAX_WINDOW_PIXELS = 4_000_000  # about 20 x 20 km at 10 m; larger requests are refused

CLIP_MIN = 0.0
NODATA_POLICY = "pixels equal to the asset nodata value are masked and excluded from statistics"
# index -> (formula, first asset, second asset): (first - second) / (first + second)
INDICES: dict[str, tuple[str, str, str]] = {
    "ndvi": ("(nir - red) / (nir + red)", "nir", "red"),
    "ndwi": ("(green - nir) / (green + nir)", "green", "nir"),
}
_ASSET_TO_BAND = {asset: band for band, asset in S2_BAND_TO_ASSET.items()}
_PROCESSING_LEVEL = {"sentinel-2-l2a": "L2A", "sentinel-2-l1c": "L1C"}


def _source(scene: SceneSummary, asset_key: str) -> Source:
    ref = scene.assets[asset_key]
    return Source(
        stac_api=get_settings().stac_endpoint,
        collection=scene.collection,
        item_id=scene.item_id,
        item_href=scene.self_href,
        asset_key=asset_key,
        band=_ASSET_TO_BAND.get(asset_key),
        href=ref.href,
        datetime=scene.datetime,
        processing_level=_PROCESSING_LEVEL.get(scene.collection),
        processing_baseline=scene.processing_baseline,
        cloud_cover_pct=scene.cloud_cover,
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


def _spatial(grid: GridInfo, aoi_geometry: dict[str, Any]) -> Spatial:
    return Spatial(
        crs=grid.crs,
        bbox=grid.bounds,
        aoi_geometry=aoi_geometry,
        resolution=grid.resolution,
        height=grid.height,
        width=grid.width,
        clipped=grid.clipped,
    )


def _stat_outputs(prefix: str, stats: ZonalStats, unit: Unit) -> list[Output]:
    outputs = [
        Output(name=f"{prefix}.{key}", value=value, unit=unit)
        for key in ("mean", "median", "std", "min", "max")
        if (value := getattr(stats, key)) is not None
    ]
    outputs += [
        Output(name="zone.valid_fraction", value=stats.valid_fraction, unit="fraction"),
        Output(name="zone.valid_pixels", value=stats.valid_pixels, unit="count"),
        Output(name="zone.total_pixels", value=stats.total_pixels, unit="count"),
    ]
    return outputs


def _grid_outputs(grid: GridInfo) -> list[Output]:
    return [
        Output(name="grid.resolution", value=grid.resolution, unit="m"),
        Output(name="grid.width", value=grid.width, unit="count"),
        Output(name="grid.height", value=grid.height, unit="count"),
    ]


def _finish(block: ProvenanceBlock, store: ProvenanceStore | None) -> ProvenanceBlock:
    if store is not None:
        store.save(block)
    return block


def search(
    bbox: BBox,
    start: date,
    end: date,
    *,
    max_cloud: float = 20.0,
    limit: int = 20,
    collection: str = DEFAULT_COLLECTION,
    client: StacClientLike | None = None,
    store: ProvenanceStore | None = None,
    run_id: str | None = None,
) -> SearchResult:
    started, t0 = datetime.now(UTC), time.perf_counter()
    scenes = _search_stac(
        bbox, start, end, collection=collection, max_cloud=max_cloud, limit=limit, client=client
    )
    args: dict[str, Any] = {
        "bbox": list(bbox.as_tuple),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "max_cloud": max_cloud,
        "limit": limit,
        "collection": collection,
    }
    outputs = [Output(name="search.count", value=len(scenes), unit="count")]
    outputs += [
        Output(name="cloud_cover", value=s.cloud_cover, unit="percent", subject=s.item_id)
        for s in scenes
        if s.cloud_cover is not None
    ]
    block = build_block(
        tool="search_stac",
        args=args,
        sources=[],
        spatial=Spatial(crs="EPSG:4326", bbox=bbox.as_tuple, aoi_geometry=bbox_polygon(bbox)),
        computation=Computation(
            operation="stac_search",
            parameters={
                "stac_api": get_settings().stac_endpoint,
                "collection": collection,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "max_cloud": max_cloud,
                "limit": limit,
                "result_item_ids": [s.item_id for s in scenes],
            },
            nodata_policy="not applicable",
        ),
        outputs=outputs,
        started_at=started,
        duration_ms=(time.perf_counter() - t0) * 1000,
        run_id=run_id,
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
        provenance=_finish(block, store),
    )


def window(
    item_id: str,
    asset_key: str,
    bbox: BBox,
    *,
    collection: str = DEFAULT_COLLECTION,
    client: StacClientLike | None = None,
    store: ProvenanceStore | None = None,
    run_id: str | None = None,
) -> WindowResult:
    started, t0 = datetime.now(UTC), time.perf_counter()
    scene = fetch_scene(item_id, collection=collection, client=client)
    data = load_window(scene, asset_key, bbox, max_pixels=MAX_WINDOW_PIXELS)
    stats = zonal_stats(data.values, data.transform, data.crs)
    grid = _grid(data)
    unit: Unit = "class" if asset_key == "scl" else "reflectance"
    outputs = _stat_outputs(asset_key, stats, unit) + _grid_outputs(grid)
    if scene.cloud_cover is not None:
        outputs.append(
            Output(name="cloud_cover", value=scene.cloud_cover, unit="percent", subject=item_id)
        )
    block = build_block(
        tool="load_window",
        args={
            "item_id": item_id,
            "asset_key": asset_key,
            "bbox": list(bbox.as_tuple),
            "collection": collection,
        },
        sources=[_source(scene, asset_key)],
        spatial=_spatial(grid, bbox_polygon(bbox)),
        computation=Computation(
            operation="load_window",
            parameters={"statistics": "mean, median, std (population), min, max over valid pixels"},
            nodata_policy=NODATA_POLICY,
        ),
        outputs=outputs,
        started_at=started,
        duration_ms=(time.perf_counter() - t0) * 1000,
        run_id=run_id,
    )
    return WindowResult(grid=grid, stats=stats, provenance=_finish(block, store))


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
    store: ProvenanceStore | None = None,
    run_id: str | None = None,
) -> IndexResult:
    """Spectral-index statistics for a scene over a bbox, or inside a polygon (EPSG:4326)."""
    if (bbox is None) == (geometry is None):
        raise ValueError("provide exactly one of bbox or geometry")
    if index not in INDICES:
        raise ValueError(f"unknown index {index!r}; choose from {sorted(INDICES)}")
    started, t0 = datetime.now(UTC), time.perf_counter()
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
    stats = zonal_stats(values, first.transform, first.crs, geometry)
    negative_fraction = float(negative[counted].mean()) if counted.any() else 0.0
    grid = _grid(first)
    zone: Literal["bbox", "polygon"] = "polygon" if geometry is not None else "bbox"

    args: dict[str, Any] = {"item_id": item_id, "index": index, "collection": collection}
    args.update({"geometry": geometry} if geometry is not None else {"bbox": list(aoi.as_tuple)})
    outputs = _stat_outputs(index, stats, "index")
    outputs.append(
        Output(name="negative_reflectance_fraction", value=negative_fraction, unit="fraction")
    )
    outputs += _grid_outputs(grid)
    if scene.cloud_cover is not None:
        outputs.append(
            Output(name="cloud_cover", value=scene.cloud_cover, unit="percent", subject=item_id)
        )
    block = build_block(
        tool="zonal_stats" if geometry is not None else "band_math",
        args=args,
        sources=[_source(scene, first_key), _source(scene, second_key)],
        spatial=_spatial(grid, geometry if geometry is not None else bbox_polygon(aoi)),
        computation=Computation(
            operation=index,
            formula=formula,
            parameters={
                "clip_min": CLIP_MIN,
                "zone": zone,
                "pixel_inclusion": "pixel centre inside the zone",
                "max_window_pixels": MAX_WINDOW_PIXELS,
            },
            nodata_policy=NODATA_POLICY,
        ),
        outputs=outputs,
        started_at=started,
        duration_ms=(time.perf_counter() - t0) * 1000,
        run_id=run_id,
    )
    return IndexResult(
        index=index,
        formula=formula,
        clip_min=CLIP_MIN,
        zone=zone,
        grid=grid,
        negative_reflectance_fraction=negative_fraction,
        stats=stats,
        provenance=_finish(block, store),
    )


def execute(
    tool: str,
    args: dict[str, Any],
    *,
    client: StacClientLike | None = None,
    store: ProvenanceStore | None = None,
    run_id: str | None = None,
) -> SearchResult | WindowResult | IndexResult:
    """Run a tool from its recorded request: the basis of replay (R5)."""
    collection = args.get("collection", DEFAULT_COLLECTION)
    common: dict[str, Any] = {"client": client, "store": store, "run_id": run_id}
    if tool == "search_stac":
        west, south, east, north = args["bbox"]
        return search(
            BBox(west=west, south=south, east=east, north=north),
            date.fromisoformat(args["start"]),
            date.fromisoformat(args["end"]),
            max_cloud=args["max_cloud"],
            limit=args["limit"],
            collection=collection,
            **common,
        )
    if tool == "load_window":
        west, south, east, north = args["bbox"]
        return window(
            args["item_id"],
            args["asset_key"],
            BBox(west=west, south=south, east=east, north=north),
            collection=collection,
            **common,
        )
    if tool in ("band_math", "zonal_stats"):
        box = None
        if "bbox" in args:
            west, south, east, north = args["bbox"]
            box = BBox(west=west, south=south, east=east, north=north)
        return index_stats(
            args["item_id"],
            args["index"],
            bbox=box,
            geometry=args.get("geometry"),
            collection=collection,
            **common,
        )
    raise ValueError(f"cannot replay unknown tool {tool!r}")
