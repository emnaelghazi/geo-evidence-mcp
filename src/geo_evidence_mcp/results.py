"""Typed tool results. `sources` mirrors the draft provenance schema (full block in M3)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from geo_evidence_mcp.zonal import ZonalStats


class SourceRef(BaseModel):
    """One asset a number was computed from."""

    item_id: str
    collection: str
    asset_key: str
    band: str | None
    href: str
    datetime: datetime
    processing_baseline: str | None
    cloud_cover: float | None
    scale: float
    offset: float
    nodata: float | None


class GridInfo(BaseModel):
    crs: str
    bounds: tuple[float, float, float, float]  # west, south, east, north in `crs`
    resolution: float  # pixel size in CRS units (metres for UTM)
    height: int
    width: int
    clipped: bool  # True when the AOI extended past the scene and was cut


class SceneBrief(BaseModel):
    item_id: str
    collection: str
    datetime: datetime
    cloud_cover: float | None
    processing_baseline: str | None
    epsg: int | None
    assets: list[str]


class SearchResult(BaseModel):
    count: int
    max_cloud: float
    scenes: list[SceneBrief]  # least cloudy first


class WindowResult(BaseModel):
    sources: list[SourceRef]
    grid: GridInfo
    stats: ZonalStats  # of the physical values (reflectance for optical bands)


class IndexResult(BaseModel):
    index: Literal["ndvi", "ndwi"]
    formula: str
    clip_min: float  # reflectance is clipped to this value before the ratio
    zone: Literal["bbox", "polygon"]
    sources: list[SourceRef]
    grid: GridInfo
    negative_reflectance_fraction: float  # share of in-zone pixels clipped (either band < 0)
    stats: ZonalStats
