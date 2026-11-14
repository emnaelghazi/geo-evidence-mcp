"""Typed tool results. Every numeric result carries its provenance block (R4)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from geo_evidence_mcp.provenance import ProvenanceBlock
from geo_evidence_mcp.zonal import ZonalStats


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
    provenance: ProvenanceBlock


class WindowResult(BaseModel):
    grid: GridInfo
    stats: ZonalStats  # of the physical values (reflectance for optical bands)
    provenance: ProvenanceBlock


class IndexResult(BaseModel):
    index: Literal["ndvi", "ndwi"]
    formula: str
    clip_min: float  # reflectance is clipped to this value before the ratio
    zone: Literal["bbox", "polygon"]
    grid: GridInfo
    negative_reflectance_fraction: float  # share of in-zone pixels clipped (either band < 0)
    stats: ZonalStats
    provenance: ProvenanceBlock


class ReplayMismatch(BaseModel):
    name: str
    subject: str | None
    stored: float
    replayed: float


class ReplayReport(BaseModel):
    matches: bool
    compared: int
    mismatches: list[ReplayMismatch]
    missing_in_replay: list[str]
    extra_in_replay: list[str]


class ProvenanceLookup(BaseModel):
    block: ProvenanceBlock
    replay: ReplayReport | None = None
