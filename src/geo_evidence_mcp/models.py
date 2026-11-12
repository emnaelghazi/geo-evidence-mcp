"""Shared data models for the data layer (R1). Full provenance schema arrives in M3 (R4)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BBox(BaseModel):
    """A WGS84 (EPSG:4326) bounding box in degrees."""

    model_config = ConfigDict(frozen=True)

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if not (self.west < self.east and self.south < self.north):
            raise ValueError("bbox must satisfy west < east and south < north")
        return self

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


class AssetRef(BaseModel):
    """One STAC asset plus what is needed to turn its digital numbers into physical values."""

    model_config = ConfigDict(frozen=True)

    href: str
    scale: float = 1.0
    offset: float = 0.0
    nodata: float | None = None


class SceneSummary(BaseModel):
    """The subset of a STAC item the data layer needs, and that provenance will later cite."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    collection: str
    datetime: datetime
    cloud_cover: float | None = None
    processing_baseline: str | None = None
    epsg: int | None = None
    self_href: str | None = None
    assets: dict[str, AssetRef]
