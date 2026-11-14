"""The structured plan the language model must produce from a question (intake -> plan)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Period(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Period:
        if self.start > self.end:
            raise ValueError("period start must not be after its end")
        return self


class Plan(BaseModel):
    task: Literal["index_value", "index_change", "unsupported"] = Field(
        description="index_value: one number for one period; index_change: compare two periods; "
        "unsupported: the question cannot be answered with Sentinel-2 L2A (10 m optical) data"
    )
    index: Literal["ndvi", "ndwi"] | None = Field(
        default=None, description="ndvi = vegetation greenness; ndwi = open water"
    )
    place: str | None = Field(default=None, description="A named place from the known-places list")
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="[west, south, east, north] in EPSG:4326 degrees, only if the user gave one",
    )
    periods: list[Period] = Field(
        default_factory=list,
        description="One period for index_value; two for index_change (the model sorts them)",
    )
    unsupported_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _consistent(self) -> Plan:
        if self.task == "unsupported":
            if not self.unsupported_reason:
                raise ValueError("an unsupported plan needs unsupported_reason")
            return self
        if self.index is None:
            raise ValueError("index is required")
        if (self.place is None) == (self.bbox is None):
            raise ValueError("give exactly one of place or bbox")
        wanted = 1 if self.task == "index_value" else 2
        if len(self.periods) != wanted:
            raise ValueError(f"{self.task} needs exactly {wanted} period(s)")
        self.periods.sort(key=lambda p: p.start)
        if wanted == 2 and self.periods[0].end >= self.periods[1].start:
            raise ValueError("the two periods must not overlap")
        return self
