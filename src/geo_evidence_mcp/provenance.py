"""The provenance block (R4): everything needed to audit, and replay, one tool call.

Pydantic is the source of truth; docs/provenance-schema-1.0.json is generated from it
(`uv run gem-export-schema`) and a test fails if the two drift apart.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from geo_evidence_mcp import __version__
from geo_evidence_mcp.models import BBox

SCHEMA_VERSION = "1.0"
SHORT_ID_LENGTH = 8

Unit = Literal["index", "reflectance", "class", "fraction", "count", "percent", "m"]
Operation = Literal["stac_search", "load_window", "ndvi", "ndwi"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolInfo(_Strict):
    name: str
    version: str


class Request(_Strict):
    """The tool call exactly as made: enough to run it again."""

    tool: str
    args: dict[str, Any]


class Source(_Strict):
    """One asset a number was computed from."""

    stac_api: str
    collection: str
    item_id: str
    item_href: str | None = None
    asset_key: str
    band: str | None = None
    href: str
    datetime: datetime
    processing_level: str | None = None  # e.g. L2A
    processing_baseline: str | None = None
    cloud_cover_pct: float | None = None
    scale: float
    offset: float
    nodata: float | None = None


class Spatial(_Strict):
    crs: str  # CRS of `bbox` and of the pixel grid
    bbox: tuple[float, float, float, float]  # west, south, east, north in `crs`
    aoi_geometry: dict[str, Any]  # GeoJSON, EPSG:4326: the area the caller asked for
    resolution: float | None = None  # pixel size in CRS units
    height: int | None = None
    width: int | None = None
    clipped: bool = False


class Computation(_Strict):
    operation: Operation
    formula: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    resampling: Literal["none"] = "none"  # all reads are on the native grid
    nodata_policy: str


class Output(_Strict):
    """One number the call produced. Claims in an answer are checked against these."""

    name: str
    value: float
    unit: Unit
    subject: str | None = None  # e.g. the item id when a search returns one value per scene


class ToolTrace(_Strict):
    trace_id: str
    span_id: str
    run_id: str | None = None  # agent run, when the caller supplies one
    started_at: datetime
    duration_ms: float = Field(ge=0)


class ProvenanceBlock(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    block_id: uuid.UUID
    generated_at: datetime
    tool: ToolInfo
    request: Request
    args_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    sources: list[Source]
    spatial: Spatial
    computation: Computation
    outputs: list[Output] = Field(min_length=1)
    tool_trace: ToolTrace

    @property
    def short_id(self) -> str:
        """Compact id an answer can cite, e.g. [3f9a1c2e]."""
        return self.block_id.hex[:SHORT_ID_LENGTH]


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return value


def compute_args_hash(tool: str, args: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the call: key order and float noise do not matter."""
    payload = json.dumps(
        _canonical({"tool": tool, "args": args}), sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bbox_polygon(bbox: BBox) -> dict[str, Any]:
    w, s, e, n = bbox.as_tuple
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def build_block(
    *,
    tool: str,
    args: dict[str, Any],
    sources: list[Source],
    spatial: Spatial,
    computation: Computation,
    outputs: list[Output],
    started_at: datetime,
    duration_ms: float,
    run_id: str | None = None,
) -> ProvenanceBlock:
    return ProvenanceBlock(
        block_id=uuid.uuid4(),
        generated_at=datetime.now(UTC),
        tool=ToolInfo(name=tool, version=__version__),
        request=Request(tool=tool, args=args),
        args_hash=compute_args_hash(tool, args),
        sources=sources,
        spatial=spatial,
        computation=computation,
        outputs=outputs,
        tool_trace=ToolTrace(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            run_id=run_id,
            started_at=started_at,
            duration_ms=duration_ms,
        ),
    )
