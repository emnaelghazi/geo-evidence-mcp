"""STAC search (R1: search_stac). Plain function; the MCP wrapper comes in M2."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date
from typing import Protocol

import pystac
import pystac_client

from geo_evidence_mcp.config import Settings, get_settings
from geo_evidence_mcp.models import AssetRef, BBox, SceneSummary

# Asset keys we keep from Earth Search v1 Sentinel-2 L2A items.
# TODO(M1 verify, live): confirm these keys against a real item (integration test does this).
S2_ASSET_KEYS = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22", "scl")
S2_BAND_TO_ASSET = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}
# From processing baseline 04.00, L2A digital numbers carry a +1000 offset:
# reflectance = (DN - 1000) / 10000.
_BASELINE_OFFSET_CHANGE = (4, 0)


class _Search(Protocol):
    def items(self) -> Iterable[pystac.Item]: ...


class StacClientLike(Protocol):
    @property
    def search(self) -> Callable[..., _Search]: ...


def _baseline_tuple(baseline: str | None) -> tuple[int, int] | None:
    if not baseline:
        return None
    try:
        major, minor = baseline.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def _scale_offset(asset: pystac.Asset, key: str, baseline: str | None) -> tuple[float, float]:
    """Use the asset's raster:bands scale/offset; else fall back to the processing-baseline rule."""
    bands = asset.extra_fields.get("raster:bands")
    if isinstance(bands, list) and bands and isinstance(bands[0], dict):
        scale = bands[0].get("scale")
        offset = bands[0].get("offset")
        if scale is not None and offset is not None:
            return float(scale), float(offset)
    if key == "scl":
        return 1.0, 0.0
    parsed = _baseline_tuple(baseline)
    offset = -0.1 if parsed is not None and parsed >= _BASELINE_OFFSET_CHANGE else 0.0
    return 1e-4, offset


def _nodata(asset: pystac.Asset) -> float | None:
    bands = asset.extra_fields.get("raster:bands")
    if isinstance(bands, list) and bands and isinstance(bands[0], dict):
        value = bands[0].get("nodata")
        if value is not None:
            return float(value)
    return None


def _epsg(item: pystac.Item) -> int | None:
    code = item.properties.get("proj:code")  # projection extension v2
    if isinstance(code, str) and code.upper().startswith("EPSG:"):
        return int(code.split(":")[1])
    epsg = item.properties.get("proj:epsg")  # projection extension v1
    return int(epsg) if epsg is not None else None


def summarize_item(item: pystac.Item) -> SceneSummary:
    baseline = item.properties.get("s2:processing_baseline")
    assets: dict[str, AssetRef] = {}
    for key in S2_ASSET_KEYS:
        asset = item.assets.get(key)
        if asset is None:
            continue
        scale, offset = _scale_offset(asset, key, baseline)
        assets[key] = AssetRef(href=asset.href, scale=scale, offset=offset, nodata=_nodata(asset))
    if item.datetime is None:
        raise ValueError(f"item {item.id} has no datetime")
    cloud = item.properties.get("eo:cloud_cover")
    return SceneSummary(
        item_id=item.id,
        collection=item.collection_id or "",
        datetime=item.datetime,
        cloud_cover=float(cloud) if cloud is not None else None,
        processing_baseline=str(baseline) if baseline is not None else None,
        epsg=_epsg(item),
        self_href=item.get_self_href(),
        assets=assets,
    )


def _active_client(client: StacClientLike | None, settings: Settings | None) -> StacClientLike:
    if client is not None:
        return client
    return pystac_client.Client.open((settings or get_settings()).stac_endpoint)


def fetch_scene(
    item_id: str,
    *,
    collection: str = "sentinel-2-l2a",
    client: StacClientLike | None = None,
    settings: Settings | None = None,
) -> SceneSummary:
    """Look one scene up by id (the server is stateless: tools take ids, not objects).

    TODO(M2 verify, live): confirm Earth Search v1 honours `ids` in item search.
    """
    search = _active_client(client, settings).search(
        collections=[collection], ids=[item_id], max_items=1
    )
    for item in search.items():
        if item.id == item_id:
            return summarize_item(item)
    raise LookupError(f"no item {item_id!r} in collection {collection!r}")


def search_stac(
    bbox: BBox,
    start: date,
    end: date,
    *,
    collection: str = "sentinel-2-l2a",
    max_cloud: float = 20.0,
    limit: int = 50,
    client: StacClientLike | None = None,
    settings: Settings | None = None,
) -> list[SceneSummary]:
    """Return scenes intersecting `bbox` in [start, end], least cloudy first (ties: newest first).

    The server-side cloud filter is a courtesy; the same filter is applied client-side so the
    result does not depend on the API honouring the `query` extension.
    TODO(M1 verify, live): confirm Earth Search v1 accepts `query={"eo:cloud_cover": {"lt": x}}`.
    """
    if start > end:
        raise ValueError("start must not be after end")
    search = _active_client(client, settings).search(
        collections=[collection],
        bbox=bbox.as_tuple,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    scenes = [summarize_item(item) for item in search.items()]
    scenes = [s for s in scenes if s.cloud_cover is None or s.cloud_cover < max_cloud]
    scenes.sort(key=lambda s: -s.datetime.timestamp())
    scenes.sort(key=lambda s: (s.cloud_cover is None, s.cloud_cover or 0.0))
    return scenes
