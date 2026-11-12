"""Synthetic Sentinel-2-like scene over the Tozeur AOI, with hand-computable values.

Layout (EPSG:32632, 10 m pixels): left half "palm", right half "desert", plus a 5x5 nodata hole.
Digital numbers carry the post-04.00 +1000 offset: reflectance = (DN - 1000) / 10000.

    band   palm DN (refl)   desert DN (refl)
    red    1500 (0.05)      3500 (0.25)
    nir    4500 (0.35)      4000 (0.30)
    green  1800 (0.08)      3300 (0.23)

    NDVI palm   = (0.35 - 0.05) / (0.35 + 0.05) = 0.75
    NDVI desert = (0.30 - 0.25) / (0.30 + 0.25) = 0.0909...
    (a naive ratio on raw DNs would give 0.5 for palm: the offset matters)
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import pytest
import rasterio
from affine import Affine
from rasterio.warp import transform

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.models import AssetRef, SceneSummary

CRS = "EPSG:32632"
SIZE = 450  # pixels per side (4.5 km), larger than the ~2 km AOI
DN = {
    "red": (1500, 3500),
    "nir": (4500, 4000),
    "green": (1800, 3300),
}


@dataclass(frozen=True)
class Synthetic:
    scene: SceneSummary
    transform: Affine
    split_col: int
    hole: tuple[slice, slice]


def _write(path: Path, data: np.ndarray, transform_: Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint16",
        crs=CRS,
        transform=transform_,
        nodata=0,
    ) as dst:
        dst.write(data, 1)


def make_scene(tmp_path: Path, *, offset: float = -0.1) -> Synthetic:
    (cx,), (cy,) = transform(
        "EPSG:4326",
        CRS,
        [(TOZEUR.west + TOZEUR.east) / 2],
        [(TOZEUR.south + TOZEUR.north) / 2],
    )
    x0 = math.floor((cx - SIZE * 5) / 10) * 10
    y0 = math.ceil((cy + SIZE * 5) / 10) * 10
    transform_ = Affine(10.0, 0.0, x0, 0.0, -10.0, y0)
    split_col = int((cx - x0) // 10)
    centre_row = int((y0 - cy) // 10)
    hole = (slice(centre_row - 2, centre_row + 3), slice(split_col - 30, split_col - 25))

    assets: dict[str, AssetRef] = {}
    for band, (palm, desert) in DN.items():
        data = np.full((SIZE, SIZE), desert, dtype="uint16")
        data[:, :split_col] = palm
        if band in ("red", "nir"):
            data[hole] = 0  # nodata
        path = tmp_path / f"{band}.tif"
        _write(path, data, transform_)
        assets[band] = AssetRef(href=str(path), scale=1e-4, offset=offset, nodata=0.0)

    scene = SceneSummary(
        item_id="SYNTH_TOZEUR_L2A",
        collection="sentinel-2-l2a",
        datetime=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
        cloud_cover=1.0,
        processing_baseline="05.11",
        epsg=32632,
        assets=assets,
    )
    return Synthetic(scene=scene, transform=transform_, split_col=split_col, hole=hole)


@pytest.fixture
def synthetic(tmp_path: Path) -> Synthetic:
    return make_scene(tmp_path)


def scene_to_item(scene: SceneSummary) -> pystac.Item:
    """Turn a SceneSummary back into a STAC item (for fake catalogs in server tests)."""
    item = pystac.Item(
        id=scene.item_id,
        geometry=None,
        bbox=None,
        datetime=scene.datetime,
        properties={
            "eo:cloud_cover": scene.cloud_cover,
            "s2:processing_baseline": scene.processing_baseline,
            "proj:code": f"EPSG:{scene.epsg}",
        },
        collection=scene.collection,
    )
    for key, ref in scene.assets.items():
        item.add_asset(
            key,
            pystac.Asset(
                href=ref.href,
                extra_fields={
                    "raster:bands": [
                        {"scale": ref.scale, "offset": ref.offset, "nodata": ref.nodata}
                    ]
                },
            ),
        )
    return item


class _FakeSearch:
    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items

    def items(self) -> Iterator[pystac.Item]:
        return iter(self._items)


class FakeCatalog:
    """Minimal STAC API stand-in: honours `ids`, otherwise returns everything it holds."""

    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items

    def search(self, **kwargs: Any) -> _FakeSearch:
        ids = kwargs.get("ids")
        found = [i for i in self._items if ids is None or i.id in ids]
        return _FakeSearch(found)
