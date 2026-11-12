"""Windowed COG reads (R1: load_window)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from geo_evidence_mcp.models import AssetRef, BBox, SceneSummary

# Settings that make remote COG reads efficient and anonymous-S3-safe.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


@dataclass(frozen=True)
class WindowData:
    """Pixels for an AOI, in physical units, with everything needed to cite them."""

    item_id: str
    asset_key: str
    href: str
    crs: str
    transform: Affine
    window: tuple[int, int, int, int]  # col_off, row_off, width, height
    bounds: tuple[float, float, float, float]  # in `crs`
    resolution: tuple[float, float]
    scale: float
    offset: float
    nodata: float | None
    clipped: bool  # True when the AOI extended beyond the asset and was cut
    values: np.ma.MaskedArray  # float32; masked where nodata


def _pixel_window(src: rasterio.DatasetReader, bbox: BBox) -> tuple[Window, bool]:
    if src.transform.b != 0 or src.transform.d != 0:
        raise NotImplementedError("rotated rasters are not supported")
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", src.crs, *bbox.as_tuple, densify_pts=21
    )
    win = from_bounds(left, bottom, right, top, transform=src.transform)
    col0, row0 = math.floor(win.col_off), math.floor(win.row_off)
    col1, row1 = math.ceil(win.col_off + win.width), math.ceil(win.row_off + win.height)
    c0, r0 = max(col0, 0), max(row0, 0)
    c1, r1 = min(col1, src.width), min(row1, src.height)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("the AOI does not overlap this asset")
    clipped = (c0, r0, c1, r1) != (col0, row0, col1, row1)
    return Window(c0, r0, c1 - c0, r1 - r0), clipped


def read_window(
    href: str,
    ref: AssetRef,
    bbox: BBox,
    *,
    item_id: str,
    asset_key: str,
    max_pixels: int | None = None,
) -> WindowData:
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        win, clipped = _pixel_window(src, bbox)
        if max_pixels is not None and win.width * win.height > max_pixels:
            raise ValueError(
                f"AOI covers {int(win.width * win.height):,} pixels, above the limit of "
                f"{max_pixels:,}; use a smaller area"
            )
        raw = src.read(1, window=win, masked=True)
        nodata = ref.nodata if ref.nodata is not None else src.nodata
        values = np.ma.masked_array(
            raw.astype("float32") * np.float32(ref.scale) + np.float32(ref.offset),
            mask=np.ma.getmaskarray(raw),
        )
        transform = src.window_transform(win)
        return WindowData(
            item_id=item_id,
            asset_key=asset_key,
            href=href,
            crs=src.crs.to_string(),
            transform=transform,
            window=(int(win.col_off), int(win.row_off), int(win.width), int(win.height)),
            bounds=window_bounds(win, src.transform),
            resolution=(abs(transform.a), abs(transform.e)),
            scale=ref.scale,
            offset=ref.offset,
            nodata=nodata,
            clipped=clipped,
            values=values,
        )


def load_window(
    scene: SceneSummary, asset_key: str, bbox: BBox, *, max_pixels: int | None = None
) -> WindowData:
    """Read one asset of `scene` over `bbox` and convert it to physical values (float32)."""
    try:
        ref = scene.assets[asset_key]
    except KeyError:
        raise KeyError(
            f"scene {scene.item_id} has no asset {asset_key!r}; available: {sorted(scene.assets)}"
        ) from None
    return read_window(
        ref.href, ref, bbox, item_id=scene.item_id, asset_key=asset_key, max_pixels=max_pixels
    )
