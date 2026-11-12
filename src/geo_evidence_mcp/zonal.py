"""Zonal statistics over a masked array (R1: zonal_stats)."""

from __future__ import annotations

from typing import Any

import numpy as np
from affine import Affine
from pydantic import BaseModel
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry


class ZonalStats(BaseModel):
    total_pixels: int  # pixels inside the zone (valid or not)
    valid_pixels: int
    valid_fraction: float
    mean: float | None
    median: float | None
    std: float | None  # population standard deviation (ddof=0)
    min: float | None
    max: float | None


def zone_mask(
    shape: tuple[int, int],
    transform: Affine,
    crs: str,
    geometry: BaseGeometry | dict[str, Any] | None = None,
    geometry_crs: str = "EPSG:4326",
) -> np.ndarray:
    """Boolean array, True where a pixel centre is inside `geometry` (everywhere when None)."""
    if geometry is None:
        return np.ones(shape, dtype=bool)
    geom = mapping(geometry) if isinstance(geometry, BaseGeometry) else geometry
    if geometry_crs != crs:
        geom = transform_geom(geometry_crs, crs, geom)
    mask: np.ndarray = geometry_mask([geom], out_shape=shape, transform=transform, invert=True)
    return mask


def zonal_stats(
    values: np.ma.MaskedArray,
    transform: Affine,
    crs: str,
    geometry: BaseGeometry | dict[str, Any] | None = None,
    geometry_crs: str = "EPSG:4326",
) -> ZonalStats:
    """Statistics of `values` inside `geometry` (whole array when None).

    A pixel counts as inside when its centre is inside the geometry (not all_touched).
    """
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array")
    masked = np.ma.getmaskarray(values)
    inside = zone_mask(values.shape, transform, crs, geometry, geometry_crs)
    total = int(inside.sum())
    selected = np.ma.getdata(values)[inside & ~masked].astype("float64")
    valid = int(selected.size)
    if valid == 0:
        return ZonalStats(
            total_pixels=total,
            valid_pixels=0,
            valid_fraction=0.0,
            mean=None,
            median=None,
            std=None,
            min=None,
            max=None,
        )
    return ZonalStats(
        total_pixels=total,
        valid_pixels=valid,
        valid_fraction=valid / total if total else 0.0,
        mean=float(selected.mean()),
        median=float(np.median(selected)),
        std=float(selected.std()),
        min=float(selected.min()),
        max=float(selected.max()),
    )
