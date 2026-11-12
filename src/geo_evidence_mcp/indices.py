"""Spectral indices as pure functions on masked arrays (feeds R1: band_math)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def normalized_difference(
    a: np.ma.MaskedArray,
    b: np.ma.MaskedArray,
    *,
    clip_min: float | None = 0.0,
) -> np.ma.MaskedArray:
    """(a - b) / (a + b); masked wherever an input is masked or a + b is ~0.

    L2A reflectance can be slightly negative (digital numbers below the +1000 offset, typically
    shadows and very dark surfaces). With a + b close to zero the ratio then explodes far outside
    [-1, 1]. Inputs are therefore clipped to `clip_min` (default 0.0) first, which bounds the
    result to [-1, 1]. Pass `clip_min=None` to get the raw, unbounded ratio.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a32, b32 = a.astype("float32"), b.astype("float32")
    if clip_min is not None:
        a32, b32 = np.ma.maximum(a32, clip_min), np.ma.maximum(b32, clip_min)
    total = a32 + b32
    diff = a32 - b32
    invalid = np.ma.getmaskarray(total) | np.ma.getmaskarray(diff)
    invalid |= np.abs(np.ma.filled(total, 1.0)) < _EPS
    safe_total = np.where(invalid, 1.0, np.ma.filled(total, 1.0))
    result = np.where(invalid, 0.0, np.ma.filled(diff, 0.0) / safe_total).astype("float32")
    return np.ma.masked_array(result, mask=invalid)


def ndvi(nir: np.ma.MaskedArray, red: np.ma.MaskedArray) -> np.ma.MaskedArray:
    return normalized_difference(nir, red)


def ndwi(green: np.ma.MaskedArray, nir: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """McFeeters NDWI (open water): (green - nir) / (green + nir)."""
    return normalized_difference(green, nir)
