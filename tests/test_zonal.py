"""R1 (zonal_stats)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.indices import ndvi
from geo_evidence_mcp.raster import load_window
from geo_evidence_mcp.zonal import zonal_stats
from tests.conftest import Synthetic

CRS = "EPSG:32632"
TRANSFORM = Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 4000000.0)  # 1 m pixels


def _column_index_array() -> np.ma.MaskedArray:
    data = np.tile(np.arange(10, dtype="float32"), (10, 1))  # value = column index
    return np.ma.masked_array(data, mask=np.zeros((10, 10), dtype=bool))


@pytest.mark.req("R1")
def test_whole_array_with_masked_pixel() -> None:
    values = np.ma.masked_array(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        mask=[[False, False], [False, True]],
    )
    stats = zonal_stats(values, TRANSFORM, CRS)
    assert (stats.total_pixels, stats.valid_pixels) == (4, 3)
    assert stats.valid_fraction == pytest.approx(0.75)
    assert stats.mean == pytest.approx(2.0)
    assert stats.median == pytest.approx(2.0)
    assert stats.std == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert (stats.min, stats.max) == (1.0, 3.0)


@pytest.mark.req("R1")
def test_pixel_aligned_polygon_in_raster_crs() -> None:
    zone = box(500000, 3999995, 500005, 4000000)  # columns 0-4, rows 0-4 -> 25 pixels
    stats = zonal_stats(_column_index_array(), TRANSFORM, CRS, zone, geometry_crs=CRS)
    assert stats.total_pixels == 25
    assert stats.mean == pytest.approx(2.0)
    assert (stats.min, stats.max) == (0.0, 4.0)


@pytest.mark.req("R1")
def test_fully_masked_zone_returns_none_stats() -> None:
    values = np.ma.masked_all((4, 4), dtype="float32")
    stats = zonal_stats(values, TRANSFORM, CRS)
    assert stats.valid_pixels == 0 and stats.valid_fraction == 0.0
    assert stats.mean is None and stats.median is None and stats.std is None


@pytest.mark.req("R1")
def test_lonlat_polygon_is_reprojected_to_raster_crs(synthetic: Synthetic) -> None:
    red = load_window(synthetic.scene, "red", TOZEUR)
    nir = load_window(synthetic.scene, "nir", TOZEUR)
    index = ndvi(nir.values, red.values)
    palm_zone = box(8.1385, 33.895, 8.1425, 33.909)  # lon/lat, well inside the palm half
    stats = zonal_stats(index, red.transform, red.crs, palm_zone)
    assert stats.mean == pytest.approx(0.75, abs=1e-4)
    assert stats.valid_fraction == pytest.approx(1.0)


@pytest.mark.req("R1")
def test_non_2d_input_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        zonal_stats(np.ma.masked_array(np.zeros(4)), TRANSFORM, CRS)
