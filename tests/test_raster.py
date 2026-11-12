"""R1 (load_window): windowed reads, scale/offset, masking, clipping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.indices import ndvi
from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.raster import load_window
from tests.conftest import Synthetic, make_scene


@pytest.mark.req("R1")
def test_window_geometry_and_dtype(synthetic: Synthetic) -> None:
    wd = load_window(synthetic.scene, "red", TOZEUR)
    height, width = wd.values.shape
    assert wd.crs == "EPSG:32632"
    assert wd.resolution == (10.0, 10.0)
    assert 195 <= width <= 215 and 195 <= height <= 215  # ~2 km at 10 m
    assert wd.values.dtype == np.float32
    assert not wd.clipped


@pytest.mark.req("R1")
def test_reflectance_scale_and_offset(synthetic: Synthetic) -> None:
    wd = load_window(synthetic.scene, "red", TOZEUR)
    assert float(wd.values[:, :20].mean()) == pytest.approx(0.05, abs=1e-4)  # palm side
    assert float(wd.values[:, -20:].mean()) == pytest.approx(0.25, abs=1e-4)  # desert side


@pytest.mark.req("R1")
def test_nodata_is_masked(synthetic: Synthetic) -> None:
    wd = load_window(synthetic.scene, "nir", TOZEUR)
    assert int(np.ma.getmaskarray(wd.values).sum()) == 25


@pytest.mark.req("R1")
def test_ndvi_end_to_end_matches_hand_values(synthetic: Synthetic) -> None:
    red = load_window(synthetic.scene, "red", TOZEUR)
    nir = load_window(synthetic.scene, "nir", TOZEUR)
    assert red.transform == nir.transform and red.values.shape == nir.values.shape
    index = ndvi(nir.values, red.values)
    assert float(index[:, :20].mean()) == pytest.approx(0.75, abs=1e-4)
    assert float(index[:, -20:].mean()) == pytest.approx(0.05 / 0.55, abs=1e-4)


@pytest.mark.req("R1")
def test_ignoring_the_dn_offset_gives_a_different_ndvi(tmp_path: Path) -> None:
    wrong = make_scene(tmp_path, offset=0.0)
    red = load_window(wrong.scene, "red", TOZEUR)
    nir = load_window(wrong.scene, "nir", TOZEUR)
    index = ndvi(nir.values, red.values)
    assert float(index[:, :20].mean()) == pytest.approx(0.5, abs=1e-4)  # not 0.75


@pytest.mark.req("R1")
def test_bbox_larger_than_asset_is_flagged_clipped(synthetic: Synthetic) -> None:
    big = BBox(west=8.0, south=33.0, east=8.3, north=34.5)
    wd = load_window(synthetic.scene, "red", big)
    assert wd.clipped
    assert wd.values.shape == (450, 450)


@pytest.mark.req("R1")
def test_bbox_outside_asset_raises(synthetic: Synthetic) -> None:
    far = BBox(west=2.0, south=36.0, east=2.1, north=36.1)
    with pytest.raises(ValueError, match="does not overlap"):
        load_window(synthetic.scene, "red", far)


@pytest.mark.req("R1")
def test_unknown_asset_key_lists_available(synthetic: Synthetic) -> None:
    with pytest.raises(KeyError, match="available"):
        load_window(synthetic.scene, "swir16", TOZEUR)
