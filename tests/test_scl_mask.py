"""R3/R1: cloud masking from the Scene Classification Layer (20 m onto the 10 m grid)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.raster import load_window, read_class_mask
from tests.conftest import make_scene

CLOUD_CLASSES = (1, 3, 8, 9, 10)


@pytest.mark.req("R3")
def test_scl_mask_lines_up_with_the_10m_grid_exactly(tmp_path: Path) -> None:
    # A block of cloud in 20 m cells; the 10 m window starts at an arbitrary (maybe odd) pixel.
    cells = (slice(105, 110), slice(100, 108))
    synthetic = make_scene(tmp_path, cloud_cells=cells)
    red = load_window(synthetic.scene, "red", TOZEUR)
    mask = read_class_mask(synthetic.scene.assets["scl"].href, red, CLOUD_CLASSES)

    expected_full = np.kron(synthetic.scl == 9, np.ones((2, 2), dtype=bool))  # 20 m -> 10 m
    col, row, width, height = red.window
    expected = expected_full[row : row + height, col : col + width]
    assert mask.shape == red.values.shape
    assert mask.sum() > 0
    assert np.array_equal(mask, expected)


@pytest.mark.req("R3")
def test_clear_scene_masks_nothing(tmp_path: Path) -> None:
    synthetic = make_scene(tmp_path)
    red = load_window(synthetic.scene, "red", TOZEUR)
    assert not read_class_mask(synthetic.scene.assets["scl"].href, red, CLOUD_CLASSES).any()
