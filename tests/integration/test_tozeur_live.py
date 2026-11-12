"""Live check against Earth Search. Skipped unless RUN_INTEGRATION=1 (needs internet).

This is where the API assumptions flagged as TODO(M1 verify, live) get confirmed or broken:
asset keys, `query` support, raster:bands scale/offset, projection fields, anonymous COG reads.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.indices import ndvi
from geo_evidence_mcp.raster import load_window
from geo_evidence_mcp.stac import search_stac
from geo_evidence_mcp.zonal import zonal_stats

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1"),
]


@pytest.mark.req("R1")
def test_live_tozeur_ndvi_is_plausible() -> None:
    scenes = search_stac(TOZEUR, date.today() - timedelta(days=90), date.today(), max_cloud=20)
    assert scenes, "no scenes found: check the endpoint, collection name and date range"
    scene = scenes[0]
    assert {"red", "nir"} <= set(scene.assets), sorted(scene.assets)
    assert scene.processing_baseline, "s2:processing_baseline missing; offset fallback is blind"

    red, nir = load_window(scene, "red", TOZEUR), load_window(scene, "nir", TOZEUR)
    assert red.transform == nir.transform
    assert red.scale == pytest.approx(1e-4)
    assert red.offset in (0.0, -0.1)

    index = ndvi(nir.values, red.values)
    stats = zonal_stats(index, red.transform, red.crs)
    assert stats.valid_fraction > 0.8
    assert stats.min is not None and stats.max is not None
    assert -1.0 <= stats.min <= stats.max <= 1.0, (
        "NDVI out of range: check scale/offset and clipping"
    )

    values = index.compressed()
    high, low = float((values > 0.4).mean()), float((values < 0.15).mean())
    assert high >= 0.05, f"only {high:.1%} vegetated pixels: the AOI box misses the oasis"
    assert low >= 0.05, f"only {low:.1%} bare-ground pixels: shift the box to include desert"
