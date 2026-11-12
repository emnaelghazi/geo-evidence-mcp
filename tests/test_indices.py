"""R1 (band_math): spectral indices."""

from __future__ import annotations

import numpy as np
import pytest

from geo_evidence_mcp.indices import ndvi, ndwi, normalized_difference


def _ma(values: list[float], mask: list[bool] | None = None) -> np.ma.MaskedArray:
    return np.ma.masked_array(np.array(values, dtype="float32"), mask=mask or [False] * len(values))


@pytest.mark.req("R1")
def test_ndvi_hand_computed() -> None:
    result = ndvi(_ma([0.35, 0.30]), _ma([0.05, 0.25]))
    assert result[0] == pytest.approx(0.75, abs=1e-6)
    assert result[1] == pytest.approx(0.05 / 0.55, abs=1e-6)


@pytest.mark.req("R1")
def test_ndwi_is_green_minus_nir() -> None:
    assert ndwi(_ma([0.3]), _ma([0.1]))[0] == pytest.approx(0.5, abs=1e-6)


@pytest.mark.req("R1")
def test_masked_inputs_stay_masked() -> None:
    result = ndvi(_ma([0.4, 0.4], [True, False]), _ma([0.1, 0.1]))
    mask = np.ma.getmaskarray(result)
    assert bool(mask[0]) and not bool(mask[1])


@pytest.mark.req("R1")
def test_zero_denominator_is_masked_not_nan() -> None:
    result = normalized_difference(_ma([0.0]), _ma([0.0]))
    assert bool(np.ma.getmaskarray(result)[0])
    assert not np.isnan(np.ma.getdata(result)).any()


@pytest.mark.req("R1")
def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        normalized_difference(_ma([1.0, 2.0]), _ma([1.0]))
