"""R6: the post-hoc claim validator."""

from __future__ import annotations

import pytest

from geo_evidence_mcp.provenance import ProvenanceBlock, Unit
from geo_evidence_mcp.validator import extract_numbers, validate_claims
from tests.helpers import make_block

NDVI: list[tuple[str, float, Unit]] = [
    ("ndvi.mean", 0.7220, "index"),
    ("ndvi.max", 1.0, "index"),
    ("zone.valid_pixels", 41612, "count"),
    ("negative_reflectance_fraction", 0.384, "fraction"),
    ("cloud_cover", 0.67, "percent"),
]


def ok(text: str, *blocks: ProvenanceBlock) -> bool:
    return validate_claims(text, blocks or [make_block(NDVI)]).ok


@pytest.mark.req("R6")
def test_numbers_that_match_a_tool_output_are_supported() -> None:
    report = validate_claims("Mean NDVI was 0.72 over 41,612 valid pixels.", [make_block(NDVI)])
    assert report.ok
    assert [c.status for c in report.checks] == ["supported", "supported"]
    assert report.checks[0].matched is not None and "ndvi.mean" in report.checks[0].matched


@pytest.mark.req("R6")
def test_a_fabricated_number_is_flagged() -> None:
    report = validate_claims("Mean NDVI was 0.65.", [make_block(NDVI)])
    assert not report.ok
    assert [m.text for m in report.unsupported] == ["0.65"]


@pytest.mark.req("R6")
def test_rounding_is_respected_in_both_directions() -> None:
    block = make_block([("ndvi.mean", 0.74, "index")])
    assert ok("NDVI about 0.7", block)  # 0.74 rounds to 0.7
    assert not ok("NDVI about 0.8", block)  # it does not round to 0.8
    assert not ok("NDVI about 0.75", block)  # more precise than the data allows


@pytest.mark.req("R6")
def test_percent_needs_a_fraction_or_percent_quantity() -> None:
    assert ok("38.4% of pixels had negative reflectance")
    assert ok("cloud cover was 0.67%")
    assert not ok("NDVI was 72.2%")  # an index is not a percentage
    assert not ok("38.4 of pixels were negative")  # bare 38.4 is not the fraction 0.384


@pytest.mark.req("R6")
def test_change_claims_are_checked_against_the_difference_of_two_blocks() -> None:
    before = make_block([("ndvi.mean", 0.80, "index")])
    after = make_block([("ndvi.mean", 0.50, "index")])
    assert ok("NDVI dropped by 0.3 between the two dates", before, after)
    assert ok("NDVI changed by -0.30", before, after)
    assert not ok("NDVI dropped by 0.4", before, after)
    assert not ok("NDVI dropped by 0.3", before)  # one block alone cannot support a change


@pytest.mark.req("R6")
def test_identifiers_dates_and_band_names_are_not_treated_as_claims() -> None:
    text = (
        "Scene S2C_32SMC_20260904_0_L2A (2026-09-04, EPSG:32632, bands B04 and B08, "
        "baseline 05.12, [3f9a1c2e]) gave a mean NDVI of 0.72 in 2026."
    )
    report = validate_claims(text, [make_block(NDVI)])
    assert report.ok, [c.mention.text for c in report.checks if c.status == "unsupported"]
    assert any(c.status == "ignored" and c.mention.text == "2026" for c in report.checks)


@pytest.mark.req("R6")
def test_number_formats() -> None:
    mentions = {m.text: m for m in extract_numbers("1,234 and 0,72 and 3.5 and \u22120.02 and 2-4")}
    assert mentions["1,234"].value == 1234 and mentions["0,72"].value == 0.72
    assert mentions["\u22120.02"].value == -0.02 and mentions["0,72"].decimals == 2
    assert {"2", "4"} <= set(mentions)  # a range is two positive numbers, not "2" and "-4"


@pytest.mark.req("R6")
def test_small_integers_are_flagged_unless_explicitly_allowed() -> None:
    block = make_block(NDVI)
    assert not validate_claims("I used 2 bands.", [block]).ok
    assert validate_claims("I used 2 bands.", [block], ignore_integers_up_to=10).ok


@pytest.mark.req("R6")
def test_text_without_numbers_is_ok() -> None:
    assert validate_claims("The oasis looks healthy.", [make_block(NDVI)]).ok
