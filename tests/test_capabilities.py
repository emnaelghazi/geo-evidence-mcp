"""R3: deterministic capability checks. Traps are refused without any model call."""

from __future__ import annotations

from datetime import date

import pytest

from geo_evidence_mcp.agent.capabilities import (
    MAX_AOI_KM2,
    bbox_area_km2,
    check_plan,
    check_question,
)
from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.models import BBox

TODAY = date(2026, 11, 14)

TRAPS = {
    "resolution_too_fine": [
        "Show me NDVI at 1 m resolution for the Tozeur oasis.",
        "What is the NDVI of my garden at 50 cm resolution?",
        "Give me 2 metre per pixel vegetation maps of Tozeur.",
        "I need sub-metre NDVI for the oasis.",
        "Can you get centimetre-level accuracy on water extent?",
        "NDVI with 5 m pixel size please.",
    ],
    "future": [
        "Forecast the NDVI of the Tozeur oasis for next year.",
        "Will the oasis be greener in 2030?",
        "Predict vegetation health there in 2027.",
        "What will the NDVI look like next season?",
    ],
    "unsupported_quantity": [
        "What was the temperature in Tozeur in July 2026?",
        "How much rainfall did the oasis get in 2025?",
        "What is the soil moisture in the oasis?",
        "Estimate the crop yield of the oasis in September 2026.",
        "What is the population of the Tozeur oasis?",
        "How bad was the air quality over the oasis?",
        "What is the elevation of the oasis?",
        "How much biomass is stored in the palms?",
    ],
    "object_counting": [
        "How many palm trees are in the Tozeur oasis?",
        "How many buildings are in the oasis?",
        "How many cars were parked there in September 2026?",
    ],
}

LEGITIMATE = [
    "What was the mean NDVI of the Tozeur oasis in September 2026?",
    "Compare NDVI between September 2025 and September 2026 in the Tozeur oasis.",
    "Is there open water in the oasis (NDWI) in September 2026?",
    "What NDVI do you get at 10 m resolution for Tozeur?",
    "NDVI for a 500 m wide area near 8.14, 33.9 in 2026.",
    "How green was the Tozeur oasis in 2024?",
    "What was the water index in the winter of 2025?",
    "Will you compare the NDVI of September 2025 and September 2026?",
    "Which window of 20 m cells shows the highest NDVI?",
    "Was the oasis greener in 2026 than in 2025?",
]


def codes(question: str) -> set[str]:
    return {c.code for c in check_question(question, TODAY) if c.severity == "refuse"}


@pytest.mark.req("R3")
@pytest.mark.parametrize(
    ("code", "question"), [(code, q) for code, qs in TRAPS.items() for q in qs]
)
def test_every_trap_is_refused_for_the_right_reason(code: str, question: str) -> None:
    assert code in codes(question), question
    assert all(c.message for c in check_question(question, TODAY))


@pytest.mark.req("R3")
@pytest.mark.parametrize("question", LEGITIMATE)
def test_legitimate_questions_are_not_refused(question: str) -> None:
    assert codes(question) == set(), question


@pytest.mark.req("R3")
def test_trap_set_is_at_least_twenty_questions() -> None:
    assert sum(len(v) for v in TRAPS.values()) >= 20


def test_area_estimate_is_close_to_geodesic_truth() -> None:
    assert bbox_area_km2(TOZEUR) == pytest.approx(4.0, rel=0.1)  # about 2 x 2 km
    assert bbox_area_km2(BBox(west=0, south=0, east=1, north=1)) == pytest.approx(12308, rel=0.01)


def plan_codes(
    period: tuple[date, date], bbox: BBox = TOZEUR, change: bool = False
) -> dict[str, str]:
    return {c.code: c.severity for c in check_plan([period], bbox, TODAY, change=change)}


@pytest.mark.req("R3")
def test_plan_checks() -> None:
    sept = (date(2026, 9, 1), date(2026, 9, 30))
    assert plan_codes(sept) == {}
    big = BBox(west=8.0, south=33.0, east=8.5, north=33.5)
    assert plan_codes(sept, big) == {"area_too_large": "refuse"}
    assert MAX_AOI_KM2 == 400
    tiny = BBox(west=8.14, south=33.9, east=8.14001, north=33.90001)
    assert plan_codes(sept, tiny) == {"area_too_small": "refuse"}
    small = BBox(west=8.14, south=33.9, east=8.1405, north=33.9005)
    assert plan_codes(sept, small) == {"few_pixels": "warn"}
    assert plan_codes((date(2027, 1, 1), date(2027, 1, 31))) == {"future_period": "refuse"}
    assert plan_codes((date(2010, 1, 1), date(2010, 12, 31))) == {"before_sensor": "refuse"}
    assert plan_codes((date(2025, 1, 1), date(2026, 6, 1))) == {"long_window": "warn"}
    assert plan_codes((date(2025, 1, 1), date(2026, 6, 1)), change=True) == {}
