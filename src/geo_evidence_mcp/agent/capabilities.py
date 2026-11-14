"""What this system can and cannot answer, checked in code (R3).

These checks do not depend on the language model's judgment: a question that asks for something
Sentinel-2 L2A cannot give is refused here, before any model call, and a plan that cannot be
served is refused or flagged as low confidence before any data is fetched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.service import MAX_WINDOW_PIXELS

PIXEL_SIZE_M = 10.0
SENSOR_START = date(2015, 6, 23)  # Sentinel-2A launch: nothing earlier exists
MAX_AOI_KM2 = MAX_WINDOW_PIXELS * PIXEL_SIZE_M**2 / 1e6  # 400 km2
MIN_AOI_KM2 = 4 * PIXEL_SIZE_M**2 / 1e6  # four pixels
LOW_CONFIDENCE_AOI_KM2 = 100 * PIXEL_SIZE_M**2 / 1e6  # under a hundred pixels
LONG_WINDOW_DAYS = 366

SUPPORTED = (
    "vegetation greenness (NDVI) and open water (NDWI) from Sentinel-2 optical imagery at 10 m"
)


@dataclass(frozen=True)
class Concern:
    code: str
    severity: Literal["refuse", "warn"]
    message: str


_UNITS_M = {"mm": 0.001, "cm": 0.01, "centimetre": 0.01, "centimeter": 0.01, "m": 1.0}
_SIZE = re.compile(
    r"(?P<n>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|centi?met(?:re|er)s?|met(?:re|er)s?|m)\b", re.I
)
_RESOLUTION_WORDS = re.compile(r"resolution|per pixel|pixel size|\bpixels?\b|\bgsd\b|detail", re.I)
_SUBMETRE = re.compile(
    r"sub[- ]?(?:met(?:re|er)|pixel)|centi?met(?:re|er)[- ]?(?:level|scale|accura|resolution)",
    re.I,
)
_FUTURE = re.compile(
    r"\b(?:forecast\w*|predict\w*|projection\w*|will (?:be|become|have|look|grow|turn)|"
    r"going to be|next (?:year|month|season|decade)|in the future)\b",
    re.I,
)
_YEAR = re.compile(r"\b(20\d{2})\b")
_UNSUPPORTED_QUANTITY = re.compile(
    r"\b(temperature|rain(?:fall)?|precipitation|humidity|wind|soil moisture|crop yield|yield|"
    r"population|pollution|air quality|co2|methane|elevation|altitude|biomass|carbon|"
    r"snow depth|sea level)\b",
    re.I,
)
_COUNTING = re.compile(
    r"\bhow many (?:cars?|vehicles?|trees?|palms?|palm trees?|buildings?|houses?|people|persons?|"
    r"boats?|ships?|animals?|sheep|camels?)\b",
    re.I,
)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def check_question(question: str, today: date) -> list[Concern]:
    """Refuse requests that no Sentinel-2 L2A analysis can serve. Pure text rules, no model."""
    concerns: list[Concern] = []

    for sentence in _sentences(question):
        if _SUBMETRE.search(sentence):
            concerns.append(_too_fine("sub-metre detail"))
            continue
        if not _RESOLUTION_WORDS.search(sentence):
            continue
        for match in _SIZE.finditer(sentence):
            unit = match.group("unit").lower()
            metres = float(match.group("n").replace(",", ".")) * _UNITS_M.get(
                unit if unit in _UNITS_M else ("m" if unit.startswith("met") else "cm"), 1.0
            )
            if metres < PIXEL_SIZE_M:
                concerns.append(_too_fine(match.group(0)))
                break

    years = [int(y) for y in _YEAR.findall(question)]
    if _FUTURE.search(question) or any(y > today.year for y in years):
        concerns.append(
            Concern(
                "future",
                "refuse",
                "I can only report what satellites have already observed, not forecasts or "
                "future conditions.",
            )
        )
    if match_q := _UNSUPPORTED_QUANTITY.search(question):
        concerns.append(
            Concern(
                "unsupported_quantity",
                "refuse",
                f"'{match_q.group(0)}' can't be measured from Sentinel-2 optical imagery. "
                f"I can report {SUPPORTED}.",
            )
        )
    if match_c := _COUNTING.search(question):
        concerns.append(
            Concern(
                "object_counting",
                "refuse",
                f"Counting individual objects ('{match_c.group(0)}') needs far finer imagery than "
                f"Sentinel-2's {PIXEL_SIZE_M:.0f} m pixels.",
            )
        )
    return concerns


def _too_fine(what: str) -> Concern:
    return Concern(
        "resolution_too_fine",
        "refuse",
        f"Sentinel-2 pixels are {PIXEL_SIZE_M:.0f} m wide, so details like '{what}' can't be "
        "resolved.",
    )


def bbox_area_km2(bbox: BBox) -> float:
    mid_lat = math.radians((bbox.south + bbox.north) / 2)
    width = (bbox.east - bbox.west) * 111.32 * math.cos(mid_lat)
    height = (bbox.north - bbox.south) * 110.57
    return abs(width * height)


def check_plan(
    periods: list[tuple[date, date]], bbox: BBox, today: date, *, change: bool
) -> list[Concern]:
    """Check a resolved plan (area and dates) before any data is fetched."""
    concerns: list[Concern] = []
    area = bbox_area_km2(bbox)
    if area > MAX_AOI_KM2:
        concerns.append(
            Concern(
                "area_too_large",
                "refuse",
                f"That area is about {area:,.0f} km2; I handle up to {MAX_AOI_KM2:,.0f} km2 "
                "(roughly 20 x 20 km). Please choose a smaller area.",
            )
        )
    elif area < MIN_AOI_KM2:
        concerns.append(
            Concern(
                "area_too_small",
                "refuse",
                f"That area covers fewer than 4 Sentinel-2 pixels ({PIXEL_SIZE_M:.0f} m each).",
            )
        )
    elif area < LOW_CONFIDENCE_AOI_KM2:
        concerns.append(
            Concern(
                "few_pixels",
                "warn",
                "The area covers fewer than 100 pixels, so the statistics are noisy.",
            )
        )
    for start, end in periods:
        if start > today:
            concerns.append(
                Concern("future_period", "refuse", f"{start} is in the future; no data exists yet.")
            )
        elif end < SENSOR_START:
            concerns.append(
                Concern(
                    "before_sensor",
                    "refuse",
                    f"Sentinel-2 began observing on {SENSOR_START}; {start} to {end} is earlier.",
                )
            )
        elif (end - start).days > LONG_WINDOW_DAYS and not change:
            concerns.append(
                Concern(
                    "long_window",
                    "warn",
                    "The period is longer than a year, but one scene is used; it may not represent "
                    "the whole period.",
                )
            )
    return concerns
