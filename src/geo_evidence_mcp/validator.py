"""Post-hoc claim validator (R6): every number in an answer must trace to a tool output.

Heuristic by design. It cannot judge whether a sentence is *true*, only whether each number it
states was actually produced by a recorded tool call (as a value, or as a difference between two
values of the same quantity). Numbers that are clearly not claims (dates, item ids, EPSG codes,
band names, years) are skipped. See docs/adr/ADR-004 for the limits.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from pydantic import BaseModel

from geo_evidence_mcp.provenance import ProvenanceBlock, Unit

_CLEANERS = [
    re.compile(r"https?://\S+"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\bsha256:[0-9a-f]+"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+Z?)?"),
    re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Za-z0-9]+){2,}\b"),  # STAC item ids
    re.compile(r"\bEPSG:\d+"),
    re.compile(r"\bB(?:0?[1-9]|1[0-2])A?\b"),  # Sentinel-2 band names
    re.compile(r"\bL[12][AC]\b"),
    re.compile(r"\bSentinel-?[12]\b", re.IGNORECASE),
    re.compile(r"\bbaseline\s+\d{2}\.\d{2}\b", re.IGNORECASE),
    re.compile(r"\[[0-9a-f]{8}\]"),  # short-id citations
]
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-+\u2212]?)"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
    r"(?P<pct>\s?%)?"
)
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


class NumberMention(BaseModel):
    text: str
    value: float  # signed, as written (a percent stays in percent units)
    decimals: int
    percent: bool
    start: int
    end: int


class ClaimCheck(BaseModel):
    mention: NumberMention
    status: Literal["supported", "unsupported", "ignored"]
    matched: str | None = None
    reason: str | None = None


class ValidationReport(BaseModel):
    ok: bool
    checks: list[ClaimCheck]

    @property
    def unsupported(self) -> list[NumberMention]:
        return [c.mention for c in self.checks if c.status == "unsupported"]


@dataclass(frozen=True)
class _Candidate:
    value: float
    unit: Unit
    label: str
    derived: bool


def _blank_out(text: str) -> str:
    for pattern in _CLEANERS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def extract_numbers(text: str) -> list[NumberMention]:
    mentions: list[NumberMention] = []
    for match in _NUMBER.finditer(_blank_out(text)):
        raw = match.group("num")
        cleaned = raw.replace(",", "") if _THOUSANDS.fullmatch(raw) else raw.replace(",", ".")
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        value = float(cleaned)
        if match.group("sign") in ("-", "\u2212"):
            value = -value
        mentions.append(
            NumberMention(
                text=match.group(0).strip(),
                value=value,
                decimals=decimals,
                percent=match.group("pct") is not None,
                start=match.start(),
                end=match.end(),
            )
        )
    return mentions


def _candidates(blocks: Iterable[ProvenanceBlock]) -> list[_Candidate]:
    base: list[tuple[str, str | None, str, float, Unit]] = []  # block, subject, name, value, unit
    for block in blocks:
        for out in block.outputs:
            base.append((block.short_id, out.subject, out.name, out.value, out.unit))
    candidates = [
        _Candidate(v, u, f"{n}{f' ({s})' if s else ''} [{b}]", False) for b, s, n, v, u in base
    ]
    for (b1, s1, n1, v1, u1), (b2, s2, n2, v2, u2) in combinations(base, 2):
        if n1 == n2 and u1 == u2 and (b1 != b2 or s1 != s2) and u1 not in ("class",):
            candidates.append(
                _Candidate(v1 - v2, u1, f"difference of {n1} between [{b1}] and [{b2}]", True)
            )
    return candidates


def _supports(mention: NumberMention, candidate: _Candidate) -> bool:
    tolerance = 0.5 * 10 ** (-mention.decimals) + 1e-9
    if mention.percent:
        if candidate.unit == "percent":
            target = candidate.value
        elif candidate.unit == "fraction":
            target = candidate.value * 100
        else:
            return False
    else:
        target = candidate.value
    if candidate.derived:
        return bool(abs(abs(mention.value) - abs(target)) <= tolerance)
    return bool(abs(mention.value - target) <= tolerance)


def validate_claims(
    text: str,
    blocks: Iterable[ProvenanceBlock],
    *,
    ignore_integers_up_to: int = 0,
) -> ValidationReport:
    """Check every number in `text` against the outputs of `blocks`.

    A number is supported when, rounded to the precision it is written with, it equals a recorded
    output, or the difference between two recorded values of the same quantity. A percent sign
    requires a fraction or percent quantity. Bare four-digit years and (optionally) small integers
    are ignored rather than flagged.
    """
    candidates = _candidates(list(blocks))
    checks: list[ClaimCheck] = []
    for mention in extract_numbers(text):
        match = next((c for c in candidates if _supports(mention, c)), None)
        if match is not None:
            checks.append(ClaimCheck(mention=mention, status="supported", matched=match.label))
            continue
        plain_int = mention.decimals == 0 and not mention.percent
        if plain_int and 1900 <= abs(mention.value) <= 2100:
            checks.append(ClaimCheck(mention=mention, status="ignored", reason="looks like a year"))
        elif plain_int and abs(mention.value) <= ignore_integers_up_to:
            checks.append(ClaimCheck(mention=mention, status="ignored", reason="small integer"))
        else:
            checks.append(
                ClaimCheck(
                    mention=mention,
                    status="unsupported",
                    reason="no tool output or difference of outputs equals this number",
                )
            )
    return ValidationReport(ok=all(c.status != "unsupported" for c in checks), checks=checks)
