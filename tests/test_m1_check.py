"""R1: the live-check CLI, run offline against the synthetic scene (search_stac patched)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from geo_evidence_mcp import m1_check
from geo_evidence_mcp.models import BBox, SceneSummary
from tests.conftest import Synthetic


@pytest.mark.req("R1")
def test_cli_end_to_end_on_synthetic_scene(
    synthetic: Synthetic,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_search(bbox: BBox, start: date, end: date, **_: object) -> list[SceneSummary]:
        return [synthetic.scene]

    monkeypatch.setattr(m1_check, "search_stac", fake_search)
    report = tmp_path / "report.json"
    assert m1_check.main(["--report", str(report)]) == 0

    out = capsys.readouterr().out
    assert "NDVI map" in out and "gdallocationinfo" in out
    data = json.loads(report.read_text())
    assert data["scene"]["item_id"] == "SYNTH_TOZEUR_L2A"
    assert data["ndvi"]["valid_fraction"] > 0.99
    assert 0.35 < data["fraction_above_0_4"] < 0.65  # about half is palm (NDVI 0.75)
    assert 0.35 < data["fraction_below_0_15"] < 0.65  # about half is desert (NDVI 0.09)
    assert data["negative_reflectance_fraction"] == 0.0
    highest = next(s for s in data["samples"] if s["label"] == "highest NDVI")
    assert (highest["red_dn"], highest["nir_dn"]) == (1500, 4500)
    assert highest["ndvi"] == pytest.approx(0.75, abs=1e-3)


@pytest.mark.req("R1")
def test_cli_bbox_override_is_used(
    synthetic: Synthetic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[BBox] = []

    def fake_search(bbox: BBox, start: date, end: date, **_: object) -> list[SceneSummary]:
        seen.append(bbox)
        return [synthetic.scene]

    monkeypatch.setattr(m1_check, "search_stac", fake_search)
    args = ["--bbox", "8.14", "33.895", "8.15", "33.905", "--report", str(tmp_path / "r.json")]
    assert m1_check.main(args) == 0
    assert seen[0].as_tuple == (8.14, 33.895, 8.15, 33.905)
