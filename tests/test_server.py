"""R1/R2: the MCP surface, exercised through FastMCP's in-process client (no network)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from geo_evidence_mcp.server import build_server
from tests.conftest import FakeCatalog, Synthetic, scene_to_item

PALM_ZONE = {
    "type": "Polygon",
    "coordinates": [
        [[8.1385, 33.895], [8.1425, 33.895], [8.1425, 33.909], [8.1385, 33.909], [8.1385, 33.895]]
    ],
}
AOI = [8.137, 33.893, 8.159, 33.911]
EXPECTED_TOOLS = {"health", "search_stac", "load_window", "band_math", "zonal_stats"}


def call(synthetic: Synthetic, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    server = build_server(client=FakeCatalog([scene_to_item(synthetic.scene)]))

    async def run() -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            assert result.structured_content is not None
            content: dict[str, Any] = result.structured_content
            return content

    return asyncio.run(run())


@pytest.mark.req("R1")
def test_exposes_exactly_the_expected_tools_with_descriptions(synthetic: Synthetic) -> None:
    server = build_server(client=FakeCatalog([]))

    async def run() -> list[Any]:
        async with Client(server) as client:
            listed: list[Any] = list(await client.list_tools())
            return listed

    tools = asyncio.run(run())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.description for t in tools)


@pytest.mark.req("R1")
def test_health(synthetic: Synthetic) -> None:
    out = call(synthetic, "health", {})
    assert out["status"] == "ok" and out["stac_endpoint"].startswith("https://")


@pytest.mark.req("R1")
def test_search_returns_scene_brief(synthetic: Synthetic) -> None:
    out = call(synthetic, "search_stac", {"bbox": AOI, "start": "2026-09-01", "end": "2026-09-30"})
    assert out["count"] == 1
    scene = out["scenes"][0]
    assert scene["item_id"] == "SYNTH_TOZEUR_L2A"
    assert {"red", "nir"} <= set(scene["assets"])


@pytest.mark.req("R1")
def test_load_window_reports_grid_and_masked_pixels(synthetic: Synthetic) -> None:
    out = call(
        synthetic,
        "load_window",
        {"item_id": "SYNTH_TOZEUR_L2A", "asset_key": "red", "bbox": AOI},
    )
    grid = out["grid"]
    assert grid["crs"] == "EPSG:32632" and grid["resolution"] == 10.0 and not grid["clipped"]
    assert out["stats"]["valid_pixels"] == out["stats"]["total_pixels"] - 25  # the nodata hole
    source = out["sources"][0]
    assert (source["band"], source["scale"], source["offset"]) == ("B04", 1e-4, -0.1)


@pytest.mark.req("R1")
def test_band_math_ndvi_over_bbox_lists_both_sources(synthetic: Synthetic) -> None:
    out = call(
        synthetic, "band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI}
    )
    assert out["zone"] == "bbox" and out["formula"] == "(nir - red) / (nir + red)"
    assert [s["band"] for s in out["sources"]] == ["B08", "B04"]
    assert out["stats"]["max"] == pytest.approx(0.75, abs=1e-4)
    assert out["stats"]["min"] == pytest.approx(0.05 / 0.55, abs=1e-4)
    assert out["negative_reflectance_fraction"] == 0.0
    assert out["clip_min"] == 0.0


@pytest.mark.req("R1")
def test_zonal_stats_inside_palm_polygon(synthetic: Synthetic) -> None:
    out = call(
        synthetic,
        "zonal_stats",
        {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "geometry": PALM_ZONE},
    )
    assert out["zone"] == "polygon"
    assert out["stats"]["mean"] == pytest.approx(0.75, abs=1e-4)
    assert out["stats"]["valid_fraction"] == pytest.approx(1.0)


@pytest.mark.req("R1")
@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("band_math", {"item_id": "NOPE", "index": "ndvi", "bbox": AOI}, "NOPE"),
        ("band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "evi", "bbox": AOI}, "ndvi"),
        ("band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": [1, 2, 3]}, "bbox"),
        (
            "band_math",
            {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": [2.0, 36.0, 2.1, 36.1]},
            "does not overlap",
        ),
        (
            "zonal_stats",
            {
                "item_id": "SYNTH_TOZEUR_L2A",
                "index": "ndvi",
                "geometry": {"type": "Point", "coordinates": [8.14, 33.9]},
            },
            "Polygon",
        ),
        (
            "load_window",
            {"item_id": "SYNTH_TOZEUR_L2A", "asset_key": "swir16", "bbox": AOI},
            "available",
        ),
    ],
)
def test_bad_requests_become_readable_tool_errors(
    synthetic: Synthetic, tool: str, args: dict[str, Any], message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        call(synthetic, tool, args)


@pytest.mark.req("R3")
def test_oversized_area_is_refused(synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_evidence_mcp import service

    monkeypatch.setattr(service, "MAX_WINDOW_PIXELS", 1_000)
    with pytest.raises(ToolError, match="above the limit"):
        call(synthetic, "band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI})


@pytest.mark.req("R1")
def test_real_stdio_entrypoint_starts_and_answers() -> None:
    """Spawn `python -m geo_evidence_mcp.server` like an MCP host would and call health."""
    import sys

    from fastmcp.client.transports import StdioTransport

    transport = StdioTransport(command=sys.executable, args=["-m", "geo_evidence_mcp.server"])

    async def run() -> tuple[set[str], dict[str, Any]]:
        async with Client(transport) as client:
            names = {t.name for t in await client.list_tools()}
            health = await client.call_tool("health", {})
            content: dict[str, Any] = health.structured_content or {}
            return names, content

    names, health = asyncio.run(run())
    assert names == EXPECTED_TOOLS
    assert health["status"] == "ok"
