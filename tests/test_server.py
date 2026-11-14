"""R1/R2: the MCP surface, exercised through FastMCP's in-process client (no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import rasterio
from fastmcp import Client
from fastmcp.exceptions import ToolError

from geo_evidence_mcp.server import build_server
from geo_evidence_mcp.store import MemoryStore, ProvenanceStore
from tests.conftest import FakeCatalog, Synthetic, scene_to_item

PALM_ZONE = {
    "type": "Polygon",
    "coordinates": [
        [[8.1385, 33.895], [8.1425, 33.895], [8.1425, 33.909], [8.1385, 33.909], [8.1385, 33.895]]
    ],
}
AOI = [8.137, 33.893, 8.159, 33.911]
EXPECTED_TOOLS = {
    "health",
    "search_stac",
    "load_window",
    "band_math",
    "zonal_stats",
    "get_provenance",
}


def call(
    synthetic: Synthetic,
    tool: str,
    args: dict[str, Any],
    store: ProvenanceStore | None = None,
) -> dict[str, Any]:
    server = build_server(
        client=FakeCatalog([scene_to_item(synthetic.scene)]),
        store=store if store is not None else MemoryStore(),
    )

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
    source = out["provenance"]["sources"][0]
    assert (source["band"], source["scale"], source["offset"]) == ("B04", 1e-4, -0.1)


@pytest.mark.req("R1")
def test_band_math_ndvi_over_bbox_lists_both_sources(synthetic: Synthetic) -> None:
    out = call(
        synthetic, "band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI}
    )
    assert out["zone"] == "bbox" and out["formula"] == "(nir - red) / (nir + red)"
    assert [s["band"] for s in out["provenance"]["sources"]] == ["B08", "B04", "SCL"]
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


@pytest.mark.req("R4")
def test_every_numeric_tool_returns_a_complete_provenance_block(synthetic: Synthetic) -> None:
    results = {
        "search_stac": call(
            synthetic, "search_stac", {"bbox": AOI, "start": "2026-09-01", "end": "2026-09-30"}
        ),
        "load_window": call(
            synthetic,
            "load_window",
            {"item_id": "SYNTH_TOZEUR_L2A", "asset_key": "red", "bbox": AOI},
        ),
        "band_math": call(
            synthetic, "band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI}
        ),
        "zonal_stats": call(
            synthetic,
            "zonal_stats",
            {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "geometry": PALM_ZONE},
        ),
    }
    for tool, out in results.items():
        block = out["provenance"]
        assert block["tool"]["name"] == tool
        assert block["request"]["tool"] == tool
        assert block["args_hash"].startswith("sha256:") and len(block["args_hash"]) == 71
        assert block["spatial"]["aoi_geometry"]["type"] == "Polygon"
        assert block["tool_trace"]["trace_id"] and block["tool_trace"]["duration_ms"] >= 0
        assert block["outputs"], tool
        if tool != "search_stac":
            src = block["sources"][0]
            assert src["item_id"] == "SYNTH_TOZEUR_L2A" and src["processing_level"] == "L2A"
            assert src["processing_baseline"] == "05.11" and src["href"]
            assert block["spatial"]["crs"] == "EPSG:32632"
            expected = "nearest_scl_to_10m" if tool in ("band_math", "zonal_stats") else "none"
            assert block["computation"]["resampling"] == expected
    index = results["band_math"]["provenance"]["computation"]
    assert (
        index["parameters"]["clip_min"] == 0.0 and index["formula"] == "(nir - red) / (nir + red)"
    )


@pytest.mark.req("R5")
def test_blocks_are_stored_and_retrievable_by_id_and_trace(synthetic: Synthetic) -> None:
    store = MemoryStore()
    out = call(
        synthetic,
        "band_math",
        {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI, "run_id": "run-42"},
        store=store,
    )
    block = out["provenance"]
    assert block["tool_trace"]["run_id"] == "run-42"
    stored = store.by_trace(block["tool_trace"]["trace_id"])
    assert [str(b.block_id) for b in stored] == [block["block_id"]]
    fetched = call(synthetic, "get_provenance", {"block_id": block["block_id"]}, store=store)
    assert fetched["block"]["args_hash"] == block["args_hash"] and fetched["replay"] is None


@pytest.mark.req("R5")
def test_replay_matches_when_data_is_unchanged(synthetic: Synthetic) -> None:
    store = MemoryStore()
    out = call(
        synthetic,
        "band_math",
        {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI},
        store=store,
    )
    report = call(
        synthetic,
        "get_provenance",
        {"block_id": out["provenance"]["block_id"], "replay": True},
        store=store,
    )["replay"]
    assert report["matches"] is True and report["compared"] > 10 and not report["mismatches"]
    assert (
        len(store.by_trace(out["provenance"]["tool_trace"]["trace_id"])) == 1
    )  # replay stores nothing new


@pytest.mark.req("R5")
def test_replay_detects_changed_source_data(synthetic: Synthetic) -> None:
    store = MemoryStore()
    out = call(
        synthetic,
        "band_math",
        {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI},
        store=store,
    )
    path = synthetic.scene.assets["red"].href
    with rasterio.open(path, "r+") as dataset:
        data = dataset.read(1)
        data[data == 1500] = 1700  # the "palm" red band changes after the answer was given
        dataset.write(data, 1)
    report = call(
        synthetic,
        "get_provenance",
        {"block_id": out["provenance"]["block_id"], "replay": True},
        store=store,
    )["replay"]
    assert report["matches"] is False
    assert "ndvi.max" in {m["name"] for m in report["mismatches"]}


@pytest.mark.req("R5")
def test_unknown_block_id_is_a_readable_error(synthetic: Synthetic) -> None:
    with pytest.raises(ToolError, match="no provenance block"):
        call(synthetic, "get_provenance", {"block_id": "12345678-1234-5678-1234-567812345678"})
    with pytest.raises(ToolError, match="badly formed"):
        call(synthetic, "get_provenance", {"block_id": "not-a-uuid"})


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


def _cloudy_world(tmp_path: Path) -> tuple[Synthetic, int]:
    """A scene whose desert side holds a block of bright cloud: 5 x 8 SCL cells = 160 pixels."""
    from geo_evidence_mcp.aoi import TOZEUR
    from geo_evidence_mcp.raster import load_window
    from tests.conftest import make_scene

    probe = make_scene(tmp_path / "probe")
    col, row, _, _ = load_window(probe.scene, "red", TOZEUR).window
    cells = (slice(row // 2 + 20, row // 2 + 25), slice(col // 2 + 60, col // 2 + 68))
    cloudy = make_scene(tmp_path / "cloudy", cloud_cells=cells, cloud_dn=9000)
    return cloudy, 5 * 8 * 4


@pytest.mark.req("R3")
def test_cloud_mask_removes_cloudy_pixels_and_reports_how_many(tmp_path: Path) -> None:
    synthetic, cloudy_pixels = _cloudy_world(tmp_path)
    args = {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI}
    masked = call(synthetic, "band_math", args)
    raw = call(synthetic, "band_math", {**args, "cloud_mask": "none"})

    total, hole = raw["stats"]["total_pixels"], 25
    assert raw["stats"]["valid_pixels"] == total - hole
    assert masked["stats"]["valid_pixels"] == total - hole - cloudy_pixels
    assert masked["cloud_masked_fraction"] == pytest.approx(cloudy_pixels / (total - hole))
    assert masked["cloud_mask"] == "scl" and raw["cloud_mask"] == "none"
    # Bright cloud (red = nir) has NDVI 0: it drags the minimum down unless it is masked out.
    assert raw["stats"]["min"] == pytest.approx(0.0, abs=1e-6)
    assert masked["stats"]["min"] == pytest.approx(0.05 / 0.55, abs=1e-4)
    # The mask is part of the recorded call and of the evidence.
    params = masked["provenance"]["computation"]["parameters"]
    assert params["cloud_mask"] == "scl" and params["scl_masked_classes"] == [1, 3, 8, 9, 10]
    assert masked["provenance"]["computation"]["resampling"] == "nearest_scl_to_10m"
    assert masked["provenance"]["request"]["args"]["cloud_mask"] == "scl"
    names = {o["name"] for o in masked["provenance"]["outputs"]}
    assert "cloud_masked_fraction" in names and "cloud_masked_fraction" not in {
        o["name"] for o in raw["provenance"]["outputs"]
    }


@pytest.mark.req("R5")
def test_replay_reproduces_a_masked_calculation(tmp_path: Path) -> None:
    synthetic, _ = _cloudy_world(tmp_path)
    store = MemoryStore()
    out = call(
        synthetic,
        "band_math",
        {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI},
        store=store,
    )
    report = call(
        synthetic,
        "get_provenance",
        {"block_id": out["provenance"]["block_id"], "replay": True},
        store=store,
    )["replay"]
    assert report["matches"] is True


@pytest.mark.req("R5")
def test_blocks_recorded_before_masking_existed_replay_without_a_mask(synthetic: Synthetic) -> None:
    from geo_evidence_mcp import service

    legacy_args = {
        "item_id": "SYNTH_TOZEUR_L2A",
        "index": "ndvi",
        "bbox": AOI,
        "collection": "sentinel-2-l2a",
    }
    result = service.execute(
        "band_math", legacy_args, client=FakeCatalog([scene_to_item(synthetic.scene)])
    )
    assert result.provenance.computation.parameters["cloud_mask"] == "none"


@pytest.mark.req("R3")
def test_scene_without_scl_gives_a_readable_error(synthetic: Synthetic) -> None:
    item = scene_to_item(synthetic.scene)
    del item.assets["scl"]
    server = build_server(client=FakeCatalog([item]), store=MemoryStore())

    async def run() -> None:
        async with Client(server) as client:
            await client.call_tool(
                "band_math", {"item_id": "SYNTH_TOZEUR_L2A", "index": "ndvi", "bbox": AOI}
            )

    with pytest.raises(ToolError, match="no SCL asset"):
        asyncio.run(run())
