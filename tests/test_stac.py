"""R1 (search_stac): parsing, filtering, ordering, with a fake STAC client (no network)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pystac
import pytest

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.stac import search_stac, summarize_item

RASTER_BANDS = [{"scale": 0.0001, "offset": -0.1, "nodata": 0}]


def make_item(
    item_id: str,
    day: int,
    cloud: float | None,
    *,
    baseline: str = "05.11",
    with_raster_bands: bool = True,
    extra_props: dict[str, Any] | None = None,
) -> pystac.Item:
    props: dict[str, Any] = {"s2:processing_baseline": baseline}
    if cloud is not None:
        props["eo:cloud_cover"] = cloud
    props.update(extra_props or {})
    item = pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2026, 9, day, 10, 0, tzinfo=UTC),
        properties=props,
        collection="sentinel-2-l2a",
    )
    for key in ("red", "nir", "scl"):
        extra = {"raster:bands": RASTER_BANDS} if with_raster_bands and key != "scl" else {}
        item.add_asset(
            key, pystac.Asset(href=f"https://example.test/{item_id}/{key}.tif", extra_fields=extra)
        )
    return item


class FakeSearch:
    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items

    def items(self) -> Iterator[pystac.Item]:
        return iter(self._items)


class FakeClient:
    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> FakeSearch:
        self.calls.append(kwargs)
        return FakeSearch(self._items)


@pytest.mark.req("R1")
def test_sorted_by_cloud_then_newest() -> None:
    client = FakeClient(
        [
            make_item("a", 1, 10.0),
            make_item("b", 5, 2.0),
            make_item("c", 9, 2.0),
            make_item("d", 3, None),
        ]
    )
    scenes = search_stac(TOZEUR, date(2026, 9, 1), date(2026, 9, 30), client=client)
    assert [s.item_id for s in scenes] == ["c", "b", "a", "d"]


@pytest.mark.req("R1")
def test_cloud_filter_is_applied_client_side_too() -> None:
    client = FakeClient([make_item("ok", 1, 5.0), make_item("cloudy", 2, 60.0)])
    scenes = search_stac(TOZEUR, date(2026, 9, 1), date(2026, 9, 30), max_cloud=20, client=client)
    assert [s.item_id for s in scenes] == ["ok"]


@pytest.mark.req("R1")
def test_request_parameters_sent_to_the_api() -> None:
    client = FakeClient([])
    search_stac(TOZEUR, date(2026, 8, 1), date(2026, 11, 10), max_cloud=15, limit=7, client=client)
    call = client.calls[0]
    assert call["collections"] == ["sentinel-2-l2a"]
    assert call["bbox"] == TOZEUR.as_tuple
    assert call["datetime"] == "2026-08-01/2026-11-10"
    assert call["query"] == {"eo:cloud_cover": {"lt": 15}}
    assert call["max_items"] == 7


@pytest.mark.req("R1")
def test_scale_offset_from_raster_bands() -> None:
    scene = summarize_item(make_item("x", 1, 1.0))
    red = scene.assets["red"]
    assert (red.scale, red.offset, red.nodata) == (0.0001, -0.1, 0.0)
    assert scene.assets["scl"].offset == 0.0  # scl has no raster:bands in this fixture


@pytest.mark.parametrize(
    ("baseline", "expected_offset"),
    [("05.11", -0.1), ("04.00", -0.1), ("03.00", 0.0), (None, 0.0)],
)
@pytest.mark.req("R1")
def test_offset_fallback_follows_processing_baseline(
    baseline: str | None, expected_offset: float
) -> None:
    item = make_item("x", 1, 1.0, with_raster_bands=False, baseline=baseline or "")
    if baseline is None:
        del item.properties["s2:processing_baseline"]
    scene = summarize_item(item)
    assert scene.assets["red"].scale == 1e-4
    assert scene.assets["red"].offset == expected_offset
    assert (scene.assets["scl"].scale, scene.assets["scl"].offset) == (1.0, 0.0)


@pytest.mark.parametrize(
    ("props", "expected"),
    [({"proj:code": "EPSG:32632"}, 32632), ({"proj:epsg": 32633}, 32633), ({}, None)],
)
@pytest.mark.req("R1")
def test_epsg_from_either_projection_extension_version(
    props: dict[str, Any], expected: int | None
) -> None:
    assert summarize_item(make_item("x", 1, 1.0, extra_props=props)).epsg == expected


@pytest.mark.req("R1")
def test_start_after_end_rejected() -> None:
    with pytest.raises(ValueError, match="start"):
        search_stac(TOZEUR, date(2026, 10, 1), date(2026, 9, 1), client=FakeClient([]))
