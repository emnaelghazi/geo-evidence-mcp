"""R4: the provenance block, its hash and its published JSON Schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from geo_evidence_mcp.export_schema import SCHEMA_PATH, render
from geo_evidence_mcp.provenance import ProvenanceBlock, compute_args_hash
from tests.helpers import make_block


@pytest.mark.req("R4")
def test_args_hash_ignores_key_order_and_float_noise() -> None:
    a = compute_args_hash("band_math", {"bbox": [8.137, 33.893, 8.159, 33.911], "index": "ndvi"})
    b = compute_args_hash(
        "band_math", {"index": "ndvi", "bbox": [8.137, 33.893, 8.1590000001, 33.911]}
    )
    assert a == b and a.startswith("sha256:") and len(a) == 71


@pytest.mark.req("R4")
def test_args_hash_changes_with_tool_or_arguments() -> None:
    base = compute_args_hash("band_math", {"index": "ndvi"})
    assert compute_args_hash("band_math", {"index": "ndwi"}) != base
    assert compute_args_hash("zonal_stats", {"index": "ndvi"}) != base


@pytest.mark.req("R4")
def test_block_requires_outputs_and_forbids_unknown_fields() -> None:
    block = make_block([("ndvi.mean", 0.7, "index")])
    data = block.model_dump(mode="json")
    ProvenanceBlock.model_validate(data)  # round trip
    with pytest.raises(ValidationError):
        ProvenanceBlock.model_validate({**data, "outputs": []})
    with pytest.raises(ValidationError):
        ProvenanceBlock.model_validate({**data, "surprise": 1})
    with pytest.raises(ValidationError):
        ProvenanceBlock.model_validate({**data, "args_hash": "md5:abc"})


@pytest.mark.req("R4")
def test_short_id_is_the_first_eight_hex_digits_of_the_block_id() -> None:
    block = make_block([("ndvi.mean", 0.7, "index")])
    assert block.short_id == block.block_id.hex[:8] and len(block.short_id) == 8


@pytest.mark.req("R16")
def test_committed_json_schema_matches_the_model() -> None:
    assert SCHEMA_PATH.exists(), "run `uv run gem-export-schema`"
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert {"block_id", "args_hash", "sources", "spatial", "outputs", "tool_trace"} <= set(
        schema["required"] + list(schema["properties"])
    )
