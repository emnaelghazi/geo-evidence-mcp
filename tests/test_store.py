"""R5: provenance storage (memory always; PostgreSQL/PostGIS when TEST_DATABASE_URL is set)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from geo_evidence_mcp.store import MemoryStore, PostgresStore
from tests.helpers import make_block

DSN = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.req("R5")
def test_memory_store_roundtrip_and_trace_lookup() -> None:
    store = MemoryStore()
    block = make_block([("ndvi.mean", 0.7, "index")])
    store.save(block)
    assert store.get(block.block_id) == block
    assert store.get(uuid.uuid4()) is None
    assert store.by_trace(block.tool_trace.trace_id) == [block]
    assert store.by_trace("nope") == []


@pytest.fixture
def pg() -> Iterator[PostgresStore]:
    if not DSN:
        pytest.skip("set TEST_DATABASE_URL to run PostgreSQL tests")
    store = PostgresStore(DSN)
    store.ensure_schema()
    yield store


@pytest.mark.db
@pytest.mark.req("R5")
def test_postgres_roundtrip_is_lossless(pg: PostgresStore) -> None:
    block = make_block([("ndvi.mean", 0.7220123456, "index")], run_id="run-7")
    pg.save(block)
    pg.save(block)  # idempotent
    fetched = pg.get(block.block_id)
    assert fetched == block
    assert [b.block_id for b in pg.by_trace(block.tool_trace.trace_id)] == [block.block_id]
    assert pg.get(uuid.uuid4()) is None


@pytest.mark.db
@pytest.mark.req("R5")
def test_postgres_stores_the_aoi_as_a_queryable_geometry(pg: PostgresStore) -> None:
    assert DSN is not None
    block = make_block([("ndvi.mean", 0.7, "index")])
    pg.save(block)
    with psycopg.connect(DSN) as conn:
        inside, outside, srid = conn.execute(
            """
            SELECT ST_Intersects(aoi, ST_SetSRID(ST_MakePoint(8.148, 33.902), 4326)),
                   ST_Intersects(aoi, ST_SetSRID(ST_MakePoint(2.0, 36.0), 4326)),
                   ST_SRID(aoi)
            FROM provenance_blocks WHERE block_id = %s
            """,
            (block.block_id,),
        ).fetchone() or (None, None, None)
    assert (inside, outside, srid) == (True, False, 4326)


@pytest.mark.db
@pytest.mark.req("R5")
def test_ensure_schema_is_idempotent(pg: PostgresStore) -> None:
    pg.ensure_schema()
    pg.ensure_schema()
