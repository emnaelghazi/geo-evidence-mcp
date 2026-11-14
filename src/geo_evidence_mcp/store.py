"""Provenance storage (R5): in memory for tests, PostgreSQL/PostGIS for real runs."""

from __future__ import annotations

import json
from importlib import resources
from typing import Protocol
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from geo_evidence_mcp.config import Settings
from geo_evidence_mcp.provenance import ProvenanceBlock


class ProvenanceStore(Protocol):
    def save(self, block: ProvenanceBlock) -> None: ...
    def get(self, block_id: UUID) -> ProvenanceBlock | None: ...
    def by_trace(self, trace_id: str) -> list[ProvenanceBlock]: ...


class MemoryStore:
    def __init__(self) -> None:
        self._blocks: dict[UUID, ProvenanceBlock] = {}

    def save(self, block: ProvenanceBlock) -> None:
        self._blocks[block.block_id] = block

    def get(self, block_id: UUID) -> ProvenanceBlock | None:
        return self._blocks.get(block_id)

    def by_trace(self, trace_id: str) -> list[ProvenanceBlock]:
        return [b for b in self._blocks.values() if b.tool_trace.trace_id == trace_id]


class PostgresStore:
    """One connection per call: simple and safe; a pool can replace it if latency matters."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        sql = (resources.files("geo_evidence_mcp") / "sql" / "001_provenance.sql").read_text(
            encoding="utf-8"
        )
        with psycopg.connect(self._dsn) as conn:
            conn.execute(sql)

    def save(self, block: ProvenanceBlock) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO provenance_blocks
                    (block_id, trace_id, run_id, tool, args_hash, created_at, aoi, block)
                VALUES (%s, %s, %s, %s, %s, %s,
                        ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s)
                ON CONFLICT (block_id) DO NOTHING
                """,
                (
                    block.block_id,
                    block.tool_trace.trace_id,
                    block.tool_trace.run_id,
                    block.tool.name,
                    block.args_hash,
                    block.generated_at,
                    json.dumps(block.spatial.aoi_geometry),
                    Jsonb(block.model_dump(mode="json")),
                ),
            )

    def get(self, block_id: UUID) -> ProvenanceBlock | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT block FROM provenance_blocks WHERE block_id = %s", (block_id,)
            ).fetchone()
        return ProvenanceBlock.model_validate(row[0]) if row else None

    def by_trace(self, trace_id: str) -> list[ProvenanceBlock]:
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT block FROM provenance_blocks WHERE trace_id = %s ORDER BY created_at",
                (trace_id,),
            ).fetchall()
        return [ProvenanceBlock.model_validate(r[0]) for r in rows]


def make_store(settings: Settings) -> ProvenanceStore:
    if settings.store_backend == "postgres":
        store = PostgresStore(settings.database_url)
        store.ensure_schema()
        return store
    return MemoryStore()
