-- Provenance blocks (R4, R5). One row per tool call that produced numbers.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS provenance_blocks (
    block_id    uuid PRIMARY KEY,
    trace_id    text        NOT NULL,
    run_id      text,
    tool        text        NOT NULL,
    args_hash   text        NOT NULL,
    created_at  timestamptz NOT NULL,
    aoi         geometry(Geometry, 4326) NOT NULL,
    block       jsonb       NOT NULL
);

CREATE INDEX IF NOT EXISTS provenance_blocks_trace_idx ON provenance_blocks (trace_id);
CREATE INDEX IF NOT EXISTS provenance_blocks_args_idx  ON provenance_blocks (args_hash);
CREATE INDEX IF NOT EXISTS provenance_blocks_aoi_gix   ON provenance_blocks USING GIST (aoi);
