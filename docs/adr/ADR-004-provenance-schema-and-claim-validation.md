# ADR-004: Provenance schema, storage and claim validation

- Status: Accepted
- Date: 2026-11-14

## Context

R4-R6: every numeric claim needs machine-checkable provenance, the trace must be stored and replayable,
and an answer may not state numbers that no tool produced.

## Decisions

**Schema.** `ProvenanceBlock` (Pydantic, `extra="forbid"`) is the source of truth;
`docs/provenance-schema-1.0.json` is generated from it (`uv run gem-export-schema`) and a test fails
on drift. Fields: `block_id`, `generated_at`, `tool`, `request` (the call as made), `args_hash`,
`sources` (STAC API, collection, item, asset, band, datetime, processing level and baseline,
cloud cover, scale, offset, nodata), `spatial` (CRS, grid bounds, the requested AOI as GeoJSON in
EPSG:4326, resolution, size, clipped), `computation` (operation, formula, parameters, resampling,
nodata policy), `outputs` (named, typed, unit-tagged numbers) and `tool_trace` (trace id, span id,
optional agent run id, start, duration).

**Hash.** `args_hash` is SHA-256 over canonical JSON of `{tool, args}` (sorted keys, floats rounded
to 6 decimals), so key order and float noise do not change it.

**Outputs are the unit of auditing.** Every number a tool can report (statistics, pixel counts,
fractions, grid facts, scene cloud cover, search counts) is an `Output` with a unit
(`index`, `reflectance`, `class`, `fraction`, `count`, `percent`, `m`). The validator only trusts these.

**Storage.** PostgreSQL 16 + PostGIS: one row per block, the full block as `jsonb`, the AOI as
`geometry(Geometry, 4326)` with a GiST index, plus indexes on `trace_id` and `args_hash`. An in-memory
store backs tests and quick local use (`STORE_BACKEND=memory|postgres`).

**Replay.** `get_provenance(block_id, replay=True)` re-executes the recorded request through the same
service code (without storing anything) and diffs every output within rel 1e-6. A mismatch means the
data or the code changed since the answer. A replayed `search_stac` legitimately differs once newer
scenes exist.

**Validator (R6).** `validate_claims(text, blocks)` extracts numbers from an answer and requires each
to equal, at the precision it is written, a recorded output or the difference of two outputs of the
same quantity from different blocks (for change claims). `%` requires a fraction or percent quantity.
Dates, item ids, EPSG codes, band names, baseline numbers and bare years are not treated as claims.

## Limits (stated, not hidden)

- It checks that a number was produced by a tool, not that the sentence around it is true. "NDVI rose
  by 0.3" passes if the difference of two blocks is 0.3, even if the blocks were for different places.
- Rounding tolerance is half a unit in the last written digit, so "0.7" accepts 0.65 to 0.75.
- Counts written as small integers ("2 bands") are flagged unless `ignore_integers_up_to` is set.
- Replay proves reproducibility against today's data, not that the original data was right.
- Negative-reflectance pixels are clipped to 0 before ratios (ADR-003 / M1 finding); the rule is
  recorded in `computation.parameters.clip_min`, and its effect in `negative_reflectance_fraction`.
