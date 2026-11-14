# geo-evidence-mcp

An MCP server and agent for Earth-observation questions where **every numeric answer ships with
machine-checkable provenance**: STAC item, collection, band, CRS, acquisition time, processing level
and a replayable tool-call trace.

Status: **M3 provenance**. The agent (M4) and the benchmark are not built yet.

| Milestone | Scope | State |
|---|---|---|
| M0 | Scaffold, compose, CI skeleton, ADR-001 | done (Docker verified in CI only) |
| M1 | Data layer (`search_stac`, `load_window`, `zonal_stats`) | done; live Tozeur scene hand-verified (evidence in `docs/evidence/`) |
| M2 | FastMCP server (`health`, `search_stac`, `load_window`, `band_math`, `zonal_stats`) | done; live MCP test passed |
| M3 | Provenance block, PostGIS storage, `get_provenance` + replay, claim validator | code + tests done (PostgreSQL tested here); CI database job PENDING |
| M4 | LangGraph agent + Langfuse tracing | pending |
| M5 | Refusal / low-confidence logic | pending |
| M6 | Benchmark + eval runner | pending |
| M7 | Naive vs provenance comparison | pending |
| M8 | Polish, demo, v0.1.0 | pending |

## Quickstart (M0)

```bash
make install        # uv sync
make init           # creates .env with generated secrets
make check          # ruff, mypy --strict, pytest
make up             # PostGIS + pgvector
make up-obs         # + self-hosted Langfuse (http://localhost:3000)
uv run gem-m1-check # live Tozeur NDVI check (needs internet); see docs/adr/ADR-003
RUN_INTEGRATION=1 uv run pytest -m integration
TEST_DATABASE_URL=postgresql://gem:<password>@localhost:5432/gem uv run pytest -m db   # after `make up`
uv run gem-export-schema   # regenerate docs/provenance-schema-1.0.json
```

## Use as an MCP server (stdio)

```bash
uv run gem-mcp
```

Host configuration (untested with a real host so far):

```json
{ "mcpServers": { "geo-evidence": { "command": "uv", "args": ["--directory", "<path-to-repo>", "run", "gem-mcp"] } } }
```

Benchmark results: **PENDING** (no real run exists yet).

License: MIT.
