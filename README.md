# geo-evidence-mcp

An MCP server and agent for Earth-observation questions where **every numeric answer ships with
machine-checkable provenance**: STAC item, collection, band, CRS, acquisition time, processing level
and a replayable tool-call trace.

Status: **M1 data layer**. The MCP server, provenance persistence and agent are not built yet.

| Milestone | Scope | State |
|---|---|---|
| M0 | Scaffold, compose, CI skeleton, ADR-001 | done (Docker verified in CI only) |
| M1 | Data layer (`search_stac`, `load_window`, `zonal_stats`) | code + synthetic tests done; live Tozeur check PENDING |
| M2 | FastMCP server | pending |
| M3 | Provenance schema + persistence + claim validator | pending |
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
```

Benchmark results: **PENDING** (no real run exists yet).

License: MIT.
