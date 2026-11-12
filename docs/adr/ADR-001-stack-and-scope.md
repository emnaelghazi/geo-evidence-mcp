# ADR-001: Stack, scope and open questions

- Status: Accepted; the FastMCP question is resolved in ADR-002
- Date: 2026-11-11

## Context

Phase 1 builds an MCP server and LangGraph agent that answers Earth-observation questions with
machine-checkable provenance, plus a public benchmark comparing a provenance-enforcing agent with
a naive one (see the project brief, requirements R1-R17).

## Decision

- Python 3.12+ (deviation from the brief's 3.11+: `rasterio==1.5.2` requires >=3.12, verified by
  dependency resolution on 2026-11-11), `uv` for environments and lockfile, `ruff` + `mypy --strict` + `pre-commit`.
- All dependencies pinned exactly in `pyproject.toml`, added in the milestone that first uses them.
- PostgreSQL 16 + PostGIS 3.4 + pgvector built from tag `v0.7.4` (one image, `docker/postgis-pgvector`).
- Langfuse self-hosted from a trimmed copy of the upstream compose file (v4 image line).
- Config only through environment variables (`pydantic-settings`); `make init` generates all secrets
  into an untracked `.env`.
- Every test carries `@pytest.mark.req("R<n>")` so CI can report requirement coverage.

## Open question: FastMCP 2.x vs the current major line

The brief specifies FastMCP 2.x. On 2026-11-11 PyPI shows the 2.x line ending at 2.14.7
(2026-04-13) while the latest release is 4.1.8 (2026-11-10). Staying on 2.14.7 follows the brief but
means building on a line that no longer receives releases. Default: pin `fastmcp==2.14.7` at M2.
Decide before M2 whether to move to 4.x; the in-process test client is what we depend on (R2/R12).

## Consequences

- stackstac's last release is 2024-08 while odc-stac is current (0.5.3, 2026-07): lean towards odc-stac,
  to be settled in ADR-003 at M1.
- Langfuse image tag `4` is a moving tag until pinned to an exact version/digest after a first pull.
- Docker behaviour (compose build, extension versions) is verified in CI (`stack-smoke`), not locally here.
