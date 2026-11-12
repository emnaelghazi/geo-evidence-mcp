# ADR-002: FastMCP version and server design

- Status: Accepted (resolves the open question in ADR-001)
- Date: 2026-11-12

## Context

The brief specifies FastMCP 2.x. PyPI's 2.x line ends at 2.14.7 (2026-04-13); the latest release is 4.1.8.
M2 needs: tool registration, typed structured results, readable validation errors, and the in-process
`Client` for tests (R2, R12).

## Probe (2026-11-12, isolated venvs, Python 3.12)

| | 2.14.7 | 4.1.8 |
|---|---|---|
| `@mcp.tool`, in-process `Client`, `structured_content` | works | works, identical output |
| Bad-argument error | `ToolError` with validation message | same |
| Resolves with our pins (pydantic 2.14.3, rasterio 1.5.2, ...) | yes | yes |
| Import-time warnings | 2 authlib deprecation warnings | none seen |
| Full project suite (55 tests, mypy --strict) | passes | passes |

## Decision

Pin `fastmcp==2.14.7`, as the brief says. Moving to 4.1.8 is a one-line change in `pyproject.toml`
(then `uv lock`); the same suite passes on both. Revisit at M8 (v0.1.0), when no further API work remains.

## Server design

- Stateless: tools take STAC item ids and areas, never server-side handles, so any answer can be replayed.
- Thin wrappers in `server.py` (validation, error mapping); all logic in `service.py` (plain functions).
- Tools return summaries and statistics, never pixel arrays.
- Expected failures (bad id, AOI outside the scene, area too large, unknown asset) become `ToolError`
  messages the agent can read; unexpected exceptions are not masked.
- Requests above 4,000,000 pixels (about 20 x 20 km at 10 m) are refused before any read (first step of R3).
- `sources` in results mirrors the draft provenance schema; M3 replaces it with the full block.
