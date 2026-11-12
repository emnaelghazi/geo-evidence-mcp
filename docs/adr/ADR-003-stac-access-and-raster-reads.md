# ADR-003: stackstac vs odc-stac, and how M1 reads rasters

- Status: Accepted
- Date: 2026-11-11

## Context

R16 asks for a decision between stackstac and odc-stac. M1 only needs single-scene, single-band
windows over a small AOI, with exact control over what ends up in provenance (asset href, CRS,
pixel window, scale/offset).

## Decision

- M1 reads COG assets directly with rasterio windowed reads (`raster.read_window`). No xarray stack.
- When multi-scene stacks are needed (`change_detect`), use **odc-stac** (pinned when first used),
  not stackstac: stackstac's last PyPI release is 2024-08 (0.5.1), odc-stac's is 2026-07 (0.5.3),
  checked on 2026-11-11.
- Reflectance is computed from the asset's own `raster:bands` scale/offset; if absent, the
  processing-baseline rule applies (offset -0.1 from baseline 04.00). A test shows that ignoring the
  offset changes a hand-computed NDVI from 0.75 to 0.50.

## Consequences

- Each read is self-describing, which keeps M3's provenance block simple.
- Resolution mismatches between bands (for example 20 m SCL vs 10 m red) are not resampled in M1;
  `m1_check` refuses to combine windows on different grids. Cloud masking via SCL is deferred to M5.
- The Earth Search specifics (asset keys, `query` support, `raster:bands`, projection fields) are
  encoded as flagged assumptions and are only confirmed by the opt-in live test.
