# ADR-006: Cloud masking and deterministic capability checks

- Status: Accepted
- Date: 2026-11-14

## Cloud masking (SCL)

Until M4 a "20% cloud" scene still let cloud, shadow and cirrus pixels into the statistics. Index tools
(`band_math`, `zonal_stats`) now take `cloud_mask` = `scl` (default) or `none`.

- Classes masked: 1 saturated/defective, 3 cloud shadow, 8 and 9 cloud (medium/high probability),
  10 thin cirrus; class 0 (no data) always. Snow (11) and vegetation/water classes are kept.
- The 20 m SCL is resampled to the 10 m grid with nearest neighbour (each 10 m pixel takes the class of
  the 20 m cell containing it). The spectral bands are never resampled. A test checks the result
  pixel for pixel against an independent upsampling, including windows that start mid-cell.
- Provenance records `cloud_mask`, `scl_masked_classes`, `computation.resampling =
  "nearest_scl_to_10m"` (a schema change, regenerated into `docs/provenance-schema-1.0.json`), the SCL
  asset as a source, and the output `cloud_masked_fraction`.
- Backward compatibility: a recorded call without `cloud_mask` (made before M5) replays with
  `cloud_mask="none"`, which is what produced its numbers.

## Capability checks (`agent/capabilities.py`)

R3 must not depend on the model noticing a trap, so two code-level checks run first:

- `check_question` (before any model call): resolution finer than 10 m, sub-metre wording, forecasts
  and future years, quantities optical imagery cannot give (temperature, rainfall, soil moisture,
  yield, population, air quality, elevation, biomass), object counting. 21 trap questions and 10
  look-alike legitimate questions are tested.
- `check_plan` (after the model's plan, before data is fetched): area over 400 km2 or under 4 pixels,
  future periods, periods before Sentinel-2 existed. Warnings (under 100 pixels; a single scene
  standing in for a period over a year) lower confidence instead of refusing.
- Low confidence is explicit in the answer (`confidence = "low"` plus a caveat) when the area is tiny
  or more than half of it was cloud. Between 20% and 50% cloud it is a caveat only. A fully masked
  scene is refused, and its search and analysis blocks stay on record.

## Limits

- The text rules are heuristics. They are deliberately conservative: a trap phrased in an unusual way
  can reach the model, where the plan's `unsupported` option and the validator are the next lines of
  defence. False positives are guarded by a list of legitimate look-alike questions, not proven absent.
- SCL is itself a model output; its errors (for example bright desert flagged as cloud) are not
  corrected. The masked fraction is reported so a reader can see when that matters.
- A tile-wide 400 km2 limit equals the pixel cap used when reading (4,000,000 pixels at 10 m).
