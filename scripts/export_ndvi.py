"""Export the full-resolution NDVI of one scene over an area as a .npy array (NaN = masked).

    uv run python scripts/export_ndvi.py            # the Tozeur scene used in the project status card
    uv run python scripts/make_status_visual.py --ndvi docs/evidence/ndvi_tozeur_2026-09-04.npy

Uses the same code path as the MCP tools (SCL cloud mask, negative reflectance clipped to 0).
Needs internet access to Earth Search and the Sentinel-2 COGs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import geo_evidence_mcp  # noqa: F401  (applies the bundled-PROJ fix before rasterio is used)
from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.indices import ndvi
from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.raster import load_window, read_class_mask
from geo_evidence_mcp.service import CLOUD_MASK_CLASSES, MAX_WINDOW_PIXELS
from geo_evidence_mcp.stac import StacClientLike, fetch_scene


def export(
    item_id: str, bbox: BBox, out: Path, client: StacClientLike | None = None
) -> tuple[int, int, float]:
    scene = fetch_scene(item_id, client=client)
    nir = load_window(scene, "nir", bbox, max_pixels=MAX_WINDOW_PIXELS)
    red = load_window(scene, "red", bbox, max_pixels=MAX_WINDOW_PIXELS)
    index = ndvi(nir.values, red.values)
    cloudy = read_class_mask(scene.assets["scl"].href, nir, CLOUD_MASK_CLASSES)
    masked = np.ma.masked_array(np.ma.getdata(index), mask=np.ma.getmaskarray(index) | cloudy)
    array = masked.filled(np.nan).astype("float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    return array.shape[0], array.shape[1], float(np.isnan(array).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", default="S2C_32SMC_20260904_0_L2A")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    parser.add_argument(
        "--out", type=Path, default=Path("docs/evidence/ndvi_tozeur_2026-09-04.npy")
    )
    args = parser.parse_args()
    bbox = (
        BBox(west=args.bbox[0], south=args.bbox[1], east=args.bbox[2], north=args.bbox[3])
        if args.bbox
        else TOZEUR
    )
    height, width, masked = export(args.item, bbox, args.out)
    print(f"wrote {args.out}: {width} x {height} px, {masked:.1%} masked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
