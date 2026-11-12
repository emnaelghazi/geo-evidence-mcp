"""Live M1 check over the Tozeur AOI: find a clear scene, compute NDVI, print what to verify by hand.

Run on a machine with internet access:  uv run gem-m1-check --start 2026-08-01 --end 2026-11-10
Writes m1_report.json. Numbers come from a real run only; nothing here is precomputed.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from pystac_client.exceptions import APIError
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.indices import ndvi
from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.raster import WindowData, load_window
from geo_evidence_mcp.stac import search_stac
from geo_evidence_mcp.zonal import zonal_stats

HIGH_NDVI = 0.4  # clearly vegetated
LOW_NDVI = 0.15  # bare ground or sparse cover
MIN_CLASS_FRACTION = 0.05


def _pixel_samples(
    index: np.ma.MaskedArray, red: WindowData, nir: WindowData
) -> list[dict[str, Any]]:
    data = np.ma.filled(index, np.nan)
    valid = np.argwhere(~np.isnan(data))
    order = valid[np.argsort(data[~np.isnan(data)])]
    picks = {
        "lowest NDVI": order[0],
        "median NDVI": order[len(order) // 2],
        "highest NDVI": order[-1],
    }
    samples: list[dict[str, Any]] = []
    for label, (row, col) in picks.items():
        x, y = xy(red.transform, int(row), int(col), offset="center")
        lon, lat = warp_transform(red.crs, "EPSG:4326", [x], [y])
        samples.append(
            {
                "label": label,
                "lon": round(lon[0], 6),
                "lat": round(lat[0], 6),
                "red_dn": round((float(red.values[row, col]) - red.offset) / red.scale),
                "nir_dn": round((float(nir.values[row, col]) - nir.offset) / nir.scale),
                "ndvi": round(float(data[row, col]), 4),
            }
        )
    return samples


def _ascii_map(index: np.ma.MaskedArray, cols: int = 56, rows: int = 22) -> str:
    """Coarse NDVI map (north at top) to judge, without any GIS tool, what the box contains."""
    ramp = " .:-=+*#%@"  # NDVI <= -0.1 ... >= 0.9
    data = np.ma.filled(index, np.nan)
    lines = []
    for r in np.array_split(np.arange(data.shape[0]), rows):
        line = ""
        for c in np.array_split(np.arange(data.shape[1]), cols):
            block = data[np.ix_(r, c)]
            if not np.isfinite(block).any():
                line += "?"
                continue
            level = (float(np.nanmean(block)) + 0.1) / 1.0
            line += ramp[max(0, min(len(ramp) - 1, int(level * len(ramp))))]
        lines.append(line)
    return "\n".join(lines)


def _gdal_command(href: str, lon: float, lat: float) -> str:
    if href.startswith("s3://"):
        return (
            f"gdallocationinfo --config AWS_NO_SIGN_REQUEST YES -valonly -wgs84 "
            f'"/vsis3/{href[5:]}" {lon} {lat}'
        )
    return f'gdallocationinfo -valonly -wgs84 "/vsicurl/{href}" {lon} {lat}'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--start", type=date.fromisoformat, default=date.today() - timedelta(days=60)
    )
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--report", type=Path, default=Path("m1_report.json"))
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="override the Tozeur AOI (lon/lat degrees)",
    )
    args = parser.parse_args(argv)
    aoi = (
        BBox(west=args.bbox[0], south=args.bbox[1], east=args.bbox[2], north=args.bbox[3])
        if args.bbox
        else TOZEUR
    )

    try:
        found = search_stac(aoi, args.start, args.end, max_cloud=args.max_cloud)
    except APIError as exc:
        print(f"Could not reach the STAC API ({exc}). This check needs internet access.")
        return 3
    scenes = [s for s in found if {"red", "nir"} <= set(s.assets)]
    if not scenes:
        print(f"No scene with red+nir under {args.max_cloud}% cloud in {args.start}..{args.end}.")
        return 1
    print(f"{len(scenes)} candidate scenes (least cloudy first):")
    for s in scenes[:5]:
        print(
            f"  {s.item_id}  {s.datetime:%Y-%m-%d}  cloud={s.cloud_cover}  baseline={s.processing_baseline}"
        )

    scene = scenes[0]
    red, nir = load_window(scene, "red", aoi), load_window(scene, "nir", aoi)
    if red.transform != nir.transform or red.values.shape != nir.values.shape:
        print("red and nir windows are not on the same grid; refusing to continue.")
        return 2
    index = ndvi(nir.values, red.values)
    stats = zonal_stats(index, red.transform, red.crs)
    negative = np.ma.getdata(red.values) < 0
    negative |= np.ma.getdata(nir.values) < 0
    negative_fraction = (
        float(negative[~np.ma.getmaskarray(index)].mean()) if stats.valid_pixels else 0.0
    )
    valid = index.compressed()
    high = float((valid > HIGH_NDVI).mean()) if valid.size else 0.0
    low = float((valid < LOW_NDVI).mean()) if valid.size else 0.0

    print(f"\nScene: {scene.item_id} ({scene.datetime:%Y-%m-%d}, cloud {scene.cloud_cover}%)")
    print(
        f"Window: {red.values.shape[1]}x{red.values.shape[0]} px @ {red.resolution[0]} m, {red.crs}"
    )
    print(
        f"NDVI mean={stats.mean:.3f} median={stats.median:.3f} min={stats.min:.3f} max={stats.max:.3f}"
    )
    print(
        f"valid pixels: {stats.valid_fraction:.1%}; NDVI>{HIGH_NDVI}: {high:.1%}; NDVI<{LOW_NDVI}: {low:.1%}"
    )
    print(
        f"pixels with negative red/nir reflectance (clipped to 0 for NDVI): {negative_fraction:.2%}"
    )
    print(f"\nNDVI map (north up; ' ' = low ... '@' = high; '?' = no data), AOI {aoi.as_tuple}:")
    print(_ascii_map(index))
    if red.clipped:
        print("WARNING: the AOI extends beyond this tile; the window was clipped.")
    if high < MIN_CLASS_FRACTION or low < MIN_CLASS_FRACTION:
        print(
            "WARNING: the box is almost all one class. Shift it so it contains palms AND bare ground."
        )

    samples = _pixel_samples(index, red, nir)
    print(
        "\nHand-check these pixels with GDAL (independent of this code); DN should match exactly:"
    )
    for sample in samples:
        print(
            f"\n  {sample['label']}: lon={sample['lon']} lat={sample['lat']} "
            f"red_dn={sample['red_dn']} nir_dn={sample['nir_dn']} ndvi={sample['ndvi']}"
        )
        print(f"    red: {_gdal_command(red.href, sample['lon'], sample['lat'])}")
        print(f"    nir: {_gdal_command(nir.href, sample['lon'], sample['lat'])}")
    print(
        "\nThen compute (nir-1000)/10000 etc. by hand, and compare the NDVI map with Copernicus Browser."
    )

    args.report.write_text(
        json.dumps(
            {
                "aoi": aoi.model_dump(),
                "scene": {
                    "item_id": scene.item_id,
                    "datetime": scene.datetime.isoformat(),
                    "cloud_cover": scene.cloud_cover,
                    "processing_baseline": scene.processing_baseline,
                    "red_href": red.href,
                    "nir_href": nir.href,
                    "crs": red.crs,
                },
                "ndvi": stats.model_dump(),
                "negative_reflectance_fraction": negative_fraction,
                "fraction_above_0_4": high,
                "fraction_below_0_15": low,
                "samples": samples,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
