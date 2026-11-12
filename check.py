import geo_evidence_mcp  # noqa: F401  (applies the PROJ fix)
import rasterio
from rasterio.warp import transform

base = ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
        "32/S/MC/2026/9/S2C_32SMC_20260904_0_L2A/")
points = {"lowest": (8.146829, 33.907388),
          "median": (8.141006, 33.894811),
          "highest": (8.137598, 33.911021)}
for name, (lon, lat) in points.items():
    values = []
    for band in ("B04", "B08"):
        with rasterio.open(base + band + ".tif") as src:
            x, y = transform("EPSG:4326", src.crs, [lon], [lat])
            values.append(int(next(src.sample([(x[0], y[0])]))[0]))
    print(name, "red/nir DN =", values)