"""geo-evidence-mcp: EO answers that ship with machine-checkable provenance."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

__version__ = "0.0.1"


def _prefer_bundled_geo_data() -> None:
    """Point PROJ/GDAL at the data bundled inside the rasterio wheel.

    Other software (for example the PostgreSQL/PostGIS installer on Windows) sets a global
    PROJ_LIB / PROJ_DATA / GDAL_DATA that points at an older, incompatible copy, which breaks CRS
    lookups with "proj.db contains DATABASE.LAYOUT.VERSION.MINOR ...". The wheel's own data always
    matches the PROJ/GDAL it was built with, so it wins here. Runs before rasterio is used.
    """
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    base = Path(spec.origin).parent
    proj_data, gdal_data = base / "proj_data", base / "gdal_data"
    if (proj_data / "proj.db").is_file():
        os.environ["PROJ_DATA"] = str(proj_data)
        os.environ.pop("PROJ_LIB", None)
    if gdal_data.is_dir():
        os.environ["GDAL_DATA"] = str(gdal_data)


_prefer_bundled_geo_data()
