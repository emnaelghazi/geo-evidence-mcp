"""R12: the project must work on machines where other software sets a global PROJ_LIB."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.req("R12")
def test_bundled_proj_wins_over_a_foreign_proj_lib(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign_proj"
    foreign.mkdir()
    (foreign / "proj.db").write_bytes(b"not a real proj database")
    env = {**os.environ, "PROJ_LIB": str(foreign), "PROJ_DATA": str(foreign)}
    code = (
        "import geo_evidence_mcp\n"
        "from rasterio.crs import CRS\n"
        "assert CRS.from_epsg(32632).to_epsg() == 32632\n"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
