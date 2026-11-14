import asyncio
import geo_evidence_mcp  # noqa: F401  (applies the PROJ fix)
from fastmcp import Client
from geo_evidence_mcp.aoi import TOZEUR
from geo_evidence_mcp.provenance import ProvenanceBlock
from geo_evidence_mcp.server import build_server
from geo_evidence_mcp.validator import validate_claims

BBOX = list(TOZEUR.as_tuple)

async def main():
    async with Client(build_server()) as c:
        found = await c.call_tool("search_stac", {"bbox": BBOX, "start": "2026-08-01", "end": "2026-11-10"})
        ids = [s["item_id"] for s in found.structured_content["scenes"]][:2]
        blocks = []
        for item in ids:
            r = await c.call_tool("band_math", {"item_id": item, "index": "ndvi", "bbox": BBOX})
            blocks.append(ProvenanceBlock.model_validate(r.structured_content["provenance"]))
    means = [next(o.value for o in b.outputs if o.name == "ndvi.mean") for b in blocks]
    delta = round(abs(means[0] - means[1]), 3)
    good = f"Mean NDVI was {means[0]:.2f} and {means[1]:.2f}; the difference is {delta}."
    bad = good + " Vegetation cover is 91.3%."
    for text in (good, bad):
        report = validate_claims(text, blocks)
        print(report.ok, report.unsupported)

asyncio.run(main())