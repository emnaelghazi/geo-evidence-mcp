"""Write (or check) the JSON Schema generated from the provenance model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geo_evidence_mcp.provenance import SCHEMA_VERSION, ProvenanceBlock

SCHEMA_PATH = Path("docs") / f"provenance-schema-{SCHEMA_VERSION}.json"


def render() -> str:
    schema = ProvenanceBlock.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args(argv)
    expected = render()
    if args.check:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        if current != expected:
            print(f"{SCHEMA_PATH} is out of date; run `uv run gem-export-schema`.")
            return 1
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(expected, encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
