"""Replay a stored provenance block (R5): run the recorded call again and diff the numbers."""

from __future__ import annotations

import math

from geo_evidence_mcp import service
from geo_evidence_mcp.provenance import Output, ProvenanceBlock
from geo_evidence_mcp.results import ReplayMismatch, ReplayReport
from geo_evidence_mcp.stac import StacClientLike

REL_TOL = 1e-6
ABS_TOL = 1e-9


def _keyed(outputs: list[Output]) -> dict[tuple[str | None, str], float]:
    return {(o.subject, o.name): o.value for o in outputs}


def replay_block(block: ProvenanceBlock, *, client: StacClientLike | None = None) -> ReplayReport:
    """Re-execute `block.request` (nothing is persisted) and compare every recorded output.

    A mismatch means the data or the code changed since the original call. Note that a replayed
    `search_stac` legitimately differs once newer scenes exist.
    """
    fresh = service.execute(block.request.tool, block.request.args, client=client)
    old, new = _keyed(block.outputs), _keyed(fresh.provenance.outputs)
    mismatches = [
        ReplayMismatch(name=name, subject=subject, stored=old[key], replayed=new[key])
        for key in old.keys() & new.keys()
        for subject, name in [key]
        if not math.isclose(old[key], new[key], rel_tol=REL_TOL, abs_tol=ABS_TOL)
    ]
    mismatches.sort(key=lambda m: (m.name, m.subject or ""))
    missing = sorted(f"{s or ''}{'/' if s else ''}{n}" for s, n in old.keys() - new.keys())
    extra = sorted(f"{s or ''}{'/' if s else ''}{n}" for s, n in new.keys() - old.keys())
    return ReplayReport(
        matches=not (mismatches or missing or extra),
        compared=len(old.keys() & new.keys()),
        mismatches=mismatches,
        missing_in_replay=missing,
        extra_in_replay=extra,
    )
