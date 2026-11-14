"""The LangGraph state machine (R2): intake -> plan -> select data -> analyze -> check evidence
-> narrate -> validate -> finalize. The model only plans and narrates; data selection, analysis
and verification are deterministic code calling the MCP tools. A number the validator cannot
trace to a tool output never reaches the user (R6)."""

from __future__ import annotations

import json
import uuid
from datetime import date
from typing import Any, Literal, TypedDict

from fastmcp import Client, FastMCP
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from geo_evidence_mcp import service
from geo_evidence_mcp.agent.llm import AgentModel, PlanError
from geo_evidence_mcp.agent.plan import Plan
from geo_evidence_mcp.agent.tracing import NullTracer, Tracer
from geo_evidence_mcp.aoi import GAZETTEER, resolve_place
from geo_evidence_mcp.models import BBox
from geo_evidence_mcp.provenance import ProvenanceBlock, compute_args_hash
from geo_evidence_mcp.store import ProvenanceStore
from geo_evidence_mcp.validator import validate_claims

CLOUD_STEPS = (20.0, 60.0)  # try clear scenes first, then accept cloudier ones with a caveat
MAX_NARRATIONS = 3
MAX_QUESTION_CHARS = 2000
SEASON_GAP_DAYS = 30


class EvidenceLine(BaseModel):
    cite: str
    tool: str
    item_id: str | None
    date: str | None
    cloud_cover_pct: float | None


class AgentAnswer(BaseModel):
    status: Literal["answered", "answered_from_template", "refused"]
    question: str
    answer: str
    caveats: list[str]
    evidence: list[EvidenceLine]
    block_ids: list[str]
    run_id: str
    plan: Plan | None
    validation_ok: bool
    unsupported_numbers: list[str]
    model: str

    def render(self) -> str:
        lines = [self.answer]
        if self.caveats:
            lines += ["", "Caveats:"] + [f"  - {c}" for c in self.caveats]
        if self.evidence:
            lines += ["", "Evidence:"]
            for e in self.evidence:
                scene = f" {e.item_id} ({e.date}, cloud {e.cloud_cover_pct}%)" if e.item_id else ""
                lines.append(f"  [{e.cite}] {e.tool}{scene}")
        lines += ["", f"status={self.status} run_id={self.run_id}"]
        return "\n".join(lines)


class State(TypedDict, total=False):
    question: str
    plan: Plan
    bbox: list[float]
    scenes: list[dict[str, Any]]  # chosen scene per period
    blocks: list[ProvenanceBlock]
    caveats: list[str]
    refusal: str
    draft: str
    attempts: int
    feedback: str | None
    unsupported: list[str]
    template_used: bool
    final: AgentAnswer


def _doy_gap(a: date, b: date) -> int:
    gap = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(gap, 365 - gap)


def _analysis_blocks(blocks: list[ProvenanceBlock]) -> list[ProvenanceBlock]:
    return [b for b in blocks if b.tool.name in ("band_math", "zonal_stats")]


def _output(block: ProvenanceBlock, name: str) -> float | None:
    return next((o.value for o in block.outputs if o.name == name), None)


def evidence_json(blocks: list[ProvenanceBlock]) -> str:
    items = []
    for block in _analysis_blocks(blocks):
        source = block.sources[0]
        items.append(
            {
                "cite": block.short_id,
                "tool": block.tool.name,
                "scene": source.item_id,
                "date": source.datetime.date().isoformat(),
                "cloud_cover_pct": source.cloud_cover_pct,
                "outputs": {o.name: round(o.value, 4) for o in block.outputs if o.subject is None},
            }
        )
    return json.dumps(items, indent=1)


def template_answer(plan: Plan, blocks: list[ProvenanceBlock]) -> str:
    """Deterministic fallback: built only from tool outputs, so it always validates."""
    analysis = _analysis_blocks(blocks)
    index = (plan.index or "index").upper()
    lines = []
    for block in analysis:
        mean, low, high = (_output(block, f"{plan.index}.{k}") for k in ("mean", "min", "max"))
        when = block.sources[0].datetime.date().isoformat()
        lines.append(
            f"{index} over the area on {when}: mean {mean:.2f} (min {low:.2f}, max {high:.2f}) "
            f"[{block.short_id}]."
        )
    if len(analysis) == 2:
        first, second = (_output(b, f"{plan.index}.mean") for b in analysis)
        assert first is not None and second is not None
        lines.append(
            f"Change in mean {index}: {second - first:+.2f} "
            f"[{analysis[0].short_id}] [{analysis[1].short_id}]."
        )
    return " ".join(lines)


class _Run:
    def __init__(
        self,
        *,
        client: Client[Any],
        model: AgentModel,
        store: ProvenanceStore,
        tracer: Tracer,
        today: date,
        run_id: str,
    ) -> None:
        self.client, self.model, self.store = client, model, store
        self.tracer, self.today, self.run_id = tracer, today, run_id

    # ---- helpers -------------------------------------------------------------------------
    async def _tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        with self.tracer.span(
            name, kind="tool", input=args, metadata={"run_id": self.run_id}
        ) as sp:
            result = await self.client.call_tool(name, {**args, "run_id": self.run_id})
            data: dict[str, Any] = result.structured_content or {}
            block = data.get("provenance", {})
            sp.set_output({"block_id": block.get("block_id"), "args_hash": block.get("args_hash")})
            return data

    @staticmethod
    def _refuse(message: str) -> dict[str, Any]:
        return {"refusal": message}

    # ---- nodes ---------------------------------------------------------------------------
    async def intake(self, state: State) -> dict[str, Any]:
        with self.tracer.span("intake", input={"question": state["question"]}) as sp:
            question = state["question"].strip()
            if not question or len(question) > MAX_QUESTION_CHARS:
                reason = f"Please ask a question of up to {MAX_QUESTION_CHARS} characters."
                sp.set_output({"refusal": reason})
                return self._refuse(reason)
            sp.set_output({"ok": True})
            return {"question": question}

    async def plan(self, state: State) -> dict[str, Any]:
        places = sorted(GAZETTEER)
        with self.tracer.span(
            "plan",
            kind="generation",
            input={"question": state["question"], "known_places": places},
            metadata={"model": self.model.name},
        ) as sp:
            try:
                plan = await self.model.plan(
                    state["question"], today=self.today, known_places=places
                )
            except (PlanError, ValidationError) as exc:
                sp.set_output({"error": str(exc)})
                return self._refuse(f"I could not turn that into a data request ({exc}).")
            sp.set_output(plan.model_dump(mode="json"))
        if plan.task == "unsupported":
            return {
                "plan": plan,
                **self._refuse(
                    f"I can't answer this from Sentinel-2 L2A data: {plan.unsupported_reason}"
                ),
            }
        return {"plan": plan}

    async def select_data(self, state: State) -> dict[str, Any]:
        plan = state["plan"]
        with self.tracer.span(
            "select_data", kind="chain", input=plan.model_dump(mode="json")
        ) as sp:
            box: BBox
            if plan.bbox is not None:
                try:
                    box = BBox(
                        west=plan.bbox[0], south=plan.bbox[1], east=plan.bbox[2], north=plan.bbox[3]
                    )
                except ValueError as exc:
                    return self._refuse(f"The bounding box is not valid ({exc}).")
            else:
                named = resolve_place(plan.place or "")
                if named is None:
                    known = ", ".join(sorted(GAZETTEER))
                    return self._refuse(
                        f"I don't know the place {plan.place!r}. Give a bounding box "
                        f"[west, south, east, north], or use a known place: {known}."
                    )
                box = named
            blocks = list(state.get("blocks", []))
            caveats = list(state.get("caveats", []))
            scenes: list[dict[str, Any]] = []
            for period in plan.periods:
                chosen = None
                for max_cloud in CLOUD_STEPS:
                    found = await self._tool(
                        "search_stac",
                        {
                            "bbox": list(box.as_tuple),
                            "start": period.start.isoformat(),
                            "end": period.end.isoformat(),
                            "max_cloud": max_cloud,
                            "limit": 20,
                            "collection": service.DEFAULT_COLLECTION,
                        },
                    )
                    blocks.append(ProvenanceBlock.model_validate(found["provenance"]))
                    if found["scenes"]:
                        chosen = found["scenes"][0]
                        if max_cloud > CLOUD_STEPS[0]:
                            caveats.append(
                                f"Only a cloudier scene was available for {period.start} to "
                                f"{period.end} ({chosen['cloud_cover']:.1f}% cloud cover)."
                            )
                        break
                if chosen is None:
                    return {
                        "blocks": blocks,
                        **self._refuse(
                            f"No Sentinel-2 scene with under {CLOUD_STEPS[-1]:.0f}% cloud cover "
                            f"covers this area between {period.start} and {period.end}."
                        ),
                    }
                scenes.append(chosen)
            sp.set_output({"scenes": [s["item_id"] for s in scenes]})
            return {
                "bbox": list(box.as_tuple),
                "scenes": scenes,
                "blocks": blocks,
                "caveats": caveats,
            }

    async def analyze(self, state: State) -> dict[str, Any]:
        plan = state["plan"]
        with self.tracer.span("analyze", kind="chain", input={"index": plan.index}) as sp:
            blocks = list(state["blocks"])
            caveats = list(state["caveats"])
            analysed: list[ProvenanceBlock] = []
            for scene in state["scenes"]:
                data = await self._tool(
                    "band_math",
                    {
                        "item_id": scene["item_id"],
                        "index": plan.index,
                        "bbox": state["bbox"],
                        "collection": service.DEFAULT_COLLECTION,
                    },
                )
                block = ProvenanceBlock.model_validate(data["provenance"])
                if data["stats"]["mean"] is None:
                    return self._refuse(
                        f"Scene {scene['item_id']} has no valid pixels over this area."
                    )
                blocks.append(block)
                analysed.append(block)
                if data["grid"]["clipped"]:
                    caveats.append(
                        "The area extends beyond the scene; only the covered part was used."
                    )
                negative = data["negative_reflectance_fraction"]
                if negative > 0.2:
                    caveats.append(
                        f"{negative:.0%} of pixels had slightly negative reflectance "
                        "and were clipped to zero before computing the index."
                    )
                if data["stats"]["valid_fraction"] < 0.8:
                    caveats.append(
                        f"Only {data['stats']['valid_fraction']:.0%} of the area had valid pixels."
                    )
            if len(analysed) == 2:
                first, second = (b.sources[0].datetime.date() for b in analysed)
                gap = _doy_gap(first, second)
                if gap > SEASON_GAP_DAYS:
                    caveats.append(
                        f"The two scenes are {gap} days apart in the seasonal cycle, so part of "
                        "the difference may be seasonal rather than a real change."
                    )
            sp.set_output({"block_ids": [str(b.block_id) for b in analysed]})
            return {"blocks": blocks, "caveats": list(dict.fromkeys(caveats))}

    async def check_evidence(self, state: State) -> dict[str, Any]:
        with self.tracer.span("check_evidence") as sp:
            for block in state["blocks"]:
                stored = self.store.get(block.block_id)
                expected = compute_args_hash(block.request.tool, block.request.args)
                if stored is None or stored != block or block.args_hash != expected:
                    sp.set_output({"failed": str(block.block_id)})
                    return self._refuse("The evidence could not be verified, so I won't answer.")
            sp.set_output({"verified_blocks": len(state["blocks"])})
            return {}

    async def narrate(self, state: State) -> dict[str, Any]:
        evidence = evidence_json(state["blocks"])
        feedback = state.get("feedback")
        with self.tracer.span(
            "narrate",
            kind="generation",
            input={
                "question": state["question"],
                "evidence": json.loads(evidence),
                "feedback": feedback,
            },
            metadata={"model": self.model.name, "attempt": state.get("attempts", 0) + 1},
        ) as sp:
            draft = await self.model.narrate(state["question"], evidence, feedback=feedback)
            sp.set_output(draft)
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1}

    async def validate(self, state: State) -> dict[str, Any]:
        with self.tracer.span("validate", input={"draft": state["draft"]}) as sp:
            report = validate_claims(state["draft"], state["blocks"])
            unsupported = [m.text for m in report.unsupported]
            sp.set_output({"ok": report.ok, "unsupported": unsupported})
        if report.ok:
            return {"unsupported": [], "feedback": None}
        if state["attempts"] < MAX_NARRATIONS:
            return {
                "unsupported": unsupported,
                "feedback": (
                    f"These numbers are not in the evidence: {', '.join(unsupported)}. "
                    "Use only numbers from the evidence."
                ),
            }
        return {
            "unsupported": unsupported,
            "template_used": True,
            "draft": template_answer(state["plan"], state["blocks"]),
        }

    async def finalize(self, state: State) -> dict[str, Any]:
        blocks = state.get("blocks", [])
        evidence = []
        for block in blocks:
            source = block.sources[0] if block.sources else None
            evidence.append(
                EvidenceLine(
                    cite=block.short_id,
                    tool=block.tool.name,
                    item_id=source.item_id if source else None,
                    date=source.datetime.date().isoformat() if source else None,
                    cloud_cover_pct=source.cloud_cover_pct if source else None,
                )
            )
        refused = "refusal" in state
        status: Literal["answered", "answered_from_template", "refused"] = (
            "refused"
            if refused
            else "answered_from_template"
            if state.get("template_used")
            else "answered"
        )
        final = AgentAnswer(
            status=status,
            question=state.get("question", ""),
            answer=state["refusal"] if refused else state["draft"],
            caveats=state.get("caveats", []),
            evidence=[] if refused else evidence,
            block_ids=[str(b.block_id) for b in blocks],
            run_id=self.run_id,
            plan=state.get("plan"),
            validation_ok=not refused and not state.get("template_used", False),
            unsupported_numbers=state.get("unsupported", []),
            model=self.model.name,
        )
        return {"final": final}

    # ---- routing -------------------------------------------------------------------------
    @staticmethod
    def _continue_or_finish(next_node: str) -> Any:
        def route(state: State) -> str:
            return "finalize" if "refusal" in state else next_node

        return route

    @staticmethod
    def _after_validate(state: State) -> str:
        if state.get("template_used") or not state.get("unsupported"):
            return "finalize"
        return "narrate"


def build_graph(run: _Run) -> Any:
    builder: StateGraph[State] = StateGraph(State)
    for name in (
        "intake",
        "plan",
        "select_data",
        "analyze",
        "check_evidence",
        "narrate",
        "validate",
        "finalize",
    ):
        builder.add_node(name, getattr(run, name))
    builder.add_edge(START, "intake")
    chain = ["intake", "plan", "select_data", "analyze", "check_evidence"]
    following = ["plan", "select_data", "analyze", "check_evidence", "narrate"]
    for current, nxt in zip(chain, following, strict=True):
        builder.add_conditional_edges(current, run._continue_or_finish(nxt))
    builder.add_edge("narrate", "validate")
    builder.add_conditional_edges("validate", run._after_validate)
    builder.add_edge("finalize", END)
    return builder.compile()


async def run_agent(
    question: str,
    *,
    model: AgentModel,
    server: FastMCP,
    store: ProvenanceStore,
    tracer: Tracer | None = None,
    today: date | None = None,
    run_id: str | None = None,
) -> AgentAnswer:
    """Answer one question. `server` is the MCP server whose tools the agent calls in-process."""
    tracer = tracer or NullTracer()
    run_id = run_id or uuid.uuid4().hex
    async with Client(server) as client:
        run = _Run(
            client=client,
            model=model,
            store=store,
            tracer=tracer,
            today=today or date.today(),
            run_id=run_id,
        )
        with tracer.span(
            "agent_run",
            kind="agent",
            input={"question": question, "run_id": run_id, "model": model.name},
        ) as root:
            state: State = await build_graph(run).ainvoke(
                {"question": question, "attempts": 0, "blocks": [], "caveats": []}
            )
            answer = state["final"]
            root.set_output(
                {
                    "status": answer.status,
                    "answer": answer.answer,
                    "block_ids": answer.block_ids,
                    "run_id": run_id,
                }
            )
    return answer
