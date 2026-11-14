"""R2/R3/R6/R11: the agent end to end, offline, on synthetic scenes with a scripted model."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from geo_evidence_mcp.agent.graph import AgentAnswer, run_agent
from geo_evidence_mcp.agent.llm import PlanError, ScriptedModel, evidence_items
from geo_evidence_mcp.agent.plan import Period, Plan
from geo_evidence_mcp.agent.tracing import RecordingTracer
from geo_evidence_mcp.aoi import resolve_place
from geo_evidence_mcp.provenance import compute_args_hash
from geo_evidence_mcp.server import build_server
from geo_evidence_mcp.store import MemoryStore
from geo_evidence_mcp.validator import validate_claims
from tests.conftest import FakeCatalog, make_scene, scene_to_item

TODAY = date(2026, 11, 14)
SEPT_2025 = Period(start=date(2025, 9, 1), end=date(2025, 9, 30))
SEPT_2026 = Period(start=date(2026, 9, 1), end=date(2026, 9, 30))


def value_plan(index: str = "ndvi", period: Period = SEPT_2026, **kw: Any) -> Plan:
    return Plan(task="index_value", index=index, place="Tozeur oasis", periods=[period], **kw)


def faithful(evidence: str, feedback: str | None, attempt: int) -> str:
    items = evidence_items(evidence)
    index = next(k for k in items[0]["outputs"] if k.endswith(".mean")).split(".")[0]
    means = [i["outputs"][f"{index}.mean"] for i in items]
    parts = [
        f"On {i['date']} the mean {index.upper()} was {m:.2f} [{i['cite']}]."
        for i, m in zip(items, means, strict=True)
    ]
    if len(items) == 2:
        parts.append(f"That is a change of {means[1] - means[0]:+.2f}.")
    return " ".join(parts)


class World:
    """Server + store + catalog with two Tozeur scenes (Sept 2025 greener, Sept 2026)."""

    def __init__(self, tmp_path: Path, *, extra: list[Any] | None = None, **scene_kw: Any) -> None:
        self.before = make_scene(
            tmp_path, item_id="S2_2025", when=datetime(2025, 9, 12, 10, tzinfo=UTC), palm_nir=5500
        )
        self.after = make_scene(
            tmp_path, item_id="S2_2026", when=datetime(2026, 9, 15, 10, tzinfo=UTC), **scene_kw
        )
        items = [scene_to_item(self.before.scene), scene_to_item(self.after.scene)]
        self.store = MemoryStore()
        self.server = build_server(client=FakeCatalog(items + (extra or [])), store=self.store)

    def ask(self, question: str, model: ScriptedModel, **kw: Any) -> AgentAnswer:
        return asyncio.run(
            run_agent(
                question,
                model=model,
                server=self.server,
                store=kw.pop("store", self.store),
                today=TODAY,
                **kw,
            )
        )


def model_for(plans: dict[str, Plan], narrate: Any = faithful) -> ScriptedModel:
    return ScriptedModel(lambda q: plans[q], narrate)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


# ---- the five golden-path questions ------------------------------------------------------
@pytest.mark.req("R2")
def test_golden_1_ndvi_value_for_a_named_place(world: World) -> None:
    q = "What was the mean NDVI of the Tozeur oasis in September 2026?"
    answer = world.ask(q, model_for({q: value_plan()}))
    assert answer.status == "answered" and answer.validation_ok
    assert "[" + answer.evidence[-1].cite + "]" in answer.answer
    assert answer.evidence[-1].item_id == "S2_2026"
    assert {
        b.tool.name
        for b in map(world.store.get, map(__import__("uuid").UUID, answer.block_ids))
        if b
    } == {
        "search_stac",
        "band_math",
    }


@pytest.mark.req("R2")
def test_golden_2_change_between_two_years(world: World) -> None:
    q = "How did NDVI change in the Tozeur oasis between September 2025 and September 2026?"
    plan = Plan(task="index_change", index="ndvi", place="Tozeur", periods=[SEPT_2026, SEPT_2025])
    answer = world.ask(q, model_for({q: plan}))
    assert answer.status == "answered"
    assert [e.item_id for e in answer.evidence if e.tool == "band_math"] == ["S2_2025", "S2_2026"]
    assert "change of -" in answer.answer  # greener before, so the change is negative
    assert not answer.caveats


@pytest.mark.req("R2")
def test_golden_3_open_water_index(world: World) -> None:
    q = "Is there open water in the Tozeur oasis in September 2026?"
    answer = world.ask(q, model_for({q: value_plan("ndwi")}))
    assert answer.status == "answered" and "NDWI" in answer.answer
    block = world.store.get(__import__("uuid").UUID(answer.block_ids[-1]))
    assert block is not None and block.computation.operation == "ndwi"


@pytest.mark.req("R2")
def test_golden_4_explicit_bounding_box(world: World) -> None:
    q = "NDVI for 8.140,33.895,8.150,33.905 in September 2026?"
    plan = Plan(
        task="index_value", index="ndvi", bbox=[8.140, 33.895, 8.150, 33.905], periods=[SEPT_2026]
    )
    answer = world.ask(q, model_for({q: plan}))
    assert answer.status == "answered"
    stored = [world.store.get(__import__("uuid").UUID(i)) for i in answer.block_ids]
    band_math = next(b for b in stored if b and b.tool.name == "band_math")
    assert band_math.request.args["bbox"] == [8.140, 33.895, 8.150, 33.905]


@pytest.mark.req("R3")
def test_golden_5_unsupported_question_is_refused_without_touching_data(world: World) -> None:
    q = "What is the NDVI of my backyard at 1 metre resolution?"
    plan = Plan(
        task="unsupported", unsupported_reason="Sentinel-2 pixels are 10 m, and no place was given."
    )
    answer = world.ask(q, model_for({q: plan}))
    assert answer.status == "refused" and "10 m" in answer.answer
    assert answer.block_ids == [] and answer.evidence == []


# ---- refusals ------------------------------------------------------------------------------
@pytest.mark.req("R3")
def test_unknown_place_is_refused_and_known_places_are_listed(world: World) -> None:
    q = "NDVI of Gafsa in September 2026?"
    plan = Plan(task="index_value", index="ndvi", place="Gafsa", periods=[SEPT_2026])
    answer = world.ask(q, model_for({q: plan}))
    assert answer.status == "refused" and "tozeur oasis" in answer.answer and answer.block_ids == []


@pytest.mark.req("R3")
def test_period_without_scenes_is_refused_after_the_search_is_recorded(world: World) -> None:
    q = "NDVI of the Tozeur oasis in 2019?"
    period = Period(start=date(2019, 1, 1), end=date(2019, 12, 31))
    answer = world.ask(q, model_for({q: value_plan(period=period)}))
    assert answer.status == "refused" and "No Sentinel-2 scene" in answer.answer
    assert len(answer.block_ids) == 2  # both cloud thresholds were searched and recorded


@pytest.mark.req("R3")
def test_cloudier_scene_is_used_with_a_caveat(tmp_path: Path) -> None:
    world = World(tmp_path, cloud=40.0)
    q = "NDVI of the Tozeur oasis in September 2026?"
    answer = world.ask(q, model_for({q: value_plan()}))
    assert answer.status == "answered"
    assert any("cloudier scene" in c and "40.0% cloud" in c for c in answer.caveats)


@pytest.mark.req("R3")
def test_seasonal_gap_between_compared_scenes_is_flagged(tmp_path: Path) -> None:
    world = World(tmp_path)
    winter = make_scene(tmp_path, item_id="S2_WINTER", when=datetime(2026, 3, 1, 10, tzinfo=UTC))
    server = build_server(
        client=FakeCatalog([scene_to_item(world.before.scene), scene_to_item(winter.scene)]),
        store=world.store,
    )
    q = "Change in NDVI between September 2025 and March 2026?"
    march = Period(start=date(2026, 3, 1), end=date(2026, 3, 31))
    plan = Plan(task="index_change", index="ndvi", place="Tozeur", periods=[SEPT_2025, march])
    answer = asyncio.run(
        run_agent(q, model=model_for({q: plan}), server=server, store=world.store, today=TODAY)
    )
    assert any("apart in the seasonal cycle" in c for c in answer.caveats)


@pytest.mark.req("R3")
@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_empty_or_oversized_questions_are_refused(world: World, question: str) -> None:
    answer = world.ask(question, model_for({}))
    assert answer.status == "refused" and "question" in answer.answer.lower()


@pytest.mark.req("R3")
def test_model_that_cannot_plan_leads_to_a_refusal(world: World) -> None:
    def broken(_: str) -> Plan:
        raise PlanError("the model did not return a plan")

    answer = world.ask("anything", ScriptedModel(broken, faithful))
    assert answer.status == "refused" and "could not turn that into a data request" in answer.answer


# ---- the validator loop (R6) ---------------------------------------------------------------
@pytest.mark.req("R6")
def test_fabricated_number_is_rejected_then_corrected(world: World) -> None:
    q = "NDVI of the Tozeur oasis in September 2026?"

    def liar_then_honest(evidence: str, feedback: str | None, attempt: int) -> str:
        return (
            "The mean NDVI was 0.91 [x]." if attempt == 1 else faithful(evidence, feedback, attempt)
        )

    model = model_for({q: value_plan()}, liar_then_honest)
    answer = world.ask(q, model)
    assert answer.status == "answered" and answer.validation_ok
    assert model.narrate_calls == 2 and model.feedback_seen[0] is None
    assert "0.91" in (model.feedback_seen[1] or "") and "not in the evidence" in (
        model.feedback_seen[1] or ""
    )
    assert "0.91" not in answer.answer


@pytest.mark.req("R6")
def test_persistent_fabrication_falls_back_to_a_template_that_validates(world: World) -> None:
    q = "NDVI of the Tozeur oasis in September 2026?"
    model = model_for({q: value_plan()}, lambda *_: "Vegetation cover is 97.5% and NDVI is 0.91.")
    answer = world.ask(q, model)
    assert answer.status == "answered_from_template" and not answer.validation_ok
    assert model.narrate_calls == 3
    assert {"97.5%", "0.91"} <= set(answer.unsupported_numbers)
    blocks = [b for b in map(world.store.get, map(__import__("uuid").UUID, answer.block_ids)) if b]
    assert validate_claims(answer.answer, blocks).ok and "0.91" not in answer.answer


@pytest.mark.req("R5")
def test_unverifiable_evidence_blocks_the_answer(world: World) -> None:
    q = "NDVI of the Tozeur oasis in September 2026?"
    answer = world.ask(
        q, model_for({q: value_plan()}), store=MemoryStore()
    )  # a different, empty store
    assert answer.status == "refused" and "could not be verified" in answer.answer


# ---- tracing (R11) -------------------------------------------------------------------------
@pytest.mark.req("R11")
def test_a_run_can_be_reconstructed_from_its_trace_alone(world: World) -> None:
    q = "How did NDVI change in the Tozeur oasis between September 2025 and September 2026?"
    plan = Plan(task="index_change", index="ndvi", place="Tozeur", periods=[SEPT_2025, SEPT_2026])
    tracer = RecordingTracer()
    answer = world.ask(q, model_for({q: plan}), tracer=tracer, run_id="run-abc")

    root = tracer.named("agent_run")[0]
    assert root.input["question"] == q and root.output["run_id"] == "run-abc"
    assert root.output["answer"] == answer.answer and root.output["block_ids"] == answer.block_ids
    assert Plan.model_validate(tracer.named("plan")[0].output) == answer.plan

    tool_spans = [r for r in tracer.records if r.kind == "tool"]
    assert len(tool_spans) == len(answer.block_ids) == 4  # 2 searches + 2 analyses
    for span in tool_spans:  # each recorded call hashes to the block it produced
        block = world.store.get(__import__("uuid").UUID(span.output["block_id"]))
        assert block is not None
        assert (
            compute_args_hash(span.name, span.input) == block.args_hash == span.output["args_hash"]
        )
        assert block.tool_trace.run_id == "run-abc"
    assert tracer.named("narrate")[0].output == answer.answer
    assert all(r.parent is not None for r in tracer.records if r.name != "agent_run")


@pytest.mark.req("R11")
def test_failed_node_is_visible_in_the_trace(world: World) -> None:
    q = "NDVI of Gafsa?"
    plan = Plan(task="index_value", index="ndvi", place="Gafsa", periods=[SEPT_2026])
    tracer = RecordingTracer()
    answer = world.ask(q, model_for({q: plan}), tracer=tracer)
    assert answer.status == "refused"
    assert tracer.named("agent_run")[0].output["status"] == "refused"
    assert tracer.named("narrate") == []  # nothing was narrated


# ---- gazetteer and plan ---------------------------------------------------------------------
@pytest.mark.req("R2")
@pytest.mark.parametrize(
    "name", ["Tozeur", "tozeur oasis", "Tozeur-Oasis", "TOZEUR_OASIS", " tozeur  oasis "]
)
def test_place_resolution_is_forgiving(name: str) -> None:
    assert resolve_place(name) is not None
    assert resolve_place("Atlantis") is None


@pytest.mark.req("R2")
def test_plan_validation() -> None:
    with pytest.raises(ValueError, match="unsupported_reason"):
        Plan(task="unsupported")
    with pytest.raises(ValueError, match="exactly one of place or bbox"):
        Plan(task="index_value", index="ndvi", periods=[SEPT_2026])
    with pytest.raises(ValueError, match="exactly one of place or bbox"):
        Plan(task="index_value", index="ndvi", place="x", bbox=[1, 2, 3, 4], periods=[SEPT_2026])
    with pytest.raises(ValueError, match="1 period"):
        Plan(task="index_value", index="ndvi", place="x", periods=[SEPT_2025, SEPT_2026])
    with pytest.raises(ValueError, match="overlap"):
        Plan(task="index_change", index="ndvi", place="x", periods=[SEPT_2026, SEPT_2026])
    plan = Plan(task="index_change", index="ndvi", place="x", periods=[SEPT_2026, SEPT_2025])
    assert [p.start.year for p in plan.periods] == [2025, 2026]  # sorted chronologically


# ---- M5: systematic refusals and low-confidence flags -------------------------------------
@pytest.mark.req("R3")
@pytest.mark.parametrize(
    "question",
    [
        "Show me NDVI at 1 m resolution for the Tozeur oasis.",
        "Forecast the NDVI of the Tozeur oasis for next year.",
        "What was the temperature in Tozeur in July 2026?",
        "How many palm trees are in the Tozeur oasis?",
    ],
)
def test_traps_are_refused_before_the_model_is_called(world: World, question: str) -> None:
    def must_not_plan(_: str) -> Plan:
        raise AssertionError("the model should not be consulted for a trap")

    model = ScriptedModel(must_not_plan, faithful)
    answer = world.ask(question, model)
    assert answer.status == "refused" and model.plan_calls == 0
    assert answer.block_ids == [] and answer.confidence == "normal"


@pytest.mark.req("R3")
def test_oversized_area_is_refused_before_any_search(world: World) -> None:
    q = "NDVI of the whole region 8.0,33.0,8.5,33.5 in September 2026?"
    plan = Plan(task="index_value", index="ndvi", bbox=[8.0, 33.0, 8.5, 33.5], periods=[SEPT_2026])
    answer = world.ask(q, model_for({q: plan}))
    assert answer.status == "refused" and "km2" in answer.answer and answer.block_ids == []


@pytest.mark.req("R3")
def test_future_period_chosen_by_the_model_is_still_refused(world: World) -> None:
    q = "NDVI of the Tozeur oasis in January 2027?"
    period = Period(start=date(2027, 1, 1), end=date(2027, 1, 31))
    answer = world.ask(q, model_for({q: value_plan(period=period)}))
    assert answer.status == "refused" and "future" in answer.answer and answer.block_ids == []


@pytest.mark.req("R3")
def test_tiny_area_is_answered_but_flagged_low_confidence(world: World) -> None:
    q = "NDVI at 8.1400,33.9000,8.1405,33.9005 in September 2026?"
    plan = Plan(
        task="index_value", index="ndvi", bbox=[8.14, 33.9, 8.1405, 33.9005], periods=[SEPT_2026]
    )
    answer = world.ask(q, model_for({q: plan}))
    assert answer.status == "answered" and answer.confidence == "low"
    assert any(
        c.startswith("Low confidence:") and "fewer than 100 pixels" in c for c in answer.caveats
    )


def _cloudy_agent_world(tmp_path: Path, fraction: float) -> World:
    from geo_evidence_mcp.aoi import TOZEUR
    from geo_evidence_mcp.raster import load_window

    probe = make_scene(tmp_path / "probe")
    col, _row, width, _height = load_window(probe.scene, "red", TOZEUR).window
    last = col // 2 + int(width / 2 * fraction)
    cells = (slice(0, 225), slice(0, last))  # the whole tile's rows, columns up to `last`
    world = World(tmp_path / "w")
    cloudy = make_scene(
        tmp_path / "cloud",
        item_id="S2_CLOUDY",
        when=datetime(2026, 9, 20, 10, tzinfo=UTC),
        cloud_cells=cells,
        cloud_dn=9000,
    )
    world.server = build_server(
        client=FakeCatalog([scene_to_item(cloudy.scene)]), store=world.store
    )
    return world


@pytest.mark.req("R3")
def test_heavy_cloud_cover_lowers_confidence_and_is_explained(tmp_path: Path) -> None:
    world = _cloudy_agent_world(tmp_path, 0.7)
    q = "NDVI of the Tozeur oasis in September 2026?"
    answer = world.ask(q, model_for({q: value_plan()}))
    assert answer.status == "answered" and answer.confidence == "low"
    assert any("Low confidence" in c and "cloud, shadow or cirrus" in c for c in answer.caveats)


@pytest.mark.req("R3")
def test_moderate_cloud_cover_is_a_caveat_not_a_low_confidence_flag(tmp_path: Path) -> None:
    world = _cloudy_agent_world(tmp_path, 0.3)
    q = "NDVI of the Tozeur oasis in September 2026?"
    answer = world.ask(q, model_for({q: value_plan()}))
    assert answer.status == "answered" and answer.confidence == "normal"
    assert any("was cloudy and was excluded" in c for c in answer.caveats)


@pytest.mark.req("R3")
def test_fully_cloudy_scene_is_refused_with_its_evidence_kept(tmp_path: Path) -> None:
    world = _cloudy_agent_world(tmp_path, 2.0)
    q = "NDVI of the Tozeur oasis in September 2026?"
    answer = world.ask(q, model_for({q: value_plan()}))
    assert answer.status == "refused" and "no valid pixels" in answer.answer
    assert len(answer.block_ids) == 2  # the search and the (empty) analysis stay on record
