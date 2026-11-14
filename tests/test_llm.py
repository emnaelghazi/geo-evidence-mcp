"""R2: the Anthropic adapter, checked against the real SDK with a mocked HTTP transport."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic

from geo_evidence_mcp.agent.llm import AnthropicModel, PlanError

PLAN_INPUT = {
    "task": "index_value",
    "index": "ndvi",
    "place": "tozeur oasis",
    "periods": [{"start": "2026-09-01", "end": "2026-09-30"}],
}


def message(content: list[dict[str, Any]], stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def model_with(reply: dict[str, Any], seen: list[dict[str, Any]] | None = None) -> AnthropicModel:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx2.Response(200, json=reply)

    client = AsyncAnthropic(
        api_key="test-key", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    return AnthropicModel(api_key="test-key", model="test-model", client=client)


def run_plan(model: AnthropicModel) -> Any:
    return asyncio.run(model.plan("Q?", today=date(2026, 11, 14), known_places=["tozeur oasis"]))


@pytest.mark.req("R2")
def test_plan_sends_a_forced_tool_call_and_parses_the_result() -> None:
    seen: list[dict[str, Any]] = []
    reply = message(
        [{"type": "tool_use", "id": "tu_1", "name": "submit_plan", "input": PLAN_INPUT}], "tool_use"
    )
    plan = run_plan(model_with(reply, seen))
    assert plan.task == "index_value" and plan.periods[0].start == date(2026, 9, 1)

    request = seen[0]
    assert request["model"] == "test-model" and request["messages"][0]["content"] == "Q?"
    assert request["tool_choice"] == {"type": "tool", "name": "submit_plan"}
    assert request["tools"][0]["name"] == "submit_plan"
    assert "task" in request["tools"][0]["input_schema"]["properties"]
    assert "2026-11-14" in request["system"] and "tozeur oasis" in request["system"]


@pytest.mark.req("R2")
def test_inconsistent_plan_from_the_model_is_a_plan_error() -> None:
    bad = {**PLAN_INPUT, "place": None, "bbox": None}
    reply = message(
        [{"type": "tool_use", "id": "tu_1", "name": "submit_plan", "input": bad}], "tool_use"
    )
    with pytest.raises(PlanError, match="inconsistent"):
        run_plan(model_with(reply))


@pytest.mark.req("R2")
def test_missing_tool_call_is_a_plan_error() -> None:
    with pytest.raises(PlanError, match="did not return a plan"):
        run_plan(model_with(message([{"type": "text", "text": "Sure!"}])))


@pytest.mark.req("R6")
def test_narration_sends_evidence_and_rejection_feedback() -> None:
    seen: list[dict[str, Any]] = []
    model = model_with(message([{"type": "text", "text": "Mean NDVI was 0.72 [abc12345]."}]), seen)
    text = asyncio.run(
        model.narrate("Q?", '[{"cite": "abc12345"}]', feedback="0.91 is not in the evidence.")
    )
    assert text == "Mean NDVI was 0.72 [abc12345]."
    content = seen[0]["messages"][0]["content"]
    assert "abc12345" in content and "0.91 is not in the evidence." in content
    assert "ONLY the evidence" in seen[0]["system"]
