"""Model access behind a small interface, so the graph is testable without any API."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock
from pydantic import ValidationError

from geo_evidence_mcp.agent.plan import Plan

PLAN_SYSTEM = """You turn a question about Earth observation into a structured plan.
Today is {today}. The data is Sentinel-2 L2A optical imagery at 10 m resolution, 2015 onward.
Known places: {places}. Use one of them as `place`, or a `bbox` only if the user typed coordinates.
Choose ndvi for vegetation/greenness questions and ndwi for open-water questions.
For a comparison between years, use the same calendar window in both years where the user
implies one (for example the same month); otherwise use whole years.
If the question cannot be answered with this data (finer than 10 m, no place or coordinates
given and none of the known places fits, non-optical quantities, forecasts), return
task=unsupported with a short reason. Never invent coordinates."""

NARRATE_SYSTEM = """You write the answer to a question using ONLY the evidence you are given.
Rules:
- Every number you state must appear in the evidence, or be a difference between two evidence
  values of the same quantity. You may round, but never invent or estimate numbers.
- Cite evidence with its id in square brackets, for example [3f9a1c2e].
- State which scene dates the numbers come from.
- Mention any caveat you are given. Do not repeat the caveat's numbers in new forms.
- If the evidence does not answer the question, say so plainly.
Be brief: three sentences or fewer."""


class PlanError(Exception):
    """The model did not return a usable plan."""


class AgentModel(Protocol):
    @property
    def name(self) -> str: ...

    async def plan(self, question: str, *, today: date, known_places: list[str]) -> Plan: ...

    async def narrate(self, question: str, evidence: str, *, feedback: str | None) -> str: ...


class AnthropicModel:
    """Claude via the Anthropic Messages API. Plan = forced tool call; narration = plain text."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: AsyncAnthropic | None = None,
        max_tokens: int = 1200,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return self._model

    async def plan(self, question: str, *, today: date, known_places: list[str]) -> Plan:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=PLAN_SYSTEM.format(today=today.isoformat(), places=", ".join(known_places)),
            messages=[{"role": "user", "content": question}],
            tools=[
                {
                    "name": "submit_plan",
                    "description": "Submit the structured plan for this question.",
                    "input_schema": Plan.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "submit_plan"},
        )
        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == "submit_plan":
                try:
                    return Plan.model_validate(block.input)
                except ValidationError as exc:
                    raise PlanError(
                        f"the model's plan was inconsistent: {exc.errors()[0]['msg']}"
                    ) from exc
        raise PlanError("the model did not return a plan")

    async def narrate(self, question: str, evidence: str, *, feedback: str | None) -> str:
        content = f"Question: {question}\n\nEvidence (JSON):\n{evidence}"
        if feedback:
            content += f"\n\nYour previous answer was rejected. {feedback}"
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=NARRATE_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in response.content if isinstance(b, TextBlock)).strip()


class ScriptedModel:
    """Deterministic stand-in for tests: a plan function and a narration function."""

    def __init__(
        self,
        plan_for: Callable[[str], Plan],
        narrate_with: Callable[[str, str | None, int], str],
    ) -> None:
        self._plan_for = plan_for
        self._narrate_with = narrate_with
        self.narrate_calls = 0
        self.feedback_seen: list[str | None] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def plan(self, question: str, *, today: date, known_places: list[str]) -> Plan:
        return self._plan_for(question)

    async def narrate(self, question: str, evidence: str, *, feedback: str | None) -> str:
        self.feedback_seen.append(feedback)
        self.narrate_calls += 1
        return self._narrate_with(evidence, feedback, self.narrate_calls)


def evidence_items(evidence: str) -> list[dict[str, Any]]:
    """Parse the evidence JSON handed to `narrate` (helper for scripted narrators)."""
    items: list[dict[str, Any]] = json.loads(evidence)
    return items
