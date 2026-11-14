"""`gem-ask "question"`: run the agent on live Sentinel-2 data with Claude (needs an API key)."""

from __future__ import annotations

import argparse
import asyncio

from geo_evidence_mcp.agent.graph import run_agent
from geo_evidence_mcp.agent.llm import AnthropicModel
from geo_evidence_mcp.agent.tracing import LangfuseTracer, make_tracer
from geo_evidence_mcp.config import get_settings
from geo_evidence_mcp.server import build_server
from geo_evidence_mcp.store import make_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.anthropic_api_key is None:
        print("Set ANTHROPIC_API_KEY in your .env first.")
        return 2
    store = make_store(settings)
    tracer = make_tracer(settings)
    model = AnthropicModel(
        api_key=settings.anthropic_api_key.get_secret_value(), model=settings.anthropic_model
    )
    answer = asyncio.run(
        run_agent(
            args.question, model=model, server=build_server(store=store), store=store, tracer=tracer
        )
    )
    if isinstance(tracer, LangfuseTracer):
        tracer.flush()
    print(answer.render())
    return 0 if answer.status != "refused" else 1


if __name__ == "__main__":
    raise SystemExit(main())
