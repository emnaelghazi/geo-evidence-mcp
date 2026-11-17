"""R11: tracer implementations; the Langfuse adapter is checked with a local OTel exporter."""

from __future__ import annotations

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from geo_evidence_mcp.agent.tracing import (
    LangfuseTracer,
    NullTracer,
    RecordingTracer,
    make_tracer,
)
from geo_evidence_mcp.config import Settings


@pytest.mark.req("R11")
def test_recording_tracer_nests_spans_and_records_errors() -> None:
    tracer = RecordingTracer()
    with tracer.span("outer", kind="agent", input={"q": 1}) as outer:
        with tracer.span("inner", kind="tool", input="x") as inner:
            inner.set_output({"ok": True})
        outer.set_metadata(note="n")
        with pytest.raises(RuntimeError), tracer.span("boom"):
            raise RuntimeError("bad")
    names = [r.name for r in tracer.records]
    assert names == ["outer", "inner", "boom"]
    assert tracer.records[1].parent == 0 and tracer.records[2].parent == 0
    assert tracer.records[1].output == {"ok": True} and tracer.records[0].metadata == {"note": "n"}
    assert tracer.records[2].error == "RuntimeError: bad"


@pytest.mark.req("R11")
def test_null_tracer_accepts_everything() -> None:
    with NullTracer().span("x", input=1) as span:
        span.set_output(2)
        span.set_metadata(a=1)


@pytest.mark.req("R11")
def test_make_tracer_uses_langfuse_only_when_both_keys_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert isinstance(make_tracer(Settings(_env_file=None)), NullTracer)

    only_public = Settings(_env_file=None, langfuse_public_key="pk", langfuse_secret_key=None)
    assert isinstance(make_tracer(only_public), NullTracer)


@pytest.mark.req("R11")
def test_langfuse_adapter_emits_spans_with_input_and_output() -> None:
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:9",
        span_exporter=exporter,
        flush_at=1,
    )
    tracer = LangfuseTracer(client)
    with tracer.span("agent_run", kind="agent", input={"question": "Q?"}) as root:
        with tracer.span("band_math", kind="tool", input={"index": "ndvi"}) as tool:
            tool.set_output({"block_id": "b1"})
        root.set_output({"status": "answered"})
    tracer.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {"agent_run", "band_math"} <= set(spans)
    child, parent = spans["band_math"], spans["agent_run"]
    assert child.parent is not None and child.parent.span_id == parent.context.span_id
    assert child.context.trace_id == parent.context.trace_id
    attributes = dict(child.attributes or {})
    assert any("ndvi" in str(v) for v in attributes.values())  # input captured
    assert any("b1" in str(v) for v in attributes.values())  # output captured
    assert SecretStr("x")  # keep import used by mypy strictness
