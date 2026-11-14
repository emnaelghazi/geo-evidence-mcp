"""Tracing behind a tiny interface: no-op, in-memory (tests), or Langfuse (R11)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from langfuse import Langfuse

from geo_evidence_mcp.config import Settings

_LANGFUSE_KIND = {"agent", "tool", "generation", "chain", "span"}


class Span(Protocol):
    def set_output(self, output: Any) -> None: ...
    def set_metadata(self, **metadata: Any) -> None: ...


class Tracer(Protocol):
    def span(
        self,
        name: str,
        *,
        kind: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Span]: ...


class _NullSpan:
    def set_output(self, output: Any) -> None:
        pass

    def set_metadata(self, **metadata: Any) -> None:
        pass


class NullTracer:
    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        yield _NullSpan()


@dataclass
class SpanRecord:
    name: str
    kind: str
    input: Any
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent: int | None = None
    duration_ms: float = 0.0
    error: str | None = None


class RecordingTracer:
    """Keeps every span in memory, so tests can prove a run is reconstructible from its trace."""

    def __init__(self) -> None:
        self.records: list[SpanRecord] = []
        self._stack: list[int] = []

    def span(
        self,
        name: str,
        *,
        kind: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Span]:
        return self._span(name, kind, input, metadata)

    @contextmanager
    def _span(
        self,
        name: str,
        kind: str,
        input: Any,
        metadata: dict[str, Any] | None,
    ) -> Iterator[Span]:
        record = SpanRecord(
            name=name,
            kind=kind,
            input=input,
            metadata=dict(metadata or {}),
            parent=self._stack[-1] if self._stack else None,
        )
        index = len(self.records)
        self.records.append(record)
        self._stack.append(index)
        started = time.perf_counter()

        class _Span:
            def set_output(self, output: Any) -> None:
                record.output = output

            def set_metadata(self, **meta: Any) -> None:
                record.metadata.update(meta)

        try:
            yield _Span()
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000
            self._stack.pop()

    def named(self, name: str) -> list[SpanRecord]:
        return [r for r in self.records if r.name == name]


class LangfuseTracer:
    """Sends spans to Langfuse (OpenTelemetry under the hood)."""

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        as_type: Any = kind if kind in _LANGFUSE_KIND else "span"
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        ) as observation:

            class _Span:
                def set_output(self, output: Any) -> None:
                    observation.update(output=output)

                def set_metadata(self, **meta: Any) -> None:
                    observation.update(metadata=meta)

            yield _Span()

    def flush(self) -> None:
        self._client.flush()


def make_tracer(settings: Settings) -> Tracer:
    """Langfuse when both keys are configured, otherwise a no-op tracer."""
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
        return LangfuseTracer(client)
    return NullTracer()
