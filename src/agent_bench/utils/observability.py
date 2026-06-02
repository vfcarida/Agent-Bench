"""Observability hooks: span-based tracing compatible with OpenTelemetry."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator
from uuid import uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class Span:
    """A lightweight span representing a unit of work."""

    name: str
    span_id: str = field(default_factory=lambda: str(uuid4())[:8])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok | error

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is not None:
            return (self.end_time - self.start_time) * 1000
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })

    def end(self, status: str = "ok") -> None:
        self.end_time = time.perf_counter()
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


class SpanCollector:
    """Collects spans during benchmark execution for observability export."""

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._active_span: Span | None = None

    @property
    def spans(self) -> list[Span]:
        return self._spans

    @contextmanager
    def trace(self, name: str, **attributes: Any) -> Generator[Span, None, None]:
        """Context manager for creating and auto-ending a span."""
        parent_id = self._active_span.span_id if self._active_span else None
        span = Span(name=name, parent_id=parent_id, attributes=attributes)
        previous = self._active_span
        self._active_span = span
        try:
            yield span
            span.end("ok")
        except Exception as e:
            span.set_attribute("error", str(e))
            span.end("error")
            raise
        finally:
            self._active_span = previous
            self._spans.append(span)
            logger.debug(
                "span_completed",
                name=name,
                duration_ms=span.duration_ms,
                status=span.status,
            )

    def export_otel_format(self) -> list[dict[str, Any]]:
        """Export spans in OpenTelemetry-compatible format."""
        return [
            {
                "traceId": "bench-" + (s.parent_id or s.span_id),
                "spanId": s.span_id,
                "parentSpanId": s.parent_id or "",
                "operationName": s.name,
                "startTime": int(s.start_time * 1_000_000),  # microseconds
                "duration": int((s.duration_ms or 0) * 1000),  # microseconds
                "tags": [{"key": k, "value": str(v)} for k, v in s.attributes.items()],
                "logs": [
                    {"timestamp": e["timestamp"], "fields": [{"key": "event", "value": e["name"]}]}
                    for e in s.events
                ],
            }
            for s in self._spans
        ]

    def summary(self) -> dict[str, Any]:
        """Generate a summary of collected spans."""
        if not self._spans:
            return {"total_spans": 0}

        durations = [s.duration_ms for s in self._spans if s.duration_ms is not None]
        errors = [s for s in self._spans if s.status == "error"]

        return {
            "total_spans": len(self._spans),
            "total_duration_ms": sum(durations),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "error_count": len(errors),
            "span_names": list(set(s.name for s in self._spans)),
        }

    def reset(self) -> None:
        self._spans.clear()
        self._active_span = None


# Global collector instance
_global_collector = SpanCollector()


def get_collector() -> SpanCollector:
    return _global_collector


@contextmanager
def bench_span(name: str, **attributes: Any) -> Generator[Span, None, None]:
    """Convenience function using global collector."""
    with _global_collector.trace(name, **attributes) as span:
        yield span
