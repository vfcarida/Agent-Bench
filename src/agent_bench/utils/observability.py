"""Observability hooks: span-based tracing compatible with OpenTelemetry."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
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
            "timestamp": datetime.now(UTC).isoformat(),
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
        """Export spans in OpenTelemetry-compatible Jaeger/Zipkin JSON format."""
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

    def export_otlp_genai_json(self, service_name: str = "agent-bench") -> dict[str, Any]:
        """Export collected spans in OTLP/JSON resourceSpans format with GenAI semantic conventions."""
        otlp_spans: list[dict[str, Any]] = []
        for s in self._spans:
            start_ns = int(s.start_time * 1_000_000_000)
            dur_ns = int((s.duration_ms or 0.0) * 1_000_000)
            otlp_spans.append({
                "traceId": ("bench-" + (s.parent_id or s.span_id)).ljust(32, "0")[:32],
                "spanId": s.span_id.ljust(16, "0")[:16],
                "parentSpanId": (s.parent_id.ljust(16, "0")[:16] if s.parent_id else ""),
                "name": s.name,
                "kind": "SPAN_KIND_INTERNAL",
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(start_ns + dur_ns),
                "attributes": [_to_otlp_kv(k, v) for k, v in s.attributes.items()],
                "events": [
                    {
                        "name": e["name"],
                        "timeUnixNano": str(start_ns),
                        "attributes": [
                            _to_otlp_kv(ek, ev)
                            for ek, ev in e.get("attributes", {}).items()
                        ],
                    }
                    for e in s.events
                ],
                "status": {
                    "code": "STATUS_CODE_OK" if s.status == "ok" else "STATUS_CODE_ERROR"
                },
            })

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _to_otlp_kv("service.name", service_name),
                            _to_otlp_kv("service.version", "1.0.0"),
                            _to_otlp_kv("telemetry.sdk.name", "agent-bench-otel-genai"),
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "agent_bench.genai.observability",
                                "version": "1.0.0",
                            },
                            "spans": otlp_spans,
                        }
                    ],
                }
            ]
        }

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
            "span_names": list({s.name for s in self._spans}),
        }

    def reset(self) -> None:
        self._spans.clear()
        self._active_span = None


# OpenTelemetry GenAI Semantic Convention attribute keys
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_COST_USD = "gen_ai.usage.cost_usd"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_AUTONOMY_TIER = "gen_ai.agent.autonomy_tier"
GEN_AI_SAFETY_GATED = "gen_ai.safety.gated"
GEN_AI_SAFETY_VIOLATIONS = "gen_ai.safety.violations"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TASK_ID = "gen_ai.task.id"
GEN_AI_DOMAIN = "gen_ai.domain"


def _to_otlp_kv(key: str, value: Any) -> dict[str, Any]:
    """Convert a Python key/value pair into an OTLP Attribute KeyValue dict."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def convert_run_traces_to_otel_genai(
    run_manifest: dict[str, Any],
    traces: list[dict[str, Any]] | None = None,
    *,
    service_name: str = "agent-bench",
) -> dict[str, Any]:
    """Convert an Agent-Bench run manifest and JSONL traces into OTLP/JSON GenAI spans.

    Maps scorecards, task results, and step-level traces to OpenTelemetry GenAI
    semantic conventions (`gen_ai.system`, `gen_ai.usage.input_tokens`,
    `gen_ai.agent.autonomy_tier`, `gen_ai.safety.gated`, etc.).
    """
    collector = SpanCollector()
    run_id = str(run_manifest.get("run_id", "run_unknown"))
    suite_id = str(run_manifest.get("suite_id", "suite_unknown"))
    metrics = run_manifest.get("metrics", {})

    with collector.trace(
        f"invoke_benchmark_suite {suite_id}",
        **{
            GEN_AI_SYSTEM: "agent_bench",
            GEN_AI_OPERATION_NAME: "invoke_benchmark_suite",
            "gen_ai.run.id": run_id,
            "gen_ai.suite.id": suite_id,
        },
    ):
        for sc in metrics.get("scorecards", []):
            sys_id = str(sc.get("system_id", "unknown"))
            dom = str(sc.get("domain", "unknown"))
            with collector.trace(
                f"invoke_agent {sys_id}",
                **{
                    GEN_AI_SYSTEM: "agent_bench",
                    GEN_AI_OPERATION_NAME: "invoke_agent",
                    GEN_AI_AGENT_ID: sys_id,
                    GEN_AI_DOMAIN: dom,
                    GEN_AI_AGENT_AUTONOMY_TIER: str(sc.get("autonomy_tier", "unclassified")),
                    GEN_AI_SAFETY_GATED: bool(sc.get("safety_gated", False)),
                    GEN_AI_SAFETY_VIOLATIONS: int(sc.get("safety_violations", 0)),
                    "gen_ai.score.functional": float(sc.get("functional_score", 0.0)),
                    "gen_ai.score.pass_hat_3": float(sc.get("pass_hat_3", 0.0)),
                    GEN_AI_USAGE_COST_USD: float(sc.get("cost_per_successful_task", 0.0)),
                },
            ):
                pass

        for tr in metrics.get("task_results", []):
            sys_id = str(tr.get("system_id", "unknown"))
            task_id = str(tr.get("task_id", "unknown"))
            dom = str(tr.get("domain", "unknown"))
            with collector.trace(
                f"chat {task_id}",
                **{
                    GEN_AI_SYSTEM: "agent_bench",
                    GEN_AI_OPERATION_NAME: "chat",
                    GEN_AI_AGENT_ID: sys_id,
                    GEN_AI_TASK_ID: task_id,
                    GEN_AI_DOMAIN: dom,
                    GEN_AI_USAGE_INPUT_TOKENS: int(tr.get("tokens_in", 0)),
                    GEN_AI_USAGE_OUTPUT_TOKENS: int(tr.get("tokens_out", 0)),
                    GEN_AI_USAGE_COST_USD: float(tr.get("cost_usd", 0.0)),
                    GEN_AI_SAFETY_GATED: bool(
                        tr.get("policy_violated", False) or tr.get("safety_violation", False)
                    ),
                    "gen_ai.task.passed": bool(tr.get("passed", False)),
                },
            ):
                pass

        for ev in traces or []:
            ev_type = str(ev.get("event_type", "event"))
            data = ev.get("data", {}) if isinstance(ev.get("data"), dict) else {}
            if ev_type in ("tool_call", "user_action"):
                t_name = str(data.get("tool_name", data.get("action_type", "tool")))
                with collector.trace(
                    f"execute_tool {t_name}",
                    **{
                        GEN_AI_SYSTEM: "agent_bench",
                        GEN_AI_OPERATION_NAME: "execute_tool",
                        GEN_AI_TOOL_NAME: t_name,
                        "gen_ai.tool.actor": str(data.get("actor", "agent")),
                    },
                ):
                    pass

    return collector.export_otlp_genai_json(service_name=service_name)


# Global collector instance
_global_collector = SpanCollector()


def get_collector() -> SpanCollector:
    return _global_collector


@contextmanager
def bench_span(name: str, **attributes: Any) -> Generator[Span, None, None]:
    """Convenience function using global collector."""
    with _global_collector.trace(name, **attributes) as span:
        yield span

