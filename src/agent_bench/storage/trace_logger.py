"""Execution trace logger for recording agent run traces, latency, token metrics, and costs."""

import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.settings import settings
from agent_bench.storage.jsonl import save_traces_jsonl


class ExecutionTraceLogger:
    """Logger that collects execution trace events, computes usage metrics, and exports JSONL/Parquet."""

    def __init__(
        self,
        run_id: str,
        task_id: str,
        system_id: str,
        cost_per_1k_input: float | None = None,
        cost_per_1k_output: float | None = None,
    ) -> None:
        """Initialize ExecutionTraceLogger.

        Args:
            run_id: Parent benchmark run ID.
            task_id: Identifier of the task being executed.
            system_id: Identifier of the agent system running.
            cost_per_1k_input: Pricing per 1,000 prompt tokens (USD).
            cost_per_1k_output: Pricing per 1,000 completion tokens (USD).
        """
        self.run_id = run_id
        self.task_id = task_id
        self.system_id = system_id
        self.cost_per_1k_input = (
            cost_per_1k_input
            if cost_per_1k_input is not None
            else settings.cost_per_1k_input_tokens
        )
        self.cost_per_1k_output = (
            cost_per_1k_output
            if cost_per_1k_output is not None
            else settings.cost_per_1k_output_tokens
        )

        self._traces: list[TraceEvent] = []
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._total_latency_ms: float = 0.0

    @property
    def traces(self) -> list[TraceEvent]:
        """Returns collected TraceEvents."""
        return list(self._traces)

    @property
    def tokens_in(self) -> int:
        """Total prompt tokens accumulated."""
        return self._tokens_in

    @property
    def tokens_out(self) -> int:
        """Total completion tokens accumulated."""
        return self._tokens_out

    @property
    def total_tokens(self) -> int:
        """Total tokens accumulated."""
        return self._tokens_in + self._tokens_out

    @property
    def total_latency_ms(self) -> float:
        """Total latency accumulated in milliseconds."""
        return self._total_latency_ms

    @property
    def total_cost_usd(self) -> float:
        """Total calculated financial cost in USD."""
        cost_in = (self._tokens_in / 1000.0) * self.cost_per_1k_input
        cost_out = (self._tokens_out / 1000.0) * self.cost_per_1k_output
        return cost_in + cost_out

    def log_event(
        self,
        event_type: TraceEventType,
        data: dict[str, Any],
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Logs a generic trace event and appends it to the execution history.

        Args:
            event_type: Type of trace event.
            data: Structured payload data for the event.
            parent_id: Optional parent event UUID.

        Returns:
            Created TraceEvent instance.
        """
        event = TraceEvent(event_type=event_type, data=data, parent_id=parent_id)
        self._traces.append(event)
        return event

    def log_model_call(
        self,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        model_id: str = "",
        thinking_content: str | None = None,
    ) -> TraceEvent:
        """Logs an LLM model invocation, tracking token usage, latency, and cost.

        Args:
            tokens_in: Number of input prompt tokens.
            tokens_out: Number of generated completion tokens.
            latency_ms: Request latency in milliseconds.
            model_id: Model identifier invoked.
            thinking_content: Optional raw internal reasoning string.

        Returns:
            Created TraceEvent instance for MODEL_RESPONSE.
        """
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out
        self._total_latency_ms += latency_ms

        call_cost = ((tokens_in / 1000.0) * self.cost_per_1k_input) + (
            (tokens_out / 1000.0) * self.cost_per_1k_output
        )

        data = {
            "model_id": model_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "latency_ms": latency_ms,
            "cost_usd": call_cost,
            "thinking_content": thinking_content,
        }
        return self.log_event(TraceEventType.MODEL_RESPONSE, data)

    def get_summary(self) -> dict[str, Any]:
        """Returns structured summary of execution metrics.

        Returns:
            Dictionary containing tokens, latency, cost, and trace count.
        """
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "system_id": self.system_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "trace_count": len(self._traces),
        }

    def export_jsonl(self, output_path: Path) -> Path:
        """Exports traces to a JSONL file atomically.

        Args:
            output_path: Target file path.

        Returns:
            Saved file path.
        """
        return save_traces_jsonl(self._traces, output_path)

    def export_parquet(self, output_path: Path) -> Path:
        """Exports execution trace events and metrics to a Parquet file atomically.

        Args:
            output_path: Target Parquet file path.

        Returns:
            Saved Parquet file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()

        records = []
        for trace in self._traces:
            records.append({
                "run_id": summary["run_id"],
                "task_id": summary["task_id"],
                "system_id": summary["system_id"],
                "event_id": trace.event_id,
                "event_type": trace.event_type.value,
                "timestamp": trace.timestamp.isoformat(),
                "latency_ms": float(trace.data.get("latency_ms", 0.0)),
                "tokens_in": int(trace.data.get("tokens_in", 0)),
                "tokens_out": int(trace.data.get("tokens_out", 0)),
                "cost_usd": float(trace.data.get("cost_usd", 0.0)),
            })

        if not records:
            records.append({
                "run_id": summary["run_id"],
                "task_id": summary["task_id"],
                "system_id": summary["system_id"],
                "event_id": "summary",
                "event_type": "summary",
                "timestamp": "",
                "latency_ms": float(summary["latency_ms"]),
                "tokens_in": int(summary["tokens_in"]),
                "tokens_out": int(summary["tokens_out"]),
                "cost_usd": float(summary["total_cost_usd"]),
            })

        table = pa.Table.from_pylist(records)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            pq.write_table(table, temp_path)  # type: ignore[no-untyped-call]
            os.replace(temp_path, output_path)
        except Exception:
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise
        return output_path
