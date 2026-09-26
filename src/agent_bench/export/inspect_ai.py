"""Inspect AI dataset and evaluation log export bridge.

Provides interoperability between Agent-Bench and the UK/US AI Safety Institute's
Inspect AI evaluation framework (https://inspect.ai-safety-institute.org.uk/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_bench.core.artifacts import RunArtifact, TraceEventType
from agent_bench.core.scenarios import Task


def task_to_inspect_sample(task: Task) -> dict[str, Any]:
    """Convert an Agent-Bench Task or EvalCase into an Inspect AI Sample dict."""
    # Convert input messages to Inspect AI chat messages format
    inspect_messages = []
    for msg in task.input_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        inspect_messages.append({"role": role, "content": content})

    # Prepare target from gold references or expected state
    target: Any = task.gold_references if task.gold_references else []
    if not target and task.expected_final_state:
        target = [json.dumps(task.expected_final_state)]

    metadata: dict[str, Any] = {
        "domain": task.domain,
        "task_id": task.task_id,
        "name": task.name,
        "description": task.description,
        "severity": getattr(task.severity, "value", str(task.severity)),
        "business_criticality": getattr(task.business_criticality, "value", str(task.business_criticality)),
        "expected_refusal_mode": getattr(task.expected_refusal_mode, "value", str(task.expected_refusal_mode)),
        "allowed_tools": task.allowed_tools,
        "expected_final_state": task.expected_final_state,
        "evidence_strings": task.evidence_strings,
        "tags": task.tags,
        "task_version": task.task_version,
    }
    if task.metadata:
        metadata["bench_metadata"] = task.metadata

    # Inspect AI Sample specification
    sample: dict[str, Any] = {
        "id": task.task_id,
        "input": inspect_messages,
        "target": target,
        "metadata": metadata,
    }

    if task.allowed_tools:
        sample["sandbox"] = {
            "type": "agent_bench_environment",
            "tools": task.allowed_tools,
        }

    return sample


def export_tasks_to_inspect_dataset(
    tasks: list[Task],
    output_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Export Agent-Bench tasks to Inspect AI JSONL dataset format.

    Args:
        tasks: List of Task instances to convert.
        output_path: Optional path to write jsonl dataset file.

    Returns:
        List of Inspect AI Sample dictionaries.
    """
    samples = [task_to_inspect_sample(t) for t in tasks]

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return samples


def run_artifact_to_inspect_log(artifact: RunArtifact) -> dict[str, Any]:
    """Convert an Agent-Bench RunArtifact into an Inspect AI evaluation log structure.

    Args:
        artifact: The RunArtifact containing metrics, traces, and execution summary.

    Returns:
        Inspect AI compatible log dictionary.
    """
    # Group trace events by task_id if present
    task_events: dict[str, list[dict[str, Any]]] = {}
    for tr in artifact.traces:
        t_id = tr.data.get("task_id", "general")
        if t_id not in task_events:
            task_events[t_id] = []
        task_events[t_id].append({
            "event_type": tr.event_type.value if hasattr(tr.event_type, "value") else str(tr.event_type),
            "timestamp": tr.timestamp.isoformat(),
            "data": tr.data,
            "event_id": tr.event_id,
            "parent_id": tr.parent_id,
        })

    # Build inspect samples evaluation results
    samples_results = []
    for task_id, events in task_events.items():
        user_msgs = [e["data"].get("content") for e in events if e["event_type"] == TraceEventType.USER_MESSAGE.value]
        model_msgs = [e["data"].get("response") for e in events if e["event_type"] == TraceEventType.MODEL_RESPONSE.value]
        tool_calls = [e["data"] for e in events if e["event_type"] == TraceEventType.TOOL_CALL.value]

        samples_results.append({
            "id": task_id,
            "input": user_msgs[0] if user_msgs else "",
            "output": model_msgs[-1] if model_msgs else "",
            "tool_calls": tool_calls,
            "events_count": len(events),
        })

    return {
        "eval": {
            "benchmark": "Agent-Bench",
            "run_id": artifact.run_id,
            "suite_id": artifact.suite_id,
            "system_id": artifact.system_id,
            "model_id": artifact.model_id,
            "config_hash": artifact.config_hash,
            "started_at": artifact.started_at.isoformat(),
            "finished_at": artifact.finished_at.isoformat() if artifact.finished_at else None,
            "duration_ms": artifact.duration_ms,
        },
        "results": {
            "tasks_total": artifact.tasks_total,
            "tasks_passed": artifact.tasks_passed,
            "tasks_failed": artifact.tasks_failed,
            "metrics": artifact.metrics,
        },
        "samples": samples_results,
    }
