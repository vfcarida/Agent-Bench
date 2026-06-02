"""Trace viewer: inspect execution traces for a specific task or run."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


def view_traces(run_id: str, task_id: str | None = None, limit: int = 50) -> None:
    """Display traces for a run, optionally filtered by task_id."""
    runs_dir = Path("data/runs")
    traces_path = runs_dir / run_id / "traces.jsonl"

    if not traces_path.exists():
        # Try prefix match
        matches = list(runs_dir.glob(f"{run_id}*/traces.jsonl"))
        if matches:
            traces_path = matches[0]
        else:
            console.print(f"[red]Traces not found for run: {run_id}[/red]")
            return

    traces = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    # Filter by task_id if specified
    if task_id:
        filtered = []
        capture = False
        for t in traces:
            if t["event_type"] == "prompt_sent":
                msgs = t.get("data", {}).get("messages", [])
                # Check if this trace group matches the task
                capture = any(task_id.lower() in json.dumps(msgs).lower() for _ in [1])
            if capture:
                filtered.append(t)
            if t["event_type"] == "judge_decision" and capture:
                capture = False
        traces = filtered

    if not traces:
        console.print("[yellow]No matching traces found.[/yellow]")
        return

    traces = traces[:limit]

    # Display
    console.print(f"\n[bold]Traces for run {run_id[:8]}[/bold]", end="")
    if task_id:
        console.print(f" [dim](task: {task_id})[/dim]")
    else:
        console.print()
    console.print(f"[dim]Showing {len(traces)} events[/dim]\n")

    for trace in traces:
        _render_trace_event(trace)


def _render_trace_event(trace: dict[str, Any]) -> None:
    """Render a single trace event."""
    event_type = trace["event_type"]
    timestamp = trace.get("timestamp", "")[:19]
    data = trace.get("data", {})

    color_map = {
        "prompt_sent": "cyan",
        "model_response": "green",
        "tool_call": "yellow",
        "tool_output": "yellow",
        "retrieval_query": "magenta",
        "retrieval_result": "magenta",
        "judge_decision": "blue",
        "metric_computed": "white",
        "error": "red",
        "system_event": "dim",
    }
    color = color_map.get(event_type, "white")

    if event_type == "prompt_sent":
        system_id = data.get("system_id", "")
        msgs = data.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        console.print(f"  [{color}]PROMPT[/{color}] [{timestamp}] system={system_id}")
        if user_msg:
            console.print(f"    [dim]user:[/dim] {user_msg[:120]}")

    elif event_type == "model_response":
        content = data.get("content", "")[:200]
        console.print(f"  [{color}]RESPONSE[/{color}] [{timestamp}]")
        console.print(f"    {content}")

    elif event_type == "tool_call":
        tool = data.get("tool_name", "")
        console.print(f"  [{color}]TOOL_CALL[/{color}] [{timestamp}] {tool}")

    elif event_type == "retrieval_result":
        docs = data.get("documents", [])
        console.print(f"  [{color}]RETRIEVAL[/{color}] [{timestamp}] {len(docs)} docs")

    elif event_type == "judge_decision":
        verdict = data.get("verdict", "")
        score = data.get("score", 0)
        reasoning = data.get("reasoning", "")[:150]
        badge = "[green]PASS[/green]" if verdict else "[red]FAIL[/red]"
        console.print(f"  [{color}]JUDGE[/{color}] [{timestamp}] {badge} score={score:.2f}")
        console.print(f"    [dim]{reasoning}[/dim]")
        individual = data.get("individual_verdicts", [])
        for v in individual:
            v_badge = "[green]OK[/green]" if v.get("passed") else "[red]X[/red]"
            console.print(f"      {v_badge} {v.get('judge_id', '')}: {v.get('score', 0):.2f}")

    elif event_type == "error":
        console.print(f"  [{color}]ERROR[/{color}] [{timestamp}] {data}")

    else:
        console.print(f"  [{color}]{event_type.upper()}[/{color}] [{timestamp}]")

    console.print()
