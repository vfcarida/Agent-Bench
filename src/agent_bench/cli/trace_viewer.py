"""Rich interactive trace viewer: inspect execution traces for a specific task or run."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


def view_traces(
    run_id: str,
    task_id: str | None = None,
    step: int | None = None,
    limit: int = 50,
) -> None:
    """Display traces for a run, optionally filtered by task_id and/or step."""
    runs_dir = Path("data/runs")
    traces_path = runs_dir / run_id / "traces.jsonl"

    if not traces_path.exists():
        # Try prefix match or direct run_id.jsonl
        matches = list(runs_dir.glob(f"{run_id}*/traces.jsonl"))
        if matches:
            traces_path = matches[0]
        else:
            direct_jsonl = list(runs_dir.glob(f"{run_id}*.jsonl"))
            if direct_jsonl:
                traces_path = direct_jsonl[0]
            else:
                console.print(f"[red]Traces not found for run: {run_id}[/red]")
                return

    traces = []
    with open(traces_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    # Filter by task_id if specified
    if task_id:
        filtered = []
        capture = False
        target_lower = task_id.lower()
        for t in traces:
            t_task = t.get("task_id") or t.get("data", {}).get("task_id")
            if t_task and str(t_task).lower() == target_lower:
                filtered.append(t)
                capture = True
            elif t.get("event_type") == "prompt_sent":
                msgs = t.get("data", {}).get("messages", [])
                capture = target_lower in json.dumps(msgs).lower()
                if capture:
                    filtered.append(t)
            elif capture:
                filtered.append(t)
                if t.get("event_type") == "judge_decision":
                    capture = False
        traces = filtered

    # Filter by step if specified
    if step is not None:
        traces = [t for t in traces if t.get("data", {}).get("step") == step]

    if not traces:
        console.print("[yellow]No matching traces found.[/yellow]")
        return

    traces = traces[:limit]

    # Header summary
    title = f"Execution Traces for Run {run_id[:8]}"
    if task_id:
        title += f" [dim](task: {task_id})[/dim]"
    if step is not None:
        title += f" [dim](step: {step})[/dim]"

    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"[dim]Displaying {len(traces)} events[/dim]\n")

    for idx, trace in enumerate(traces, 1):
        _render_trace_event(trace, idx)


def _render_trace_event(trace: dict[str, Any], event_index: int = 1) -> None:
    """Render a single trace event using rich formatted panels and tables."""
    event_type = trace.get("event_type", "unknown")
    timestamp = trace.get("timestamp", "")[:19]
    data = trace.get("data", {})

    if event_type == "prompt_sent":
        system_id = data.get("system_id", "")
        template = data.get("template", "default")
        msgs = data.get("messages", [])
        user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        content_text = Text()
        content_text.append(f"System: {system_id} | Template: {template}\n\n", style="bold cyan")
        content_text.append(f"User Prompt:\n{user_msg}", style="white")

        console.print(
            Panel(
                content_text,
                title=f"[cyan]#{event_index} PROMPT SENT[/cyan] [dim]({timestamp})[/dim]",
                border_style="cyan",
            )
        )

    elif event_type == "thinking_block":
        blocks = data.get("thinking_blocks", [])
        tokens_count = data.get("thinking_token_count", 0)
        ratio = data.get("thinking_ratio", 0.0)
        thought_body = "\n---\n".join(blocks) if blocks else "(no thinking text recorded)"

        text = Text()
        text.append(f"Thinking Tokens: {tokens_count} | Ratio: {ratio * 100:.1f}%\n\n", style="italic magenta")
        text.append(thought_body, style="dim white")

        console.print(
            Panel(
                text,
                title=f"[magenta]#{event_index} INTERNAL REASONING (<think>)[/magenta] [dim]({timestamp})[/dim]",
                border_style="magenta",
            )
        )

    elif event_type == "tool_call":
        tool_name = data.get("tool_name", "")
        arguments = data.get("arguments", {})
        step_idx = data.get("step")
        step_str = f" [dim](step {step_idx})[/dim]" if step_idx is not None else ""

        arg_json = json.dumps(arguments, indent=2)
        syntax = Syntax(arg_json, "json", theme="monokai", line_numbers=False)

        console.print(
            Panel(
                syntax,
                title=f"[yellow]#{event_index} TOOL CALL: [bold]{tool_name}[/bold]{step_str}[/yellow] [dim]({timestamp})[/dim]",
                border_style="yellow",
            )
        )

    elif event_type == "tool_output":
        tool_name = data.get("tool_name", "")
        output = data.get("output", "")
        success = data.get("success", True)
        step_idx = data.get("step")
        step_str = f" [dim](step {step_idx})[/dim]" if step_idx is not None else ""
        badge = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"

        out_str = json.dumps(output, indent=2) if isinstance(output, (dict, list)) else str(output)

        console.print(
            Panel(
                out_str,
                title=f"[yellow]#{event_index} OBSERVATION: [bold]{tool_name}[/bold] {badge}{step_str}[/yellow] [dim]({timestamp})[/dim]",
                border_style="yellow",
            )
        )

    elif event_type in ("model_call", "model_response"):
        content = data.get("content") or data.get("response", "")
        tokens_in = data.get("tokens_in", 0)
        tokens_out = data.get("tokens_out", 0)
        latency = data.get("latency_ms", 0.0)

        meta_line = f"Tokens In: {tokens_in} | Tokens Out: {tokens_out} | Latency: {latency:.1f}ms\n\n"
        text = Text()
        text.append(meta_line, style="bold green")
        text.append(content, style="white")

        console.print(
            Panel(
                text,
                title=f"[green]#{event_index} MODEL RESPONSE[/green] [dim]({timestamp})[/dim]",
                border_style="green",
            )
        )

    elif event_type == "judge_decision":
        verdict = data.get("verdict", False)
        score = data.get("score", 0.0)
        reasoning = data.get("reasoning", "")
        judge_id = data.get("judge_id", "evaluator")
        badge = "[bold green]PASS[/bold green]" if verdict else "[bold red]FAIL[/bold red]"

        body = Text()
        body.append(f"Outcome: {badge} | Score: {score:.3f} | Judge: {judge_id}\n\n", style="bold")
        body.append(f"Reasoning:\n{reasoning}\n", style="italic white")

        individual = data.get("metadata", {}).get("individual_verdicts") or data.get("individual_verdicts", [])
        if individual:
            table = Table(title="Sub-Judge Scores", show_edge=False, box=None)
            table.add_column("Judge", style="cyan")
            table.add_column("Passed", justify="center")
            table.add_column("Score", justify="right")
            for iv in individual:
                v_badge = "[green]✓[/green]" if iv.get("passed") else "[red]✗[/red]"
                table.add_row(iv.get("judge_id", ""), v_badge, f"{iv.get('score', 0):.2f}")
            console.print(
                Panel(
                    body,
                    title=f"[blue]#{event_index} EVALUATION VERDICT[/blue] [dim]({timestamp})[/dim]",
                    border_style="blue",
                )
            )
            console.print(table)
        else:
            console.print(
                Panel(
                    body,
                    title=f"[blue]#{event_index} EVALUATION VERDICT[/blue] [dim]({timestamp})[/dim]",
                    border_style="blue",
                )
            )

    else:
        console.print(
            Panel(
                str(data),
                title=f"[white]#{event_index} {event_type.upper()}[/white] [dim]({timestamp})[/dim]",
                border_style="white",
            )
        )
