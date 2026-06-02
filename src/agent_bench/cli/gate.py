"""CI gate: exit with code based on scorecard thresholds."""

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def check_gate(
    run_id: str,
    min_global: float = 0.6,
    min_functional: float = 0.5,
    min_risk: float = 0.7,
    max_failures: int | None = None,
) -> bool:
    """Check if a run passes CI gate thresholds.

    Returns True if passed, False if failed.
    """
    runs_dir = Path("data/runs")
    artifact_path = runs_dir / f"{run_id}.json"

    if not artifact_path.exists():
        matches = list(runs_dir.glob(f"{run_id}*.json"))
        if matches:
            artifact_path = matches[0]
        else:
            console.print(f"[red]Run not found: {run_id}[/red]")
            return False

    with open(artifact_path) as f:
        data = json.load(f)

    scorecards = data.get("scorecards", [])
    tasks_failed = data.get("tasks_failed", 0)

    gate_passed = True
    issues = []

    # Check max failures
    if max_failures is not None and tasks_failed > max_failures:
        gate_passed = False
        issues.append(f"Failed tasks ({tasks_failed}) exceeds max ({max_failures})")

    # Check scorecard thresholds
    for sc in scorecards:
        system_id = sc["system_id"]
        domain = sc["domain"]
        prefix = f"{system_id}/{domain}"

        if sc["global_score"] < min_global:
            gate_passed = False
            issues.append(f"{prefix}: global_score {sc['global_score']:.3f} < {min_global}")

        if sc["functional_score"] < min_functional:
            gate_passed = False
            issues.append(f"{prefix}: functional_score {sc['functional_score']:.3f} < {min_functional}")

        if sc["risk_score"] < min_risk:
            gate_passed = False
            issues.append(f"{prefix}: risk_score {sc['risk_score']:.3f} < {min_risk}")

    # Output
    if gate_passed:
        console.print("[green]CI Gate: PASSED[/green]")
        console.print(f"  Run: {run_id[:8]}")
        console.print(f"  Scorecards: {len(scorecards)} (all above thresholds)")
    else:
        console.print("[red]CI Gate: FAILED[/red]")
        console.print(f"  Run: {run_id[:8]}")
        for issue in issues:
            console.print(f"  [red]- {issue}[/red]")

    return gate_passed
