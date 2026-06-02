"""Compare multiple benchmark runs and systems."""

import json
from pathlib import Path
from typing import Any

from agent_bench.storage.parquet import save_comparison_parquet


def compare(run_ids: list[str], output_path: Path) -> None:
    """Generate a comparison report across runs."""
    runs_dir = Path("data/runs")
    runs_data = []

    for run_id in run_ids:
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            matches = list(runs_dir.glob(f"{run_id}*.json"))
            if matches:
                path = matches[0]
            else:
                continue
        with open(path) as f:
            runs_data.append(json.load(f))

    if len(runs_data) < 2:
        raise ValueError("Need at least 2 valid runs to compare.")

    lines = ["# Run Comparison\n"]
    lines.append("| Field | " + " | ".join(r["run_id"][:8] for r in runs_data) + " |")
    lines.append("|-------|" + "|".join(["--------"] * len(runs_data)) + "|")

    for field in ["suite_id", "system_id", "tasks_passed", "tasks_failed", "duration_ms", "seed"]:
        values = [str(r.get(field, "N/A")) for r in runs_data]
        lines.append(f"| {field} | " + " | ".join(values) + " |")

    # Scorecard comparison
    lines.append("\n## Scorecards\n")
    for run_data in runs_data:
        scorecards = run_data.get("scorecards", [])
        if scorecards:
            lines.append(f"### Run {run_data['run_id'][:8]}\n")
            lines.append("| System | Domain | Functional | Risk | Cost | Latency | Reliability | Global |")
            lines.append("|--------|--------|-----------|------|------|---------|-------------|--------|")
            for sc in scorecards:
                lines.append(
                    f"| {sc['system_id'][:20]} | {sc['domain']} | "
                    f"{sc['functional_score']:.2f} | {sc['risk_score']:.2f} | "
                    f"{sc['cost_score']:.2f} | {sc['latency_score']:.2f} | "
                    f"{sc['reliability_score']:.2f} | {sc['global_score']:.2f} |"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    # Also save CSV
    csv_path = output_path.with_suffix(".csv")
    _save_comparison_csv(runs_data, csv_path)


def _save_comparison_csv(runs_data: list[dict[str, Any]], output_path: Path) -> None:
    """Save comparison as CSV."""
    lines = ["run_id,system_id,suite_id,tasks_total,tasks_passed,tasks_failed,pass_rate,duration_ms"]
    for r in runs_data:
        total = r.get("tasks_total", 0)
        passed = r.get("tasks_passed", 0)
        rate = f"{passed/total*100:.1f}" if total > 0 else "0"
        lines.append(
            f"{r['run_id']},{r.get('system_id','')},{r.get('suite_id','')},{total},"
            f"{passed},{r.get('tasks_failed',0)},{rate},{r.get('duration_ms','')}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def compare_systems(
    scorecards: list[dict[str, Any]], output_path: Path
) -> Path:
    """Generate a system comparison from scorecards."""
    lines = ["# System Comparison\n"]
    lines.append("| System | Domain | Functional | Risk | Cost | Latency | Reliability | Global |")
    lines.append("|--------|--------|-----------|------|------|---------|-------------|--------|")

    sorted_cards = sorted(scorecards, key=lambda s: s.get("global_score", 0), reverse=True)
    for sc in sorted_cards:
        lines.append(
            f"| {sc['system_id']} | {sc['domain']} | "
            f"{sc['functional_score']:.3f} | {sc['risk_score']:.3f} | "
            f"{sc['cost_score']:.3f} | {sc['latency_score']:.3f} | "
            f"{sc['reliability_score']:.3f} | **{sc['global_score']:.3f}** |"
        )

    lines.append(f"\n**Winner:** {sorted_cards[0]['system_id']} (global={sorted_cards[0]['global_score']:.3f})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    # Also save Parquet
    parquet_path = output_path.with_suffix(".parquet")
    save_comparison_parquet(
        [sc["system_id"] for sc in sorted_cards],
        sorted_cards,
        parquet_path,
    )

    return output_path
