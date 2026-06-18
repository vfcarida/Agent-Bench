"""Local leaderboard: persists and ranks system performance across runs.

Extended with Athena lineage tracking to display model genealogy
(merge method, parent models) alongside benchmark scores.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_bench.models.lineage import AthenaLineage, format_lineage_compact, format_lineage_display


LEADERBOARD_PATH = Path("data/reports/leaderboard.json")


def update_leaderboard(
    run_id: str,
    suite_id: str,
    scorecards: list[dict[str, Any]],
    leaderboard_path: Path | None = None,
    athena_lineage: AthenaLineage | None = None,
) -> Path:
    """Update the leaderboard with new run results.

    Args:
        run_id: Unique run identifier.
        suite_id: Suite that was evaluated.
        scorecards: List of scorecard dicts from the run.
        leaderboard_path: Custom path for the leaderboard file.
        athena_lineage: Optional Athena model lineage metadata.
    """
    path = leaderboard_path or LEADERBOARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    entries = []
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            entries = data.get("entries", [])

    # Build lineage fields
    lineage_data: dict[str, Any] = {}
    if athena_lineage is not None:
        lineage_data = {
            "lineage": format_lineage_display(athena_lineage),
            "lineage_compact": format_lineage_compact(athena_lineage),
            "merge_method": athena_lineage.merge_method,
            "parents": athena_lineage.parents,
            "parent_weights": athena_lineage.parent_weights,
            "athena_phase": athena_lineage.athena_phase,
            "base_model": athena_lineage.base_model,
        }

    # Add new entries
    for sc in scorecards:
        entry: dict[str, Any] = {
            "run_id": run_id,
            "suite_id": suite_id,
            "system_id": sc["system_id"],
            "domain": sc["domain"],
            "global_score": sc["global_score"],
            "functional_score": sc["functional_score"],
            "risk_score": sc["risk_score"],
            "cost_score": sc["cost_score"],
            "latency_score": sc["latency_score"],
            "reliability_score": sc["reliability_score"],
            "weighting_profile": sc.get("weighting_profile", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Merge lineage data if available
        entry.update(lineage_data)
        entries.append(entry)

    # Sort by global score descending
    entries.sort(key=lambda e: e["global_score"], reverse=True)

    leaderboard = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "entries": entries,
    }

    path.write_text(json.dumps(leaderboard, indent=2))
    return path


def get_leaderboard(
    domain: str | None = None,
    top_n: int = 20,
    leaderboard_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Get top N entries from leaderboard, optionally filtered by domain."""
    path = leaderboard_path or LEADERBOARD_PATH
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if domain:
        entries = [e for e in entries if e["domain"] == domain]

    return entries[:top_n]  # type: ignore[no-any-return]


def render_leaderboard_markdown(
    domain: str | None = None,
    top_n: int = 20,
    leaderboard_path: Path | None = None,
) -> str:
    """Render leaderboard as markdown with optional lineage column."""
    entries = get_leaderboard(domain, top_n, leaderboard_path)
    if not entries:
        return "# Leaderboard\n\nNo entries yet.\n"

    # Check if any entries have lineage data
    has_lineage = any(e.get("lineage") for e in entries)

    lines = ["# Leaderboard\n"]
    if domain:
        lines.append(f"**Domain:** {domain}\n")
    lines.append(f"**Top {min(top_n, len(entries))} systems**\n")

    if has_lineage:
        lines.append("| # | System | Domain | Global | Func | Risk | Cost | Lat | Rel | Lineage | Run |")
        lines.append("|---|--------|--------|--------|------|------|------|-----|-----|---------|-----|")
    else:
        lines.append("| # | System | Domain | Global | Func | Risk | Cost | Lat | Rel | Run |")
        lines.append("|---|--------|--------|--------|------|------|------|-----|-----|-----|")

    for i, e in enumerate(entries, 1):
        base_cols = (
            f"| {i} | {e['system_id'][:20]} | {e['domain']} | "
            f"**{e['global_score']:.3f}** | {e['functional_score']:.3f} | "
            f"{e['risk_score']:.3f} | {e['cost_score']:.3f} | "
            f"{e['latency_score']:.3f} | {e['reliability_score']:.3f}"
        )
        if has_lineage:
            lineage_str = e.get("lineage_compact", "—")
            lines.append(f"{base_cols} | {lineage_str} | {e['run_id'][:8]} |")
        else:
            lines.append(f"{base_cols} | {e['run_id'][:8]} |")

    return "\n".join(lines) + "\n"


def render_leaderboard_html(
    domain: str | None = None,
    top_n: int = 20,
    leaderboard_path: Path | None = None,
) -> str:
    """Render leaderboard as HTML snippet with lineage tooltips."""
    entries = get_leaderboard(domain, top_n, leaderboard_path)
    has_lineage = any(e.get("lineage") for e in entries)

    rows = ""
    for i, e in enumerate(entries, 1):
        medal = {1: "&#x1F947;", 2: "&#x1F948;", 3: "&#x1F949;"}.get(i, str(i))

        lineage_cell = ""
        if has_lineage:
            lineage_full = e.get("lineage", "—")
            lineage_short = e.get("lineage_compact", "—")
            lineage_cell = f'<td title="{lineage_full}">{lineage_short}</td>'

        rows += f"""<tr>
            <td>{medal}</td>
            <td>{e['system_id']}</td>
            <td>{e['domain']}</td>
            <td><strong>{e['global_score']:.3f}</strong></td>
            <td>{e['functional_score']:.3f}</td>
            <td>{e['risk_score']:.3f}</td>
            {lineage_cell}
            <td>{e['run_id'][:8]}</td>
        </tr>"""

    lineage_header = "<th>Lineage</th>" if has_lineage else ""

    return f"""<table>
    <thead><tr><th>#</th><th>System</th><th>Domain</th><th>Global</th><th>Func</th><th>Risk</th>{lineage_header}<th>Run</th></tr></thead>
    <tbody>{rows}</tbody>
    </table>"""
