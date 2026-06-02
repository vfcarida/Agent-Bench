"""Local leaderboard: persists and ranks system performance across runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEADERBOARD_PATH = Path("data/reports/leaderboard.json")


def update_leaderboard(
    run_id: str,
    suite_id: str,
    scorecards: list[dict[str, Any]],
    leaderboard_path: Path | None = None,
) -> Path:
    """Update the leaderboard with new run results."""
    path = leaderboard_path or LEADERBOARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    entries = []
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            entries = data.get("entries", [])

    # Add new entries
    for sc in scorecards:
        entries.append({
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
        })

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
    """Render leaderboard as markdown."""
    entries = get_leaderboard(domain, top_n, leaderboard_path)
    if not entries:
        return "# Leaderboard\n\nNo entries yet.\n"

    lines = ["# Leaderboard\n"]
    if domain:
        lines.append(f"**Domain:** {domain}\n")
    lines.append(f"**Top {min(top_n, len(entries))} systems**\n")
    lines.append("| # | System | Domain | Global | Func | Risk | Cost | Lat | Rel | Run |")
    lines.append("|---|--------|--------|--------|------|------|------|-----|-----|-----|")

    for i, e in enumerate(entries, 1):
        lines.append(
            f"| {i} | {e['system_id'][:20]} | {e['domain']} | "
            f"**{e['global_score']:.3f}** | {e['functional_score']:.3f} | "
            f"{e['risk_score']:.3f} | {e['cost_score']:.3f} | "
            f"{e['latency_score']:.3f} | {e['reliability_score']:.3f} | "
            f"{e['run_id'][:8]} |"
        )

    return "\n".join(lines) + "\n"


def render_leaderboard_html(
    domain: str | None = None,
    top_n: int = 20,
    leaderboard_path: Path | None = None,
) -> str:
    """Render leaderboard as HTML snippet."""
    entries = get_leaderboard(domain, top_n, leaderboard_path)
    rows = ""
    for i, e in enumerate(entries, 1):
        medal = {1: "&#x1F947;", 2: "&#x1F948;", 3: "&#x1F949;"}.get(i, str(i))
        rows += f"""<tr>
            <td>{medal}</td>
            <td>{e['system_id']}</td>
            <td>{e['domain']}</td>
            <td><strong>{e['global_score']:.3f}</strong></td>
            <td>{e['functional_score']:.3f}</td>
            <td>{e['risk_score']:.3f}</td>
            <td>{e['run_id'][:8]}</td>
        </tr>"""

    return f"""<table>
    <thead><tr><th>#</th><th>System</th><th>Domain</th><th>Global</th><th>Func</th><th>Risk</th><th>Run</th></tr></thead>
    <tbody>{rows}</tbody>
    </table>"""
