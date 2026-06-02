"""HTML report generator with scorecard tables and leaderboard."""

import json
from pathlib import Path
from typing import Any

from jinja2 import Template

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Benchmark Report - {{ run_id[:8] }}</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --card: #16213e;
            --accent: #0f3460;
            --highlight: #e94560;
            --text: #eaeaea;
            --muted: #a0a0a0;
            --green: #4caf50;
            --yellow: #ff9800;
            --red: #f44336;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: var(--highlight); margin-bottom: 0.5rem; font-size: 1.8rem; }
        h2 { color: var(--text); margin: 2rem 0 1rem; border-bottom: 1px solid var(--accent); padding-bottom: 0.5rem; }
        h3 { color: var(--muted); margin: 1.5rem 0 0.5rem; }
        .meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }
        .meta span { margin-right: 2rem; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
        .card { background: var(--card); padding: 1.2rem; border-radius: 8px; text-align: center; }
        .card .value { font-size: 2rem; font-weight: bold; color: var(--highlight); }
        .card .label { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; background: var(--card); border-radius: 8px; overflow: hidden; }
        th, td { padding: 0.7rem 1rem; text-align: left; border-bottom: 1px solid var(--accent); }
        th { background: var(--accent); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }
        td { font-size: 0.9rem; }
        tr:last-child td { border-bottom: none; }
        .score-high { color: var(--green); font-weight: bold; }
        .score-mid { color: var(--yellow); font-weight: bold; }
        .score-low { color: var(--red); font-weight: bold; }
        .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
        .badge-pass { background: var(--green); color: #fff; }
        .badge-fail { background: var(--red); color: #fff; }
        .leaderboard { margin: 1.5rem 0; }
        .leaderboard .rank { font-size: 1.4rem; font-weight: bold; color: var(--highlight); width: 3rem; text-align: center; }
        .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--accent); color: var(--muted); font-size: 0.8rem; }
        .bar { height: 8px; border-radius: 4px; background: var(--accent); margin-top: 0.3rem; }
        .bar-fill { height: 100%; border-radius: 4px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Benchmark Report</h1>
    <div class="meta">
        <span>Run: {{ run_id[:8] }}</span>
        <span>Suite: {{ suite_id }}</span>
        <span>Date: {{ date }}</span>
        <span>Version: {{ benchmark_version }}</span>
        <span>Config: {{ config_hash }}</span>
    </div>

    <div class="cards">
        <div class="card">
            <div class="value">{{ tasks_total }}</div>
            <div class="label">Total Tasks</div>
        </div>
        <div class="card">
            <div class="value score-high">{{ tasks_passed }}</div>
            <div class="label">Passed</div>
        </div>
        <div class="card">
            <div class="value score-low">{{ tasks_failed }}</div>
            <div class="label">Failed</div>
        </div>
        <div class="card">
            <div class="value">{{ pass_rate }}%</div>
            <div class="label">Pass Rate</div>
        </div>
        <div class="card">
            <div class="value">{{ duration_ms }}</div>
            <div class="label">Duration (ms)</div>
        </div>
        <div class="card">
            <div class="value">{{ seed }}</div>
            <div class="label">Seed</div>
        </div>
    </div>

    {% if scorecards %}
    <h2>System Scorecards</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>System</th>
                <th>Domain</th>
                <th>Functional</th>
                <th>Risk</th>
                <th>Cost</th>
                <th>Latency</th>
                <th>Reliability</th>
                <th>Global</th>
            </tr>
        </thead>
        <tbody>
        {% for sc in scorecards_sorted %}
            <tr>
                <td class="rank">{{ loop.index }}</td>
                <td>{{ sc.system_id }}</td>
                <td>{{ sc.domain }}</td>
                <td class="{{ score_class(sc.functional_score) }}">{{ "%.3f"|format(sc.functional_score) }}</td>
                <td class="{{ score_class(sc.risk_score) }}">{{ "%.3f"|format(sc.risk_score) }}</td>
                <td class="{{ score_class(sc.cost_score) }}">{{ "%.3f"|format(sc.cost_score) }}</td>
                <td class="{{ score_class(sc.latency_score) }}">{{ "%.3f"|format(sc.latency_score) }}</td>
                <td class="{{ score_class(sc.reliability_score) }}">{{ "%.3f"|format(sc.reliability_score) }}</td>
                <td class="{{ score_class(sc.global_score) }}"><strong>{{ "%.3f"|format(sc.global_score) }}</strong></td>
            </tr>
        {% endfor %}
        </tbody>
    </table>

    <h3>Score Distribution</h3>
    {% for sc in scorecards_sorted %}
    <div style="margin: 0.5rem 0;">
        <span style="display:inline-block;width:200px;">{{ sc.system_id[:20] }}</span>
        <div class="bar" style="display:inline-block;width:60%;vertical-align:middle;">
            <div class="bar-fill" style="width:{{ (sc.global_score * 100)|int }}%;background:{% if sc.global_score >= 0.7 %}var(--green){% elif sc.global_score >= 0.4 %}var(--yellow){% else %}var(--red){% endif %};"></div>
        </div>
        <span style="margin-left:0.5rem;">{{ "%.1f"|format(sc.global_score * 100) }}%</span>
    </div>
    {% endfor %}
    {% endif %}

    {% if pass_k_results %}
    <h2>Pass@K Results</h2>
    <table>
        <thead><tr><th>System</th><th>Domain</th><th>Pass@1</th><th>Pass@3</th><th>Pass@5</th></tr></thead>
        <tbody>
        {% for pk in pass_k_results %}
            <tr>
                <td>{{ pk.system_id }}</td>
                <td>{{ pk.domain }}</td>
                <td class="{{ score_class(pk.pass_1) }}">{{ "%.3f"|format(pk.pass_1) }}</td>
                <td class="{{ score_class(pk.pass_3) }}">{{ "%.3f"|format(pk.pass_3) }}</td>
                <td class="{{ score_class(pk.pass_5) }}">{{ "%.3f"|format(pk.pass_5) }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% endif %}

    <div class="footer">
        Generated by agent-bench v{{ benchmark_version }} | Config hash: {{ config_hash }}
    </div>
</div>
</body>
</html>"""


def score_class(value: float) -> str:
    if value >= 0.7:
        return "score-high"
    elif value >= 0.4:
        return "score-mid"
    return "score-low"


def generate_html_report(run_id: str, output_dir: Path) -> Path:
    """Generate a full HTML report with scorecards and visualizations."""
    runs_dir = Path("data/runs")
    artifact_path = runs_dir / f"{run_id}.json"

    if not artifact_path.exists():
        matches = list(runs_dir.glob(f"{run_id}*.json"))
        if matches:
            artifact_path = matches[0]
        else:
            raise FileNotFoundError(f"Run artifact not found: {run_id}")

    with open(artifact_path) as f:
        data = json.load(f)

    tasks_total = data.get("tasks_total", 0)
    tasks_passed = data.get("tasks_passed", 0)
    pass_rate = round(tasks_passed / tasks_total * 100, 1) if tasks_total > 0 else 0

    scorecards = data.get("scorecards", [])
    scorecards_sorted = sorted(scorecards, key=lambda s: s.get("global_score", 0), reverse=True)

    context = {
        "run_id": data["run_id"],
        "suite_id": data.get("suite_id", "N/A"),
        "system_id": data.get("system_id", "N/A"),
        "date": data.get("started_at", "")[:19],
        "duration_ms": data.get("duration_ms", "N/A"),
        "config_hash": data.get("config_hash", ""),
        "benchmark_version": data.get("benchmark_version", ""),
        "tasks_total": tasks_total,
        "tasks_passed": tasks_passed,
        "tasks_failed": data.get("tasks_failed", 0),
        "pass_rate": pass_rate,
        "seed": data.get("seed", "N/A"),
        "scorecards": scorecards,
        "scorecards_sorted": scorecards_sorted,
        "pass_k_results": data.get("pass_k_results", []),
        "score_class": score_class,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    template = Template(_HTML_TEMPLATE)
    content = template.render(**context)
    out_path = output_dir / f"{run_id[:8]}_report.html"
    out_path.write_text(content)
    return out_path
