"""Unit tests for bench leaderboard CLI command."""

import json
from pathlib import Path

from click.testing import CliRunner

from agent_bench.cli.main import cli


def test_leaderboard_empty(tmp_path: Path):
    runner = CliRunner()
    fake_path = tmp_path / "nonexistent.json"
    result = runner.invoke(cli, ["leaderboard", "--leaderboard-path", str(fake_path)])
    assert result.exit_code == 0
    assert "No leaderboard entries found" in result.output


def test_leaderboard_with_entries(tmp_path: Path):
    runner = CliRunner(env={"COLUMNS": "180"})
    lb_path = tmp_path / "leaderboard.json"
    data = {
        "last_updated": "2026-09-25T00:00:00Z",
        "total_entries": 2,
        "entries": [
            {
                "run_id": "run_alpha_12345",
                "suite_id": "pix_suite",
                "system_id": "gpt4o_agent",
                "domain": "pix_assist",
                "global_score": 0.925,
                "functional_score": 0.950,
                "risk_score": 0.900,
                "cost_score": 0.850,
                "latency_score": 0.880,
                "reliability_score": 0.930,
            },
            {
                "run_id": "run_beta_67890",
                "suite_id": "cyber_suite",
                "system_id": "claude_agent",
                "domain": "cyber_sandbox",
                "global_score": 0.880,
                "functional_score": 0.910,
                "risk_score": 0.850,
                "cost_score": 0.800,
                "latency_score": 0.840,
                "reliability_score": 0.890,
            },
        ],
    }
    lb_path.write_text(json.dumps(data))

    # 1. Default table output
    res_table = runner.invoke(cli, ["leaderboard", "--leaderboard-path", str(lb_path)])
    assert res_table.exit_code == 0
    assert "gpt4o_agent" in res_table.output
    assert "claude_agent" in res_table.output

    # 2. Markdown output
    res_md = runner.invoke(cli, ["leaderboard", "--leaderboard-path", str(lb_path), "--format", "markdown"])
    assert res_md.exit_code == 0
    assert "# Leaderboard" in res_md.output
    assert "| # | System" in res_md.output

    # 3. HTML output
    res_html = runner.invoke(cli, ["leaderboard", "--leaderboard-path", str(lb_path), "--format", "html"])
    assert res_html.exit_code == 0
    assert "<table>" in res_html.output

    # 4. JSON output
    res_json = runner.invoke(cli, ["leaderboard", "--leaderboard-path", str(lb_path), "--format", "json"])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert len(parsed) == 2

    # 5. Filter by domain
    res_filt = runner.invoke(
        cli,
        ["leaderboard", "--leaderboard-path", str(lb_path), "--domain", "pix_assist", "--format", "json"],
    )
    assert res_filt.exit_code == 0
    filtered = json.loads(res_filt.output)
    assert len(filtered) == 1
    assert filtered[0]["system_id"] == "gpt4o_agent"
