"""Unit tests for leaderboard."""

import json
import pytest
from pathlib import Path

from agent_bench.reports.leaderboard import (
    get_leaderboard,
    render_leaderboard_markdown,
    update_leaderboard,
)


@pytest.fixture
def scorecards():
    return [
        {
            "system_id": "sys_a",
            "domain": "pix_assist",
            "global_score": 0.85,
            "functional_score": 0.9,
            "risk_score": 0.8,
            "cost_score": 0.95,
            "latency_score": 0.9,
            "reliability_score": 0.85,
            "weighting_profile": "transactional_high_risk",
        },
        {
            "system_id": "sys_b",
            "domain": "pix_assist",
            "global_score": 0.72,
            "functional_score": 0.7,
            "risk_score": 0.75,
            "cost_score": 0.9,
            "latency_score": 0.8,
            "reliability_score": 0.7,
            "weighting_profile": "transactional_high_risk",
        },
    ]


def test_update_leaderboard(tmp_path, scorecards):
    lb_path = tmp_path / "lb.json"
    update_leaderboard("run_001", "suite_1", scorecards, lb_path)
    assert lb_path.exists()

    data = json.loads(lb_path.read_text())
    assert data["total_entries"] == 2
    # Should be sorted by global_score desc
    assert data["entries"][0]["system_id"] == "sys_a"


def test_update_leaderboard_accumulates(tmp_path, scorecards):
    lb_path = tmp_path / "lb.json"
    update_leaderboard("run_001", "suite_1", scorecards, lb_path)
    update_leaderboard("run_002", "suite_1", scorecards, lb_path)

    data = json.loads(lb_path.read_text())
    assert data["total_entries"] == 4


def test_get_leaderboard_with_domain_filter(tmp_path, scorecards):
    lb_path = tmp_path / "lb.json"
    extra = [{**scorecards[0], "domain": "investment_advisor", "global_score": 0.6}]
    update_leaderboard("run_001", "s1", scorecards + extra, lb_path)

    entries = get_leaderboard(domain="pix_assist", leaderboard_path=lb_path)
    assert all(e["domain"] == "pix_assist" for e in entries)


def test_render_markdown(tmp_path, scorecards):
    lb_path = tmp_path / "lb.json"
    update_leaderboard("run_001", "s1", scorecards, lb_path)
    md = render_leaderboard_markdown(leaderboard_path=lb_path)
    assert "sys_a" in md
    assert "0.850" in md


def test_empty_leaderboard(tmp_path):
    entries = get_leaderboard(leaderboard_path=tmp_path / "nope.json")
    assert entries == []
