"""Unit tests for HTML report generation."""

import json
import pytest
from pathlib import Path

from agent_bench.reports.html_report import generate_html_report, score_class


def test_score_class():
    assert score_class(0.9) == "score-high"
    assert score_class(0.5) == "score-mid"
    assert score_class(0.2) == "score-low"


def test_generate_html_report(tmp_path):
    # Create a fake run artifact
    runs_dir = tmp_path / "data" / "runs"
    runs_dir.mkdir(parents=True)

    run_data = {
        "run_id": "test-run-12345678",
        "suite_id": "test_suite",
        "system_id": "sys_a,sys_b",
        "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:01:00",
        "config_hash": "abc123",
        "benchmark_version": "1.0.0",
        "tasks_total": 20,
        "tasks_passed": 15,
        "tasks_failed": 5,
        "duration_ms": 5000,
        "seed": 42,
        "metrics": {},
        "scorecards": [
            {
                "system_id": "sys_a",
                "domain": "pix_assist",
                "functional_score": 0.9,
                "risk_score": 0.8,
                "cost_score": 0.95,
                "latency_score": 0.85,
                "reliability_score": 0.9,
                "global_score": 0.87,
            },
            {
                "system_id": "sys_b",
                "domain": "pix_assist",
                "functional_score": 0.6,
                "risk_score": 0.5,
                "cost_score": 0.9,
                "latency_score": 0.7,
                "reliability_score": 0.6,
                "global_score": 0.59,
            },
        ],
    }
    (runs_dir / "test-run-12345678.json").write_text(json.dumps(run_data))

    # Monkey-patch the runs dir path
    import agent_bench.reports.html_report as hr
    original = Path("data/runs")

    output_dir = tmp_path / "reports"
    # We need to call with proper working dir context
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = generate_html_report("test-run-12345678", output_dir)
    finally:
        os.chdir(old_cwd)

    assert result.exists()
    content = result.read_text()
    assert "test-run" in content
    assert "sys_a" in content
    assert "0.870" in content
    assert "score-high" in content
    assert "<!DOCTYPE html>" in content
