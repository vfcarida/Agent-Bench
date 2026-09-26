"""Unit tests for HTML report generation."""

import json

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


def test_generate_html_report_with_pass_k_and_cis(tmp_path):
    runs_dir = tmp_path / "data" / "runs"
    runs_dir.mkdir(parents=True)

    run_data = {
        "run_id": "run-passk-123",
        "suite_id": "pix_basic_v1",
        "system_id": "test_sys",
        "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:01:00",
        "config_hash": "abc123",
        "benchmark_version": "1.0.0",
        "tasks_total": 10,
        "tasks_passed": 6,
        "tasks_failed": 4,
        "duration_ms": 1000,
        "seed": 42,
        "metrics": {
            "pass_k": [
                {
                    "system_id": "test_sys",
                    "domain": "pix_assist",
                    "pass_1": 0.6,
                    "pass_1_ci": [0.3, 0.9],
                    "pass_3": 0.8,
                    "pass_3_ci": [0.5, 1.0],
                    "pass_hat_3": 0.4,
                    "pass_hat_3_ci": [0.1, 0.7],
                    "sampling_denominator": "all_trials_including_failures_timeouts",
                }
            ]
        },
        "scorecards": [],
    }
    (runs_dir / "run-passk-123.json").write_text(json.dumps(run_data))

    output_dir = tmp_path / "reports"
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = generate_html_report("run-passk-123", output_dir)
    finally:
        os.chdir(old_cwd)

    assert result.exists()
    content = result.read_text()
    assert "Pass@1 (95% CI)" in content
    assert "Pass^3 (Reliability)" in content
    assert "0.600" in content
    assert "[0.30, 0.90]" in content
    assert "0.800" in content
    assert "[0.10, 0.70]" in content
    assert '<html lang="en">' in content


def test_generate_html_report_locale(tmp_path):
    runs_dir = tmp_path / "data" / "runs"
    runs_dir.mkdir(parents=True)
    run_data = {
        "run_id": "run-locale-test",
        "started_at": "2026-01-01T00:00:00",
        "tasks_total": 1,
        "tasks_passed": 1,
        "tasks_failed": 0,
    }
    (runs_dir / "run-locale-test.json").write_text(json.dumps(run_data))
    output_dir = tmp_path / "reports"

    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        res_default = generate_html_report("run-locale-test", output_dir)
        assert '<html lang="en">' in res_default.read_text()

        res_pt = generate_html_report("run-locale-test", output_dir, locale="pt-BR")
        assert '<html lang="pt-BR">' in res_pt.read_text()
    finally:
        os.chdir(old_cwd)


def test_generate_html_report_custom_runs_dir_and_filter(tmp_path):
    custom_runs = tmp_path / "custom_runs"
    custom_runs.mkdir()
    run_data = {
        "run_id": "run-filter-custom-1234",
        "started_at": "2026-01-01T00:00:00",
        "tasks_total": 5,
        "tasks_passed": 5,
        "tasks_failed": 0,
        "scorecards": [
            {
                "system_id": "test_agent",
                "domain": "pix_assist",
                "functional_score": 1.0,
                "global_score": 0.95,
            }
        ],
    }
    (custom_runs / "run-filter-custom-1234.json").write_text(json.dumps(run_data))
    output_dir = tmp_path / "custom_reports"

    # Call directly with runs_dir parameter
    result = generate_html_report(
        "run-filter-custom-1234", output_dir, runs_dir=custom_runs
    )
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "run-filter-custom" in content
    assert "id=\"tableFilter\"" in content
    assert "test_agent" in content



