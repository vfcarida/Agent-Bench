"""Unit tests for Inspect AI export CLI commands."""

import json
from pathlib import Path

from click.testing import CliRunner

from agent_bench.cli.main import cli


def test_cli_export_inspect(tmp_path: Path):
    runner = CliRunner()
    out_file = tmp_path / "inspect_pix.jsonl"
    result = runner.invoke(
        cli,
        ["export-inspect", "--domain", "pix_assist", "--split", "dev", "--output", str(out_file)],
    )
    assert result.exit_code == 0
    assert "Exported" in result.output
    assert out_file.exists()

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) > 0
    first = lines[0]
    assert "id" in first
    assert "input" in first
    assert "target" in first
    assert first["metadata"]["domain"] == "pix_assist"


def test_cli_export_inspect_log(tmp_path: Path):
    runner = CliRunner()
    run_id = "test_run_inspect_cli"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "suite_id": "pix_suite",
        "system_id": "mock_agent",
        "model_id": "gpt-4o",
        "tasks_total": 1,
        "tasks_passed": 1,
        "tasks_failed": 0,
        "metrics": {"global_score": 0.98},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    trace_line = {
        "event_type": "user_message",
        "timestamp": "2026-09-25T12:00:00Z",
        "data": {"task_id": "T_1", "content": "Hello Inspect"},
        "event_id": "ev_001",
    }
    (run_dir / "traces.jsonl").write_text(json.dumps(trace_line) + "\n", encoding="utf-8")

    out_file = tmp_path / "inspect_log.json"
    result = runner.invoke(
        cli,
        ["export-inspect-log", run_id, "--runs-dir", str(tmp_path), "--output", str(out_file)],
    )
    assert result.exit_code == 0
    assert "Inspect AI evaluation log exported" in result.output
    assert out_file.exists()

    log_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert log_data["eval"]["run_id"] == run_id
    assert log_data["results"]["tasks_passed"] == 1
