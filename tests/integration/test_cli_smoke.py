"""Smoke tests for the CLI."""

from click.testing import CliRunner

from agent_bench.cli.main import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "agent-bench" in result.output


def test_validate_config_missing_dir():
    runner = CliRunner()
    result = runner.invoke(cli, ["--config-dir", "/nonexistent", "validate-config"])
    assert result.exit_code == 1


def test_validate_config_valid(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "suites").mkdir()
    (tmp_path / "domains").mkdir()
    runner = CliRunner()
    result = runner.invoke(cli, ["--config-dir", str(tmp_path), "validate-config"])
    assert result.exit_code == 0
    assert "Config valid" in result.output


def test_export_dataset_template(tmp_path):
    out = tmp_path / "template.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["export-dataset-template", "test_domain", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
