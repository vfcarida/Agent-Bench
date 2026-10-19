"""Unit tests for report generators, run comparators, v2 dataset loader, cross-artifact judge, and public SDK."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import agent_bench
from agent_bench.cli.main import cli
from agent_bench.core.scenarios import Task
from agent_bench.datasets.loader_v2 import load_eval_cases, load_from_legacy
from agent_bench.generators import templates as gen_templates
from agent_bench.judges.cross_artifact import CrossArtifactConsistencyJudge
from agent_bench.reports.comparator import compare, compare_systems
from agent_bench.reports.generator import generate
from agent_bench.reports.html_report import generate_html_report
from agent_bench.storage.analytics import (
    aggregate_by_domain,
    aggregate_by_system,
    cross_system_comparison,
    trend_over_runs,
)


def _write_sample_run(runs_dir: Path, run_id: str, func_score: float = 0.85, cost: float = 0.02) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    scorecards = [
        {
            "system_id": "sys_alpha",
            "domain": "pix_assist",
            "functional_score": func_score,
            "risk_score": 1.0,
            "cost_score": 0.98,
            "latency_score": 0.90,
            "reliability_score": func_score,
            "global_score": func_score,
            "pass_hat_3": func_score,
            "pass_at_3": min(1.0, func_score + 0.05),
            "cost_per_successful_task": cost,
            "latency_p50": 300.0,
            "latency_p90": 500.0,
            "latency_p99": 700.0,
            "safety_violations": 0,
            "safety_gated": False,
        }
    ]
    artifact = {
        "run_id": run_id,
        "suite_id": "pix_basic_v1",
        "system_id": "sys_alpha",
        "started_at": "2026-10-19T12:00:00Z",
        "duration_ms": 420.0,
        "config_hash": "abc123def456",
        "benchmark_version": "1.0.0",
        "seed": 42,
        "tasks_total": 10,
        "tasks_passed": 8,
        "tasks_failed": 2,
        "scorecards": scorecards,
        "metrics": {
            "scorecards": scorecards,
            "task_results": [
                {
                    "system_id": "sys_alpha",
                    "task_id": "PIX_001",
                    "domain": "pix_assist",
                    "difficulty": "easy",
                    "passed": True,
                },
                {
                    "system_id": "sys_alpha",
                    "task_id": "PIX_002",
                    "domain": "pix_assist",
                    "difficulty": "hard",
                    "passed": False,
                },
            ],
        },
    }
    out = runs_dir / f"{run_id}.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return out


class TestReportGenerationAndComparison:
    def test_generate_markdown_and_html_reports(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "reports"
        _write_sample_run(runs_dir, "run_0001_alpha")

        md_path = generate("run_0001_alpha", "markdown", out_dir, runs_dir=runs_dir)
        assert md_path.exists()
        md_text = md_path.read_text(encoding="utf-8")
        assert "System Scorecards & Graduated Autonomy Tiers" in md_text
        assert "Safety-Constrained Pareto Frontier" in md_text

        wrapped_html = generate("run_0001_alpha", "html", out_dir, runs_dir=runs_dir)
        assert wrapped_html.exists()

        rich_html = generate_html_report("run_0001_alpha", out_dir, runs_dir=runs_dir)
        assert rich_html.exists()
        assert "Autonomy Tier" in rich_html.read_text(encoding="utf-8")

    def test_compare_runs_and_systems(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _write_sample_run(runs_dir, "run_a_1234", func_score=0.80, cost=0.03)
        _write_sample_run(runs_dir, "run_b_5678", func_score=0.92, cost=0.01)

        comp_md = tmp_path / "reports" / "comparison.md"
        compare(["run_a_1234", "run_b_5678"], comp_md, runs_dir=runs_dir)
        assert comp_md.exists()
        assert comp_md.with_suffix(".csv").exists()
        assert "Run Comparison" in comp_md.read_text(encoding="utf-8")

        sys_md = tmp_path / "reports" / "sys_comp.md"
        scorecards = [
            {
                "system_id": "sys_1",
                "domain": "pix_assist",
                "functional_score": 0.90,
                "risk_score": 1.0,
                "cost_score": 0.95,
                "latency_score": 0.90,
                "reliability_score": 0.88,
                "global_score": 0.91,
            },
            {
                "system_id": "sys_2",
                "domain": "pix_assist",
                "functional_score": 0.70,
                "risk_score": 1.0,
                "cost_score": 0.85,
                "latency_score": 0.80,
                "reliability_score": 0.65,
                "global_score": 0.75,
            },
        ]
        compare_systems(scorecards, sys_md)
        assert sys_md.exists()
        assert "Winner: sys_1" in sys_md.read_text(encoding="utf-8") or "**Winner:** sys_1" in sys_md.read_text(encoding="utf-8")

    def test_cli_diagnose_run_and_reports(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _write_sample_run(runs_dir, "diag_run_9999", func_score=0.92, cost=0.015)

        runner = CliRunner()
        res_table = runner.invoke(
            cli,
            ["diagnose-run", "diag_run_9999", "--runs-dir", str(runs_dir), "--format", "table"],
        )
        assert res_table.exit_code == 0
        assert "PARETO-OPTIM" in res_table.output
        assert "Tier 4" in res_table.output

        res_json = runner.invoke(
            cli,
            ["diagnose-run", "diag_run_9999", "--runs-dir", str(runs_dir), "--format", "json"],
        )
        assert res_json.exit_code == 0
        parsed = json.loads(res_json.output)
        assert parsed["run_id"] == "diag_run_9999"
        assert len(parsed["pareto_frontier"]["frontier"]) == 1


class TestLoaderV2AndTemplates:
    def test_load_eval_cases_and_legacy(self, tmp_path: Path) -> None:
        gold_dev = tmp_path / "gold" / "dev"
        gold_dev.mkdir(parents=True)
        sample_yaml = gold_dev / "pix.yaml"
        sample_yaml.write_text(
            "cases:\n"
            "  - id: T_V2_1\n"
            "    family: transactional_tools\n"
            "    domain: pix_assist\n"
            "    source_type: human_gold\n"
            "    prompt_or_user_goal: Check PIX balance\n"
            "    input_messages:\n"
            "      - role: user\n"
            "        content: Check PIX balance\n",
            encoding="utf-8",
        )
        cases = load_eval_cases(tmp_path, split="dev", source_type="human_gold")
        assert len(cases) == 1
        assert cases[0].id == "T_V2_1"

        all_cases = load_eval_cases(tmp_path)
        assert len(all_cases) == 1

        legacy = load_from_legacy("pix_assist")
        assert len(legacy) >= 1
        assert load_from_legacy("nonexistent_domain_xyz") == []

    def test_generator_templates_integrity(self) -> None:
        assert "base_prompt" in gen_templates.PIX_TRANSFER_TEMPLATE
        assert "base_prompt" in gen_templates.SUITABILITY_TEMPLATE


class TestCrossArtifactConsistencyJudge:
    async def test_consistent_and_contradictory_artifacts(self) -> None:
        judge = CrossArtifactConsistencyJudge()
        assert judge.judge_id == "cross_artifact_consistency"
        assert judge.judge_type == "deterministic"

        task = Task(
            task_id="T_CA_1",
            domain="investment_advisor",
            name="Cross-artifact check",
            description="Verify numeric consistency across artifacts",
            input_messages=[],
        )

        single_verdict = await judge.evaluate(task, {"response": "Only one output"}, [])
        assert single_verdict.passed is True
        assert single_verdict.score == 1.0

        consistent_res = {
            "response": "Gross return is R$ 15.000,00 with rate: 12.5%.",
            "structured_data": "Gross return: R$ 15.000,00; rate: 12.5%",
        }
        v_ok = await judge.evaluate(task, consistent_res, [])
        assert v_ok.passed is True
        assert v_ok.score >= 0.9

        contradictory_res = {
            "artifacts": [
                {"id": "memo", "type": "doc", "content": "revenue: R$ 15.000,00\nmargin: 12.5%"},
                {"id": "model", "type": "sheet", "content": "revenue: R$ 28.000,00\nmargin: 4.0%"},
            ]
        }
        v_fail = await judge.evaluate(task, contradictory_res, [])
        assert v_fail.passed is False
        assert v_fail.score < 0.9
        assert len(v_fail.metadata.get("contradictions", [])) >= 1


class TestStorageAnalyticsAndPublicSDK:
    def test_storage_analytics_queries(self, tmp_path: Path) -> None:
        from agent_bench.storage.parquet import save_metrics_parquet

        runs_dir = tmp_path / "runs"
        run1 = runs_dir / "r1"
        run1.mkdir(parents=True)
        save_metrics_parquet(
            [
                {
                    "run_id": "r1",
                    "system_id": "sys_a",
                    "task_id": "t1",
                    "domain": "pix_assist",
                    "metric_name": "functional_score",
                    "metric_value": 0.9,
                    "metric_category": "functional",
                    "passed": True,
                    "timestamp": "2026-10-19T10:00:00Z",
                }
            ],
            run1 / "metrics.parquet",
        )
        by_sys = aggregate_by_system(runs_dir)
        by_dom = aggregate_by_domain(runs_dir)
        trend = trend_over_runs(runs_dir, "sys_a")
        cross = cross_system_comparison(runs_dir, "pix_assist")
        assert len(by_sys) >= 1
        assert len(by_dom) >= 1
        assert len(trend) >= 1
        assert len(cross) >= 1

    def test_public_sdk_exports_and_py_typed(self) -> None:
        assert agent_bench.__version__ == "1.0.0"
        assert agent_bench.__author__ == "Vinicius Caridá"
        for symbol in ("AgentRunner", "TaskEnvironment", "Evaluator", "compute_pareto_frontier", "AutonomyTier"):
            assert hasattr(agent_bench, symbol)
            assert symbol in agent_bench.__all__

        py_typed_path = Path(agent_bench.__file__).resolve().parent / "py.typed"
        assert py_typed_path.exists()
