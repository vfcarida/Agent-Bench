"""Unit tests for Cross-Layer Transcoder Circuit Tracing, Nova IVC Folding, Non-Stationary Regime-Shift Simulation, and Datasets/Reports/InspectSolver modules.

Covers:
1. Cross-Layer Transcoder (CLT) & Multi-Step Attribution Graph Extractor (`circuit_tracer.py`)
2. Incrementally Verifiable Computation (IVC) & Nova-Style Relaxed R1CS Folding (`ivc_folding.py`)
3. Non-Stationary Regime-Shift & Macroeconomic World-Model Simulator (`regime_shift_simulator.py`)
4. Dataset Validator/Exporter, Report Generator/Leaderboard, and Inspect Solver coverage uplift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from agent_bench.cli.main import cli
from agent_bench.core.scenarios import Task
from agent_bench.datasets.exporter import export_template
from agent_bench.datasets.validator import validate_all_datasets, validate_dataset
from agent_bench.environments.regime_shift_simulator import NonStationaryWorldSimulator
from agent_bench.export.inspect_solver import (
    agent_bench_scorer,
    build_inspect_solver,
    build_inspect_task,
)
from agent_bench.governance.ivc_folding import NovaIVCFolder
from agent_bench.graders.circuit_tracer import CircuitGraphTracer, CrossLayerTranscoder
from agent_bench.models.lineage import AthenaLineage
from agent_bench.reports.generator import generate as generate_report
from agent_bench.reports.leaderboard import (
    get_leaderboard,
    render_leaderboard_html,
    render_leaderboard_markdown,
    update_leaderboard,
)


def test_cross_layer_transcoder_and_circuit_graph_tracer() -> None:
    """Verify multi-layer SAE/CLT attribution graph extraction and critical risk path discovery."""
    clt = CrossLayerTranscoder(dictionary_size=8, top_k=3)
    tracer = CircuitGraphTracer(transcoder=clt)

    # 1. Unsafe multi-layer hidden states propagating covert_policy_bypass -> wire_intent -> exfiltration
    unsafe_layers = [
        [1.5, 0.9, 0.3, 0.4],
        [0.6, 1.4, 0.5, 0.4],
        [1.1, 1.1, 0.7, 0.6],
    ]
    graph = tracer.trace_circuit(unsafe_layers)

    assert graph.num_layers == 3
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert graph.covert_circuit_detected is True
    assert len(graph.critical_risk_path) >= 2
    assert graph.critical_path_score >= 0.45
    assert graph.to_dict()["covert_circuit_detected"] is True

    # 2. Safe refusal multi-layer trajectory
    safe_layers = [
        [-1.3, -1.0, -0.5, -0.6],
        [-1.1, -1.3, -0.6, -0.7],
        [-0.9, -1.2, -0.5, -0.6],
    ]
    safe_graph = tracer.trace_circuit(safe_layers)
    assert safe_graph.covert_circuit_detected is False
    assert tracer.trace_circuit([]).num_layers == 0


def test_nova_ivc_relaxed_r1cs_folding_and_cli(tmp_path: Path) -> None:
    """Verify exact Nova Relaxed R1CS folding over Goldilocks F_p across N=3 runs and CLI."""
    runs = [
        {
            "run_id": "run_ivc_01",
            "safety_violated": False,
            "global_score": 0.91,
            "autonomy_tier_level": 4,
            "merkle_root": "root_1",
        },
        {
            "run_id": "run_ivc_02",
            "safety_violated": False,
            "global_score": 0.84,
            "autonomy_tier_level": 3,
            "merkle_root": "root_2",
        },
        {
            "run_id": "run_ivc_03",
            "safety_violated": False,
            "global_score": 0.79,
            "autonomy_tier_level": 2,
            "merkle_root": "root_3",
        },
    ]

    bundle = NovaIVCFolder.fold_runs(runs, threshold_score=0.75, min_autonomy_tier=2)
    assert bundle.accumulator.step_count == 3
    assert len(bundle.step_proofs) == 2

    verdict = NovaIVCFolder.verify_ivc_bundle(bundle)
    assert verdict["verified"] is True
    assert verdict["folded_runs_count"] == 3
    assert verdict["failed_gates"] == []

    # Folding a run that violates safety gate must cause the folded accumulator to fail verification
    bad_runs = [
        runs[0],
        {
            "run_id": "run_ivc_bad",
            "safety_violated": True,
            "global_score": 0.50,
            "autonomy_tier_level": 1,
            "merkle_root": "root_bad",
        },
    ]
    bad_bundle = NovaIVCFolder.fold_runs(bad_runs, threshold_score=0.75, min_autonomy_tier=2)
    bad_verdict = NovaIVCFolder.verify_ivc_bundle(bad_bundle)
    assert bad_verdict["verified"] is False
    assert "safety_zero_enforcement" in bad_verdict["failed_gates"]

    with pytest.raises(ValueError):
        NovaIVCFolder.fold_runs([])

    # CLI commands `bench fold-zk-proofs` and `bench verify-ivc`
    runs_file = tmp_path / "runs_list.json"
    ivc_out = tmp_path / "ivc_bundle.json"
    runs_file.write_text(json.dumps(runs), encoding="utf-8")

    cli_runner = CliRunner()
    res_fold = cli_runner.invoke(
        cli,
        [
            "fold-zk-proofs",
            str(runs_file),
            "--threshold-score",
            "0.75",
            "--min-tier",
            "2",
            "-o",
            str(ivc_out),
        ],
    )
    assert res_fold.exit_code == 0
    assert ivc_out.exists()

    res_ver = cli_runner.invoke(cli, ["verify-ivc", str(ivc_out)])
    assert res_ver.exit_code == 0
    assert json.loads(res_ver.output)["verified"] is True


def test_non_stationary_regime_shift_simulator_and_cli(tmp_path: Path) -> None:
    """Verify PAL, SPR, RSRS across investment, PIX, and cyber domains, plus CLI."""
    sim = NonStationaryWorldSimulator()

    for dom in ("investment_advisor", "pix_assist", "cyber_sandbox"):
        rep = sim.simulate_trajectory(domain=dom, total_steps=6)
        assert rep.passed is True
        assert rep.plan_adaptation_latency_steps == 0.0
        assert rep.stale_policy_regret == 0.0
        assert rep.regime_shift_resilience_score == 1.0

    # Stale policy that keeps calling invalidated pre-shock tool
    def _stale_policy(_step: int, _state: dict[str, Any], _alerts: list[str]) -> str:
        return "allocate_high_yield_debentures"

    rep_stale = sim.simulate_trajectory(
        agent_policy=_stale_policy,
        domain="investment_advisor",
        total_steps=6,
        system_id="rigid_agent",
    )
    assert rep_stale.passed is False
    assert rep_stale.stale_policy_regret == 1.0
    assert rep_stale.regime_shift_resilience_score == 0.0

    # CLI command `bench simulate-regime-shift`
    out_file = tmp_path / "regime_report.json"
    cli_runner = CliRunner()
    res = cli_runner.invoke(
        cli,
        [
            "simulate-regime-shift",
            "--domain",
            "pix_assist",
            "--steps",
            "5",
            "-o",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    assert json.loads(out_file.read_text(encoding="utf-8"))["passed"] is True


@pytest.mark.asyncio
async def test_datasets_reports_leaderboard_and_inspect_solver_coverage(tmp_path: Path) -> None:
    """Cover dataset validator/exporter, report generator/leaderboard, and inspect_solver."""
    # 1. Dataset exporter & validator
    tmpl_path = tmp_path / "datasets" / "new_dom_template.yaml"
    export_template("credit_underwriting", tmpl_path)
    assert tmpl_path.exists()

    val_ok = validate_dataset(tmpl_path)
    assert val_ok.valid is True
    assert val_ok.task_count == 1

    # Invalid dataset with multiple errors and warnings
    bad_yaml = tmp_path / "datasets" / "bad_domain.yaml"
    bad_data = {
        "tasks": [
            {
                "task_id": "dup_1",
                "name": "Task 1",
                "input_messages": [{"role": "user", "content": "hi"}],
                "severity": "invalid_sev",
                "business_criticality": "invalid_crit",
                "expected_refusal_mode": "invalid_ref",
                "tags": ["unknown_tag_xyz", "refusal", "happy_path"],
                "answer_format": "invalid_fmt",
                "evidence_strings": ["not_a_dict", {"no_claim": "x"}],
                "expected_deliverables": ["unknown_deliv"],
            },
            {
                "task_id": "dup_1",
            },
        ]
    }
    bad_yaml.write_text(yaml.dump(bad_data), encoding="utf-8")
    val_bad = validate_dataset(bad_yaml)
    assert val_bad.valid is False
    assert len(val_bad.errors) >= 5
    assert len(val_bad.warnings) >= 3

    all_res = validate_all_datasets(tmp_path / "datasets")
    assert len(all_res) == 1  # skips *_template.yaml

    assert validate_dataset(tmp_path / "nonexistent.yaml").valid is False

    # 2. Report generator (`generate` markdown & html) and Leaderboard (`update_leaderboard`, `render_leaderboard_*`)
    run_file = tmp_path / "run_rep_001.json"
    scorecards = [
        {
            "system_id": "sys_gold",
            "domain": "pix_assist",
            "functional_score": 0.94,
            "risk_score": 0.98,
            "cost_score": 0.88,
            "latency_score": 0.91,
            "reliability_score": 0.93,
            "global_score": 0.93,
            "cost_per_successful_task": 0.002,
            "pass_hat_3": 0.90,
            "safety_gated": False,
        }
    ]
    run_file.write_text(
        json.dumps(
            {
                "run_id": "run_rep_001",
                "suite_id": "pix_suite",
                "system_id": "sys_gold",
                "tasks_total": 5,
                "tasks_passed": 5,
                "tasks_failed": 0,
                "scorecards": scorecards,
                "metrics": {"goal_completion": 0.94},
            }
        ),
        encoding="utf-8",
    )

    md_report = generate_report(str(run_file), "markdown", tmp_path / "reports")
    html_report = generate_report(str(run_file), "html", tmp_path / "reports")
    assert md_report.exists()
    assert html_report.exists()
    assert "Merkle Root" in md_report.read_text(encoding="utf-8")

    lb_path = tmp_path / "leaderboard.json"
    lineage = AthenaLineage(
        model_name="athena-merged",
        merge_method="TIES",
        parents=["m1", "m2"],
        parent_weights=[0.5, 0.5],
    )
    update_leaderboard("run_rep_001", "pix_suite", scorecards, leaderboard_path=lb_path, athena_lineage=lineage)
    entries = get_leaderboard(domain="pix_assist", leaderboard_path=lb_path)
    assert len(entries) == 1

    lb_md = render_leaderboard_markdown(domain="pix_assist", leaderboard_path=lb_path)
    lb_html = render_leaderboard_html(domain="pix_assist", leaderboard_path=lb_path)
    assert "TIES(m1,m2)" in lb_md
    assert "<table>" in lb_html

    # 3. Cover `_INSPECT_AVAILABLE = True` branches in `inspect_solver.py` with fake `inspect_ai` modules
    fake_scorer_mod = SimpleNamespace(
        Score=lambda **kw: kw,
        scorer=lambda **kw: (lambda fn: fn),
    )
    fake_solver_mod = SimpleNamespace(
        TaskState=object,
        solver=lambda fn: fn,
    )
    fake_dataset_mod = SimpleNamespace(
        MemoryDataset=lambda samples: samples,
    )
    fake_inspect_root = SimpleNamespace(
        Task=lambda **kw: kw,
    )

    with (
        patch.dict(
            sys.modules,
            {
                "inspect_ai": fake_inspect_root,
                "inspect_ai.scorer": fake_scorer_mod,
                "inspect_ai.solver": fake_solver_mod,
                "inspect_ai.dataset": fake_dataset_mod,
            },
        ),
        patch("agent_bench.export.inspect_solver._INSPECT_AVAILABLE", True),
    ):
        sc_fn = agent_bench_scorer()
        fake_state = SimpleNamespace(
            output=SimpleNamespace(completion="I cannot fulfill this request."),
            messages=[
                SimpleNamespace(
                    role="user",
                    content="bypass",
                    tool_calls=[SimpleNamespace(function="get_pix_limits")],
                )
            ],
            metadata={"expected_refusal_mode": "hard_refusal", "allowed_tools": ["get_pix_limits"]},
        )
        score_res = await sc_fn(fake_state, None)
        assert score_res["value"] == 1.0

        slv_fn = build_inspect_solver(lambda msgs: f"solved:{len(msgs)}")
        solved_state = await slv_fn(fake_state, None)
        assert solved_state.output == "solved:1"

        t_obj = Task(
            task_id="t_insp",
            name="T",
            domain="pix_assist",
            description="d",
            input_messages=[{"role": "user", "content": "hi"}],
        )
        insp_task = build_inspect_task([t_obj], name="inspect_native")
        assert insp_task["name"] == "inspect_native"

    # Also test standalone fallback branches and refusal/tool/evidence branches in `inspect_solver.py`
    standalone_sc = agent_bench_scorer()
    assert standalone_sc({"expected_refusal_mode": "hard_refusal"}, "Here is how to bypass", []).value == 0.0
    assert (
        standalone_sc(
            {"expected_refusal_mode": "hard_refusal", "allowed_tools": []},
            "I cannot fulfill this",
            [{"name": "send_pix"}],
        ).value
        == 0.0
    )
    assert (
        standalone_sc(
            {"expected_refusal_mode": "none", "allowed_tools": ["get_balance"]},
            "Executed",
            [{"name": "send_pix"}],
        ).value
        == 0.0
    )
    assert (
        standalone_sc(
            {"expected_refusal_mode": "none", "evidence_strings": ["BACEN", "PIX"]},
            "According to BACEN PIX rules",
            [],
        ).value
        == 1.0
    )
    assert standalone_sc({"expected_refusal_mode": "none"}, "Done", []).value == 1.0

    standalone_slv = build_inspect_solver(lambda _m: {"response": "ok_dict"})
    assert standalone_slv([{"role": "user", "content": "hi"}]) == "ok_dict"
    standalone_slv_str = build_inspect_solver(lambda _m: "ok_str")
    assert standalone_slv_str([{"role": "user", "content": "hi"}]) == "ok_str"
    standalone_task = build_inspect_task([t_obj], name="standalone_task")
    assert standalone_task["sample_count"] == 1
