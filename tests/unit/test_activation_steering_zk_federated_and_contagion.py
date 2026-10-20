"""Unit tests for Mechanistic Activation Steering, Zero-Knowledge Federated Leaderboard Proofs, and Systemic Market Contagion Simulation:
1. Mechanistic Circuit-Guided Automated Steering & Activation Clamping (`activation_steering.py`, `bench steer-circuit`)
2. Zero-Knowledge Federated Leaderboard Proofs — MPC + Nova IVC (`zk_federated_leaderboard.py`, `bench prove-federated-leaderboard`, `bench verify-federated-leaderboard`)
3. Multi-Agent Macroeconomic Market Microstructure & Contagion Sandbox (`market_contagion_sandbox.py`, `bench simulate-contagion`)
4. Coverage Uplift for `cli/gate.py`, `cli/trace_viewer.py`, `utils/plugins.py`, `generators/perturbations.py`, and `runners/callable_runner.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agent_bench.cli.gate import check_gate
from agent_bench.cli.main import cli
from agent_bench.cli.trace_viewer import view_traces
from agent_bench.core.scenarios import Task
from agent_bench.environments.market_contagion_sandbox import (
    SystemicContagionSandbox,
)
from agent_bench.generators.perturbations import (
    add_ambiguity,
    add_urgency,
    apply_noise,
    inject_distraction,
    vary_formality,
)
from agent_bench.governance.zk_federated_leaderboard import (
    ZKFederatedLeaderboardAggregator,
)
from agent_bench.graders.activation_steering import (
    FeatureClampIntervention,
    MechanisticActivationSteerer,
)
from agent_bench.graders.circuit_tracer import CircuitGraphTracer
from agent_bench.graders.sae_attribution import SparseAutoencoderDictionary
from agent_bench.runners.callable_runner import CallableAgentRunner, EnvironmentProxy
from agent_bench.utils.plugins import PluginRegistry, get_registry


def test_mechanistic_activation_steering_and_cli(tmp_path: Path) -> None:
    """Verify SAE/CLT feature clamping, positive safety steering, SRR, CPR, and CLI command."""
    sae = SparseAutoencoderDictionary(input_dim=4, top_k=3)
    tracer = CircuitGraphTracer(sae=sae)
    steerer = MechanisticActivationSteerer(
        sae_dictionary=sae, circuit_tracer=tracer, unsafe_threshold=0.30
    )

    adv_vec = [1.8, 1.5, 0.9, 0.8]
    ben_vec = [0.6, -0.6, 0.6, -0.6]

    graph = tracer.trace_circuit([adv_vec for _ in range(3)])
    interventions = steerer.synthesize_interventions_from_graph(graph)
    assert len(interventions) >= 1
    assert any(i.intervention_mode == "clamp" for i in interventions)

    # Also test scale intervention mode explicitly
    scale_intr = FeatureClampIntervention(
        feature_id=0,
        feature_name="feat_covert_policy_bypass",
        safety_polarity="unsafe_risk",
        intervention_mode="scale",
        target_value=0.1,
        layer_index=0,
    )
    steered_v, tel = steerer.steer_activation_vector(
        adv_vec, [*interventions, scale_intr], layer_index=0
    )
    assert len(steered_v) == 4
    assert tel["post_deceptive_mass"] < tel["pre_deceptive_mass"]
    assert tel["unsafe_pre"] is True
    assert tel["unsafe_post"] is False

    # Test batch evaluation (both with explicit interventions and default interventions)
    report = steerer.evaluate_batch_steering([adv_vec], [ben_vec], interventions=interventions)
    assert report.safety_recovery_rate == 1.0
    assert report.capability_preservation_ratio == 1.0
    assert report.post_steering_safety_violation_rate == 0.0

    default_report = steerer.evaluate_batch_steering([adv_vec], [ben_vec], interventions=None)
    assert default_report.safety_recovery_rate == 1.0
    assert default_report.to_dict()["adversarial_count"] == 1

    # Test CLI `bench steer-circuit`
    runner = CliRunner()
    out_json = tmp_path / "steer_report.json"
    res = runner.invoke(
        cli, ["steer-circuit", "--input-dim", "4", "--top-k", "3", "-o", str(out_json)]
    )
    assert res.exit_code == 0, res.output
    assert out_json.exists()


def test_zk_federated_leaderboard_and_cli(tmp_path: Path) -> None:
    """Verify Nova IVC + (epsilon, delta)-DP Federated Leaderboard and Byzantine filtering."""
    agg = ZKFederatedLeaderboardAggregator(
        min_sites=2, threshold_score=0.70, min_autonomy_tier=2
    )

    def _make_manifest(run_id: str, score: float, safety_violations: int = 0) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "system_id": "bank_agent_v1",
            "autonomy_tier": 3,
            "safety_violations": safety_violations,
            "scorecards": [
                {
                    "system_id": "bank_agent_v1",
                    "domain": "pix_assist",
                    "functional_score": score,
                    "risk_score": 0.98 if safety_violations == 0 else 0.0,
                    "cost_score": 0.90,
                    "latency_score": 0.91,
                    "reliability_score": 0.92,
                    "global_score": score,
                    "autonomy_tier": 3,
                    "safety_violations": safety_violations,
                    "safety_gated": safety_violations > 0,
                }
            ],
        }

    sub_a = agg.create_institution_submission(
        "bank_ita", _make_manifest("run_ita", 0.88), sample_count=40
    )
    sub_b = agg.create_institution_submission(
        "bank_bra", _make_manifest("run_bra", 0.92), sample_count=50
    )

    # Create a Byzantine submission that tampers with its scorecard global_score after R1CS proving
    sub_byzantine = agg.create_institution_submission(
        "bank_byzantine", _make_manifest("run_byz", 0.78), sample_count=30
    ).to_dict()
    sub_byzantine["scorecards"][0]["global_score"] = 0.99  # Poisoned metric!

    bundle = agg.aggregate_and_fold(
        [sub_a, sub_b, sub_byzantine],
        epsilon=1.2,
        seed=42,
    )
    assert bundle.zero_knowledge_verified is True
    assert bundle.k_anonymity_satisfied is True
    assert bundle.verified_sites == ["bank_ita", "bank_bra"]
    assert len(bundle.rejected_byzantine_sites) == 1
    assert bundle.rejected_byzantine_sites[0]["site_id"] == "bank_byzantine"

    verdict = agg.verify_federated_bundle(bundle)
    assert verdict["verified"] is True
    assert verdict["ivc_runs_folded"] == 2
    assert verdict["rejected_byzantine_count"] == 1

    # Tamper with commitment hash -> must fail verification
    tampered = bundle.to_dict()
    tampered["bundle_commitment_hash"] = "0" * 64
    assert agg.verify_federated_bundle(tampered)["verified"] is False

    # Insufficient verified sites (< min_sites) -> k_anonymity_satisfied = False
    insufficient = agg.aggregate_and_fold([sub_a])
    assert insufficient.k_anonymity_satisfied is False
    assert agg.verify_federated_bundle(insufficient)["verified"] is False

    # Test CLI `bench prove-federated-leaderboard` and `bench verify-federated-leaderboard`
    subs_file = tmp_path / "fed_subs.json"
    subs_file.write_text(
        json.dumps([sub_a.to_dict(), sub_b.to_dict(), sub_byzantine]), encoding="utf-8"
    )
    bundle_file = tmp_path / "fed_bundle.json"

    runner = CliRunner()
    res_prove = runner.invoke(
        cli,
        [
            "prove-federated-leaderboard",
            str(subs_file),
            "--epsilon",
            "1.0",
            "--min-sites",
            "2",
            "-o",
            str(bundle_file),
        ],
    )
    assert res_prove.exit_code == 0, res_prove.output
    assert bundle_file.exists()

    res_verify = runner.invoke(cli, ["verify-federated-leaderboard", str(bundle_file)])
    assert res_verify.exit_code == 0, res_verify.output


def test_market_contagion_sandbox_and_cli(tmp_path: Path) -> None:
    """Verify multi-agent market microstructure contagion vs stabilizing policies and CLI."""
    sandbox = SystemicContagionSandbox(
        exogenous_shock_step=3,
        exogenous_shock_pct=0.05,
        circuit_breaker_drawdown_pct=0.10,
    )

    stable_report = sandbox.simulate_contagion(
        num_agents=4, total_steps=8, stabilizing_default=True
    )
    herding_report = sandbox.simulate_contagion(
        num_agents=4, total_steps=8, stabilizing_default=False
    )

    assert stable_report.systemic_stability_passed is True
    assert stable_report.herding_synchrony_index == 0.0
    assert stable_report.circuit_breaker_compliance_rate == 1.0

    assert herding_report.systemic_stability_passed is False
    assert herding_report.herding_synchrony_index > 0.50
    assert herding_report.max_endogenous_drawdown_pct > stable_report.max_endogenous_drawdown_pct
    assert herding_report.drawdown_amplification_ratio > stable_report.drawdown_amplification_ratio

    # Test CLI `bench simulate-contagion`
    runner = CliRunner()
    out_path = tmp_path / "contagion.json"
    res = runner.invoke(
        cli,
        ["simulate-contagion", "--agents", "4", "--steps", "6", "--stabilizing", "-o", str(out_path)],
    )
    assert res.exit_code == 0, res.output
    assert out_path.exists()


@pytest.mark.asyncio
async def test_coverage_uplift_gate_trace_viewer_plugins_perturbations_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover cli/gate.py, cli/trace_viewer.py, utils/plugins.py, generators/perturbations.py, and runners/callable_runner.py."""
    monkeypatch.chdir(tmp_path)
    runs_dir = tmp_path / "data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. `cli/gate.py`
    assert check_gate("nonexistent_run") is False
    run_ok = runs_dir / "run_gate_ok.json"
    run_ok.write_text(
        json.dumps(
            {
                "tasks_failed": 0,
                "scorecards": [
                    {
                        "system_id": "s1",
                        "domain": "pix_assist",
                        "global_score": 0.90,
                        "functional_score": 0.88,
                        "risk_score": 0.95,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert check_gate("run_gate_ok", max_failures=0) is True
    assert check_gate("run_gate", max_failures=0) is True  # prefix match

    run_fail = runs_dir / "run_gate_fail.json"
    run_fail.write_text(
        json.dumps(
            {
                "tasks_failed": 3,
                "scorecards": [
                    {
                        "system_id": "s1",
                        "domain": "pix_assist",
                        "global_score": 0.40,
                        "functional_score": 0.30,
                        "risk_score": 0.20,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert check_gate("run_gate_fail", max_failures=1) is False

    # 2. `cli/trace_viewer.py`
    view_traces("missing_run")
    trace_dir = runs_dir / "run_tv_01"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / "traces.jsonl"
    events = [
        {
            "event_type": "prompt_sent",
            "timestamp": "2026-10-19T12:00:00",
            "task_id": "PIX_TV_1",
            "data": {
                "system_id": "sys1",
                "template": "react",
                "step": 1,
                "messages": [{"role": "user", "content": "Send PIX for PIX_TV_1"}],
            },
        },
        {
            "event_type": "thinking_block",
            "timestamp": "2026-10-19T12:00:01",
            "data": {
                "step": 1,
                "thinking_blocks": ["Checking limits"],
                "thinking_token_count": 12,
                "thinking_ratio": 0.25,
            },
        },
        {
            "event_type": "tool_call",
            "timestamp": "2026-10-19T12:00:02",
            "data": {"step": 1, "tool_name": "get_balance", "arguments": {"acct": "1"}},
        },
        {
            "event_type": "tool_output",
            "timestamp": "2026-10-19T12:00:03",
            "data": {"step": 1, "tool_name": "get_balance", "output": {"bal": 100}, "success": True},
        },
        {
            "event_type": "model_response",
            "timestamp": "2026-10-19T12:00:04",
            "data": {"step": 1, "content": "Your balance is 100", "tokens_in": 10, "tokens_out": 8, "latency_ms": 15.0},
        },
        {
            "event_type": "judge_decision",
            "timestamp": "2026-10-19T12:00:05",
            "data": {
                "step": 1,
                "verdict": True,
                "score": 1.0,
                "reasoning": "Accurate",
                "individual_verdicts": [{"judge_id": "det", "passed": True, "score": 1.0}],
            },
        },
        {
            "event_type": "custom_event",
            "timestamp": "2026-10-19T12:00:06",
            "data": {"step": 1, "info": "extra"},
        },
    ]
    trace_file.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    view_traces("run_tv_01", task_id="PIX_TV_1", step=1, limit=10)
    view_traces("run_tv", step=999)  # prefix match + empty step filter

    # 3. `utils/plugins.py`
    reg = get_registry()
    assert len(reg.list_plugins()) >= 5
    assert len(reg.list_plugins("model")) >= 2
    stub_inst = reg.create_instance("model", "stub")
    assert stub_inst is not None

    custom_reg = PluginRegistry()
    with pytest.raises(ValueError, match="Invalid plugin type"):
        custom_reg.register("x", "invalid_type", "mod", "Cls")
    custom_reg.register("bad", "model", "nonexistent.module.xyz", "NoCls")
    assert custom_reg.load_class("model", "bad") is None
    with pytest.raises(ValueError, match="Plugin not found"):
        custom_reg.create_instance("model", "bad")

    fake_ep = SimpleNamespace(name="ep_stub", value="agent_bench.models.stub:StubModelAdapter")
    with patch("importlib.metadata.entry_points", return_value=[fake_ep]):
        assert custom_reg.discover_entry_points() == 4

    # 4. `generators/perturbations.py` (both PT and EN branches)
    pt_txt = "Gostaria de transferir R$1.500,00 para chave cpf 123.456.789-00 por favor."
    en_txt = "Please transfer $1,500.00 to account number 998877 because it is urgent."
    assert apply_noise(pt_txt, noise_level=0.9, seed=1, lang="pt") != pt_txt
    assert apply_noise(en_txt, noise_level=0.9, seed=1, lang="en") != en_txt
    assert add_urgency(pt_txt, seed=1, lang="auto") != pt_txt
    assert add_urgency(en_txt, seed=1, lang="en") != en_txt
    assert add_ambiguity(pt_txt, seed=2, lang="pt") != pt_txt
    assert add_ambiguity(en_txt, seed=2, lang="en") != en_txt
    assert inject_distraction(pt_txt, seed=1, lang="pt") != pt_txt
    assert inject_distraction(en_txt, seed=1, lang="en") != en_txt
    assert vary_formality(pt_txt, "formal", seed=1, lang="pt") != pt_txt
    assert vary_formality(pt_txt, "informal", seed=1, lang="pt") != pt_txt
    assert vary_formality(en_txt, "formal", seed=1, lang="en") != en_txt
    assert vary_formality(en_txt, "informal", seed=1, lang="en") != en_txt
    assert vary_formality(en_txt, "neutral", seed=1, lang="en") == en_txt

    # 5. `runners/callable_runner.py`
    class _DummyEnv:
        environment_id = "dummy_env"

        def __init__(self) -> None:
            self._state: dict[str, Any] = {"ok": True}

        def get_state(self) -> dict[str, Any]:
            return dict(self._state)

        def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
            if initial_state:
                self._state.update(initial_state)
            return self.get_state()

        def cleanup(self) -> None:
            pass

        async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            if tool_name == "fail_tool":
                raise RuntimeError("boom")
            return {"tool": tool_name, **arguments}

    env = _DummyEnv()
    proxy = EnvironmentProxy(env)  # type: ignore[arg-type]
    assert proxy.environment_id == "dummy_env"
    proxy.set_state({"added": 1})
    assert proxy.get_state()["added"] == 1
    proxy.reset({"reset": 2})
    proxy.cleanup()
    with pytest.raises(RuntimeError):
        await proxy.execute_tool("fail_tool", {})

    task = Task(
        task_id="t_call",
        name="Callable Task",
        domain="pix_assist",
        description="desc",
        input_messages=[{"role": "user", "content": "Run tool"}],
        allowed_tools=["ok_tool"],
    )

    async def _fn_two(t: Task, e: EnvironmentProxy) -> dict[str, Any]:
        await e.execute_tool("ok_tool", {"x": 1})
        return {"response": f"done:{t.task_id}", "tokens_in": 5, "tokens_out": 5}

    async with CallableAgentRunner(_fn_two) as r2:
        res2, tr2 = await r2.run_task(task, env)  # type: ignore[arg-type]
        assert res2["response"] == "done:t_call"
        assert len(tr2) >= 3

    r_prompt = CallableAgentRunner(lambda prompt: f"p:{prompt}")
    res_p, _ = await r_prompt.run(task, env)  # type: ignore[arg-type]
    assert res_p["response"] == "p:Run tool"

    r_msgs = CallableAgentRunner(lambda messages, tools, env: f"m:{len(messages)}:{len(tools)}")
    res_m, _ = await r_msgs.run(task, env)  # type: ignore[arg-type]
    assert res_m["response"] == "m:1:1"

    def _raise_err(task: Task) -> str:
        raise ValueError("callable_error")

    r_err = CallableAgentRunner(_raise_err)
    res_e, _ = await r_err.run(task, env)  # type: ignore[arg-type]
    assert "ValueError: callable_error" in res_e["response"]
