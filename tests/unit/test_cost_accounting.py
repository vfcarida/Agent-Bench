"""Unit tests for AB-T04: Real cost/latency/token accounting in suite path (T-COST-1).

Verifies:
1. No hardcoded per-task constants (0.001, 50.0) in suite_runner.py.
2. CaseResult supports both typed attribute access and backwards-compatible tuple unpacking.
3. Per-model pricing resolution and documented fallback with warning log.
4. Scorecard latency percentiles (p50, p90, p99) and cost_per_successful_task calculation.
5. Verification that cost and latency scores vary across different agents (forbidden regression guard).
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_bench.core.adapters import ModelAdapter, ModelResponse
from agent_bench.core.config import BenchConfig, ModelConfig, SuiteConfig, SystemConfig
from agent_bench.core.scenarios import Task
from agent_bench.core.settings import settings
from agent_bench.metrics.scorecard import compute_scorecard
from agent_bench.runners.case_runner import CaseResult, DefaultAgentRunner, execute_task
from agent_bench.runners.suite_runner import run_suite


class ConfiguredTokenStubAdapter(ModelAdapter):
    """Stub adapter reporting exact deterministic token counts and latency."""

    def __init__(
        self,
        model_id: str = "stub",
        tokens_in: int = 200,
        tokens_out: int = 100,
        latency_ms: float = 150.0,
    ) -> None:
        self._model_id = model_id
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._latency_ms = latency_ms

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "stub"

    async def generate(self, messages, **kwargs) -> ModelResponse:
        return ModelResponse(
            content="Stub response text for cost accounting verification.",
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            latency_ms=self._latency_ms,
        )


def test_no_hardcoded_cost_latency_constants_in_suite_runner():
    """T-COST-1: Verify no literal 0.001 or 50.0 per-task constants remain in suite_runner.py."""
    suite_runner_path = Path("src/agent_bench/runners/suite_runner.py")
    assert suite_runner_path.exists()
    content = suite_runner_path.read_text(encoding="utf-8")

    assert "0.001" not in content, "Literal constant 0.001 found in suite_runner.py"
    assert "50.0" not in content, "Literal constant 50.0 found in suite_runner.py"


def test_case_result_attributes_and_tuple_unpacking():
    """Verify CaseResult backwards compatibility and field accessibility."""
    res = CaseResult(
        passed=True,
        traces=[],
        latency_ms=123.4,
        tokens_in=150,
        tokens_out=75,
        cost_usd=0.0035,
    )
    # Typed attribute access
    assert res.passed is True
    assert res.traces == []
    assert res.latency_ms == 123.4
    assert res.tokens_in == 150
    assert res.tokens_out == 75
    assert res.cost_usd == 0.0035

    # Tuple unpacking backward compatibility: passed, traces = result
    passed, traces = res
    assert passed is True
    assert traces == []
    assert len(res) == 2
    assert res[0] is True
    assert res[1] == []


@pytest.mark.asyncio
async def test_pricing_fallback_with_warning():
    """Verify that execute_task logs a warning and falls back to settings when pricing is absent."""
    task = Task(
        task_id="TEST_001",
        domain="pix_assist",
        name="Test Task 1",
        description="Test task for pricing fallback",
        input_messages=[{"role": "user", "content": "hello"}],
    )
    # Model without explicit pricing
    config = BenchConfig(
        models=[ModelConfig(model_id="unpriced_model", provider="stub")],
        systems=[SystemConfig(system_id="unpriced_sys", architecture="prompt_only", model="unpriced_model")],
        suites=[],
    )

    with patch("agent_bench.runners.case_runner.logger.warning") as mock_warn:
        adapter = ConfiguredTokenStubAdapter(tokens_in=1000, tokens_out=1000, latency_ms=100.0)
        runner = DefaultAgentRunner(system_id="unpriced_sys", model=adapter)
        result = await execute_task(task, "unpriced_sys", config, agent_runner=runner)

        # Warning was logged
        mock_warn.assert_called_once()
        call_args = mock_warn.call_args
        assert call_args[0][0] == "pricing_fallback_to_defaults"
        assert call_args[1]["cost_per_1k_input"] == settings.cost_per_1k_input_tokens
        assert call_args[1]["cost_per_1k_output"] == settings.cost_per_1k_output_tokens

        # Cost was computed using settings defaults: 1k * in + 1k * out
        expected_cost = settings.cost_per_1k_input_tokens + settings.cost_per_1k_output_tokens
        assert abs(result.cost_usd - expected_cost) < 1e-6


@pytest.mark.asyncio
async def test_configured_pricing_used_without_warning():
    """Verify configured per-model pricing is used without fallback warning."""
    task = Task(
        task_id="TEST_002",
        domain="pix_assist",
        name="Test Task 2",
        description="Test task with explicit pricing",
        input_messages=[{"role": "user", "content": "test message"}],
    )
    config = BenchConfig(
        models=[
            ModelConfig(
                model_id="custom_priced_model",
                provider="stub",
                price_per_1k_input=0.0050,
                price_per_1k_output=0.0150,
            )
        ],
        systems=[
            SystemConfig(
                system_id="custom_sys",
                architecture="prompt_only",
                model="custom_priced_model",
            )
        ],
        suites=[],
    )

    with patch("agent_bench.runners.case_runner.logger.warning") as mock_warn:
        adapter = ConfiguredTokenStubAdapter(tokens_in=2000, tokens_out=1000, latency_ms=250.0)
        runner = DefaultAgentRunner(system_id="custom_sys", model=adapter)
        result = await execute_task(task, "custom_sys", config, agent_runner=runner)

        # No fallback warning should be logged
        mock_warn.assert_not_called()

        # Expected cost: (2 * 0.0050) + (1 * 0.0150) = 0.0100 + 0.0150 = 0.0250
        assert abs(result.cost_usd - 0.0250) < 1e-6
        assert result.tokens_in == 2000
        assert result.tokens_out == 1000
        assert result.latency_ms == 250.0


@pytest.mark.asyncio
async def test_suite_path_measured_cost_and_scorecard_percentiles(tmp_path):
    """T-COST-1 integration: assert scorecard cost derives from trace logger and percentiles are emitted."""
    suite_cfg = SuiteConfig(
        suite_id="test_suite_cost",
        name="Test Suite Cost",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["test_system"],
        repeat_n=2,
        seed=42,
    )
    config = BenchConfig(
        models=[
            ModelConfig(
                model_id="priced_model",
                provider="stub",
                price_per_1k_input=0.0040,
                price_per_1k_output=0.0080,
            )
        ],
        systems=[
            SystemConfig(
                system_id="test_system",
                architecture="prompt_only",
                model="priced_model",
            )
        ],
        suites=[suite_cfg],
    )

    adapter = ConfiguredTokenStubAdapter(
        tokens_in=500,  # 0.5 * 0.0040 = 0.0020
        tokens_out=250, # 0.25 * 0.0080 = 0.0020 => cost per call = 0.0040
        latency_ms=120.0,
    )
    runner = DefaultAgentRunner(system_id="test_system", model=adapter)

    artifact = await run_suite(suite_cfg, config, tmp_path, agent_runner=runner)

    scorecards = artifact.metrics.get("scorecards", [])
    assert len(scorecards) == 1
    sc = scorecards[0]

    # Repeat_n = 2, so per task: 2 repetitions * 0.0040 = 0.0080 total task cost
    # Average task cost across all tasks in domain = 0.0080
    expected_cost_score = max(0.0, 1.0 - 0.0080)
    assert abs(sc["cost_score"] - expected_cost_score) < 1e-4

    # Assert latency percentiles are present in emitted scorecard
    assert "latency_p50" in sc
    assert "latency_p90" in sc
    assert "latency_p99" in sc
    assert sc["latency_p50"] == 120.0
    assert sc["latency_p90"] == 120.0
    assert sc["latency_p99"] == 120.0

    # Assert cost_per_successful_task is present in emitted scorecard
    assert "cost_per_successful_task" in sc
    assert sc["cost_per_successful_task"] > 0.0

    # Assert flat artifact JSON includes latency percentiles and cost_per_successful_task
    artifact_json_path = tmp_path / f"{artifact.run_id}.json"
    assert artifact_json_path.exists()
    saved_data = json.loads(artifact_json_path.read_text(encoding="utf-8"))
    saved_sc = saved_data["scorecards"][0]
    assert "latency_p50" in saved_sc
    assert "latency_p90" in saved_sc
    assert "latency_p99" in saved_sc
    assert "cost_per_successful_task" in saved_sc


def test_scorecard_percentile_computation():
    """Verify _percentile helper accuracy on varied distributions."""
    results = [
        {"passed": True, "cost_usd": 0.01, "latency_ms": 100.0, "repetitions": [True]},
        {"passed": True, "cost_usd": 0.02, "latency_ms": 200.0, "repetitions": [True]},
        {"passed": True, "cost_usd": 0.03, "latency_ms": 300.0, "repetitions": [True]},
        {"passed": False, "cost_usd": 0.04, "latency_ms": 400.0, "repetitions": [False]},
    ]
    sc = compute_scorecard("sys", "domain", results)
    # 4 items: p50 should be median (250.0), p90 should be near 370.0, p99 near 397.0
    assert sc.latency_p50 == 250.0
    assert sc.latency_p90 == 370.0
    assert sc.latency_p99 == 397.0

    # Cost per successful task: total = 0.10, passed = 3 => 0.10 / 3 = 0.0333...
    assert abs(sc.cost_per_successful_task - (0.10 / 3)) < 1e-4

    # Check metric list contains them
    metric_names = [m.name for m in sc.metrics]
    assert "latency_p50_ms" in metric_names
    assert "latency_p90_ms" in metric_names
    assert "latency_p99_ms" in metric_names
    assert "cost_per_successful_task" in metric_names


@pytest.mark.asyncio
async def test_cost_latency_scores_vary_across_agents(tmp_path):
    """Forbidden regression: cost/latency scores must not be constant across different agents."""
    suite_cfg = SuiteConfig(
        suite_id="test_suite_diff",
        name="Test Suite Diff",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["system_fast_cheap", "system_slow_expensive"],
        repeat_n=1,
        seed=42,
    )
    config = BenchConfig(
        models=[
            ModelConfig(
                model_id="cheap_model",
                provider="stub",
                price_per_1k_input=0.0010,
                price_per_1k_output=0.0020,
            ),
            ModelConfig(
                model_id="expensive_model",
                provider="stub",
                price_per_1k_input=0.0500,
                price_per_1k_output=0.1000,
            ),
        ],
        systems=[
            SystemConfig(system_id="system_fast_cheap", architecture="prompt_only", model="cheap_model"),
            SystemConfig(system_id="system_slow_expensive", architecture="prompt_only", model="expensive_model"),
        ],
        suites=[suite_cfg],
    )

    adapter_fast_cheap = ConfiguredTokenStubAdapter(
        tokens_in=50,
        tokens_out=25,
        latency_ms=100.0,
    )
    runner_fast_cheap = DefaultAgentRunner(system_id="system_fast_cheap", model=adapter_fast_cheap)

    adapter_slow_expensive = ConfiguredTokenStubAdapter(
        tokens_in=5000,
        tokens_out=2500,
        latency_ms=8000.0,
    )
    runner_slow_expensive = DefaultAgentRunner(system_id="system_slow_expensive", model=adapter_slow_expensive)

    # Run for both runners
    art_cheap = await run_suite(suite_cfg, config, tmp_path / "cheap", agent_runner=runner_fast_cheap)
    art_expensive = await run_suite(suite_cfg, config, tmp_path / "expensive", agent_runner=runner_slow_expensive)

    sc_cheap = art_cheap.metrics["scorecards"][0]
    sc_expensive = art_expensive.metrics["scorecards"][0]

    # Scores MUST differ
    assert sc_cheap["cost_score"] != sc_expensive["cost_score"]
    assert sc_cheap["latency_score"] != sc_expensive["latency_score"]
    assert sc_cheap["cost_score"] > sc_expensive["cost_score"]
    assert sc_cheap["latency_score"] > sc_expensive["latency_score"]
    assert sc_cheap["latency_p50"] < sc_expensive["latency_p50"]
    assert sc_cheap["cost_per_successful_task"] < sc_expensive["cost_per_successful_task"]
