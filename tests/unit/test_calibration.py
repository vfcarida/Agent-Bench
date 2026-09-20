"""Unit tests for judges/calibration.py (offline, no LLM calls)."""

import pytest

from agent_bench.judges.calibration import (
    CALIBRATION_ACCURACY_THRESHOLD,
    CALIBRATION_KAPPA_THRESHOLD,
    LENGTH_BIAS_THRESHOLD,
    POSITION_BIAS_THRESHOLD,
    CalibrationReport,
    CalibrationSample,
    compute_cohens_kappa,
    compute_length_bias,
    compute_position_bias,
    compute_simple_accuracy,
    run_calibration_experiment,
)
from agent_bench.judges.semantic import SemanticJudge
from agent_bench.models.stub import StubModelAdapter

# ---------- Agreement metric tests ----------

class TestSimpleAccuracy:
    def test_perfect_agreement(self):
        j = [0.8, 0.6, 0.9, 0.4]
        r = [0.8, 0.6, 0.9, 0.4]
        assert compute_simple_accuracy(j, r) == 1.0

    def test_within_tolerance(self):
        j = [0.8, 0.6]
        r = [0.75, 0.65]  # both within 0.1
        assert compute_simple_accuracy(j, r) == 1.0

    def test_outside_tolerance(self):
        j = [0.9, 0.1]
        r = [0.0, 1.0]  # both outside 0.1
        assert compute_simple_accuracy(j, r) == 0.0

    def test_partial_agreement(self):
        j = [0.8, 0.5]
        r = [0.8, 0.0]  # first matches, second doesn't
        assert compute_simple_accuracy(j, r) == 0.5

    def test_empty_list(self):
        assert compute_simple_accuracy([], []) == 0.0


class TestCohensKappa:
    def test_perfect_agreement(self):
        # All scores in PASS bin (>= 0.7)
        j = [0.8, 0.9, 0.7]
        r = [0.8, 0.9, 0.7]
        kappa = compute_cohens_kappa(j, r)
        assert kappa == pytest.approx(1.0, abs=1e-6)

    def test_zero_agreement(self):
        # Systematic disagreement: judge always PASS, reference always FAIL
        # With all predictions in one bin, kappa formula gives <= 0.0
        j = [0.9, 0.9, 0.9]   # all PASS
        r = [0.1, 0.1, 0.1]   # all FAIL
        kappa = compute_cohens_kappa(j, r)
        assert kappa <= 0.0  # at or below chance

    def test_moderate_agreement(self):
        j = [0.9, 0.5, 0.1, 0.8]
        r = [0.9, 0.6, 0.2, 0.3]
        kappa = compute_cohens_kappa(j, r)
        # Exact value depends on distribution; just check it's a valid float
        assert isinstance(kappa, float)
        assert -1.0 <= kappa <= 1.0

    def test_empty_list(self):
        assert compute_cohens_kappa([], []) == 0.0

    def test_mismatched_lengths(self):
        assert compute_cohens_kappa([0.5], [0.5, 0.6]) == 0.0


# ---------- Bias probe tests ----------

class TestPositionBias:
    def test_no_bias(self):
        mean_diff, flagged = compute_position_bias([0.8, 0.7], [0.8, 0.7])
        assert mean_diff == pytest.approx(0.0)
        assert flagged is False

    def test_high_bias(self):
        # Large score swings when order is reversed
        mean_diff, flagged = compute_position_bias([0.9, 0.9], [0.5, 0.5])
        assert mean_diff == pytest.approx(0.4)
        assert flagged is True

    def test_at_threshold(self):
        _, flagged = compute_position_bias(
            [POSITION_BIAS_THRESHOLD], [0.0]
        )
        assert flagged is True

    def test_empty(self):
        mean_diff, flagged = compute_position_bias([], [])
        assert mean_diff == 0.0
        assert flagged is False


class TestLengthBias:
    def test_no_bias(self):
        mean_diff, flagged = compute_length_bias([0.7, 0.8], [0.7, 0.8])
        assert mean_diff == pytest.approx(0.0)
        assert flagged is False

    def test_high_bias(self):
        mean_diff, flagged = compute_length_bias([0.5, 0.5], [0.9, 0.9])
        assert mean_diff == pytest.approx(0.4)
        assert flagged is True

    def test_empty(self):
        mean_diff, flagged = compute_length_bias([], [])
        assert mean_diff == 0.0
        assert flagged is False


# ---------- CalibrationReport tests ----------

class TestCalibrationReport:
    def _make_report(
        self,
        accuracy: float,
        kappa: float,
        pos_bias: float = 0.0,
        len_bias: float = 0.0,
    ) -> CalibrationReport:
        report = CalibrationReport()
        report.n_samples = 5
        report.simple_accuracy = accuracy
        report.cohens_kappa = kappa
        report.position_bias_mean = pos_bias
        report.position_bias_flagged = pos_bias >= POSITION_BIAS_THRESHOLD
        report.length_bias_mean = len_bias
        report.length_bias_flagged = len_bias >= LENGTH_BIAS_THRESHOLD
        return report

    def test_go_decision_when_all_pass(self):
        report = self._make_report(
            accuracy=CALIBRATION_ACCURACY_THRESHOLD + 0.05,
            kappa=CALIBRATION_KAPPA_THRESHOLD + 0.1,
        )
        report.compute_decision()
        assert report.go_decision == "GO"

    def test_no_go_low_accuracy(self):
        report = self._make_report(
            accuracy=CALIBRATION_ACCURACY_THRESHOLD - 0.1,
            kappa=CALIBRATION_KAPPA_THRESHOLD + 0.1,
        )
        report.compute_decision()
        assert report.go_decision == "NO_GO"
        assert any("simple_accuracy" in r for r in report.decision_reasons)

    def test_no_go_low_kappa(self):
        report = self._make_report(
            accuracy=CALIBRATION_ACCURACY_THRESHOLD + 0.05,
            kappa=CALIBRATION_KAPPA_THRESHOLD - 0.1,
        )
        report.compute_decision()
        assert report.go_decision == "NO_GO"
        assert any("kappa" in r for r in report.decision_reasons)

    def test_no_go_position_bias(self):
        report = self._make_report(
            accuracy=CALIBRATION_ACCURACY_THRESHOLD + 0.05,
            kappa=CALIBRATION_KAPPA_THRESHOLD + 0.1,
            pos_bias=POSITION_BIAS_THRESHOLD + 0.05,
        )
        report.compute_decision()
        assert report.go_decision == "NO_GO"
        assert any("position_bias" in r for r in report.decision_reasons)

    def test_no_go_length_bias(self):
        report = self._make_report(
            accuracy=CALIBRATION_ACCURACY_THRESHOLD + 0.05,
            kappa=CALIBRATION_KAPPA_THRESHOLD + 0.1,
            len_bias=LENGTH_BIAS_THRESHOLD + 0.05,
        )
        report.compute_decision()
        assert report.go_decision == "NO_GO"
        assert any("length_bias" in r for r in report.decision_reasons)

    def test_summary_text(self):
        report = self._make_report(0.80, 0.55)
        report.compute_decision()
        text = report.summary_text
        assert "GO" in text or "NO_GO" in text
        assert "samples" in text


# ---------- Integration: run_calibration_experiment (stub model) ----------

@pytest.mark.asyncio
async def test_run_calibration_experiment_with_stub():
    """Offline smoke test: runs the full experiment harness using a stub judge."""
    stub_response = (
        '{"correctness": 0.8, "completeness": 0.8, "safety": 1.0, '
        '"helpfulness": 0.8, "tone": 0.9, "aggregate": 0.82, '
        '"reasoning": "Solid response that meets all criteria."}'
    )
    stub = StubModelAdapter(responses=[stub_response])
    judge = SemanticJudge(model=stub)

    task_snapshot = {
        "task_id": "calib_001",
        "domain": "pix_assist",
        "description": "How do I check my balance?",
        "input_messages": [{"role": "user", "content": "How do I check my balance?"}],
        "required_capabilities": [],
        "gold_references": ["Open the app and go to Home tab."],
        "expected_refusal_mode": "none",
    }

    samples = [
        CalibrationSample(
            sample_id="calib_001",
            task_snapshot=task_snapshot,
            execution_result={"response": "Go to the Home tab in the app."},
            reference_label=0.85,
        ),
        CalibrationSample(
            sample_id="calib_002",
            task_snapshot=task_snapshot,
            execution_result={"response": "I don't know."},
            reference_label=0.2,
        ),
    ]

    report = await run_calibration_experiment(judge, samples)

    assert report.n_samples == 2
    assert len(report.judge_scores) == 2
    assert len(report.reference_scores) == 2
    assert report.go_decision in ("GO", "NO_GO")
    assert isinstance(report.simple_accuracy, float)
    assert isinstance(report.cohens_kappa, float)
    # Stub always returns 0.82 → first sample correct, second wrong
    # Accuracy depends on tolerance comparison
    assert 0.0 <= report.simple_accuracy <= 1.0
