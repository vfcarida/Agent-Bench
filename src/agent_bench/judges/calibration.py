"""Calibration harness for LLM-as-judge (AB-T08, EXPERIMENT lane).

Purpose
-------
Before the SemanticJudge may be used for headline quality scores, it must
pass a calibration experiment that demonstrates:
  (a) agreement with reference labels (Cohen's κ or simple accuracy);
  (b) absence of systematic bias (score inflation vs. a rule-based baseline).

This module provides the data structures and computation primitives for that
experiment.  Execution requires network access (credentialed lane) because it
calls a real LLM.  The offline CI lane can only run the harness in MOCK mode
using a stub model.

GO/NO-GO decision
-----------------
The calibration experiment must produce a report.  A human reviewer inspects
the report and records a GO or NO-GO verdict in
``docs/llm_judge_calibration_report.md`` before ``--enable-llm-judge`` is
allowed in production evaluation runs.

Bias probes
-----------
Two systematic bias probes are included:
  1. Position bias: judge twice with (response_a vs response_b) and
     (response_b vs response_a); a consistent judge should give the same
     relative ordering.
  2. Length bias: compare scores for short vs long responses that contain
     identical content.  Score difference > BIAS_THRESHOLD flags a bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_bench.judges.semantic import SemanticJudge

# ---------- Thresholds ----------

# Minimum simple-accuracy (exact agreement within ±0.1) for GO decision.
CALIBRATION_ACCURACY_THRESHOLD = 0.70

# Minimum Cohen's κ for GO decision.
CALIBRATION_KAPPA_THRESHOLD = 0.40  # moderate agreement

# Maximum allowed score difference between position-swapped pairs.
POSITION_BIAS_THRESHOLD = 0.15

# Maximum allowed score difference between length-varied pairs with same content.
LENGTH_BIAS_THRESHOLD = 0.20


# ---------- Data structures ----------

@dataclass
class CalibrationSample:
    """A single labelled sample for calibration."""

    sample_id: str
    task_snapshot: dict[str, Any]  # minimal task-like dict
    execution_result: dict[str, Any]  # agent result to be judged
    reference_label: float  # gold label in [0, 1]
    notes: str = ""


@dataclass
class CalibrationReport:
    """Aggregated results of a calibration experiment run."""

    n_samples: int = 0
    judge_scores: list[float] = field(default_factory=list)
    reference_scores: list[float] = field(default_factory=list)

    # Agreement metrics
    simple_accuracy: float = 0.0   # fraction within ±0.1
    cohens_kappa: float = 0.0

    # Bias probe results
    position_bias_mean: float = 0.0
    position_bias_flagged: bool = False
    length_bias_mean: float = 0.0
    length_bias_flagged: bool = False

    # GO/NO-GO
    go_decision: str = "NOT_RUN"   # NOT_RUN | GO | NO_GO
    decision_reasons: list[str] = field(default_factory=list)

    def compute_decision(self) -> None:
        """Derive the GO/NO-GO decision from the accumulated metrics."""
        reasons: list[str] = []
        go = True

        if self.simple_accuracy < CALIBRATION_ACCURACY_THRESHOLD:
            go = False
            reasons.append(
                f"simple_accuracy={self.simple_accuracy:.3f} < {CALIBRATION_ACCURACY_THRESHOLD}"
            )

        if self.cohens_kappa < CALIBRATION_KAPPA_THRESHOLD:
            go = False
            reasons.append(
                f"cohens_kappa={self.cohens_kappa:.3f} < {CALIBRATION_KAPPA_THRESHOLD}"
            )

        if self.position_bias_flagged:
            go = False
            reasons.append(
                f"position_bias={self.position_bias_mean:.3f} >= {POSITION_BIAS_THRESHOLD}"
            )

        if self.length_bias_flagged:
            go = False
            reasons.append(
                f"length_bias={self.length_bias_mean:.3f} >= {LENGTH_BIAS_THRESHOLD}"
            )

        self.go_decision = "GO" if go else "NO_GO"
        self.decision_reasons = reasons if not go else ["All thresholds met."]

    @property
    def summary_text(self) -> str:
        lines = [
            f"Calibration report — {self.n_samples} samples",
            f"  Simple accuracy (±0.1): {self.simple_accuracy:.3f} (threshold ≥ {CALIBRATION_ACCURACY_THRESHOLD})",
            f"  Cohen's κ:              {self.cohens_kappa:.3f} (threshold ≥ {CALIBRATION_KAPPA_THRESHOLD})",
            f"  Position bias (mean Δ): {self.position_bias_mean:.3f} (flagged: {self.position_bias_flagged})",
            f"  Length bias (mean Δ):   {self.length_bias_mean:.3f} (flagged: {self.length_bias_flagged})",
            f"  Decision:               {self.go_decision}",
        ]
        if self.decision_reasons:
            lines.append("  Reasons:")
            for r in self.decision_reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


# ---------- Agreement metrics ----------

def compute_simple_accuracy(
    judge_scores: list[float],
    reference_scores: list[float],
    tolerance: float = 0.1,
) -> float:
    """Fraction of pairs where |judge - reference| <= tolerance."""
    if not judge_scores:
        return 0.0
    matches = sum(
        1 for j, r in zip(judge_scores, reference_scores) if abs(j - r) <= tolerance
    )
    return matches / len(judge_scores)


def compute_cohens_kappa(
    judge_scores: list[float],
    reference_scores: list[float],
    thresholds: tuple[float, float] = (0.4, 0.7),
) -> float:
    """Cohen's κ computed over ordinal bins: FAIL / PARTIAL / PASS.

    Bins:
      score < thresholds[0]  → 0 (FAIL)
      thresholds[0] ≤ score < thresholds[1] → 1 (PARTIAL)
      score >= thresholds[1] → 2 (PASS)
    """
    if len(judge_scores) != len(reference_scores) or not judge_scores:
        return 0.0

    n_classes = 3

    def _bin(score: float) -> int:
        if score < thresholds[0]:
            return 0
        elif score < thresholds[1]:
            return 1
        return 2

    j_bins = [_bin(s) for s in judge_scores]
    r_bins = [_bin(s) for s in reference_scores]
    n = len(j_bins)

    # Confusion matrix
    conf: list[list[int]] = [[0] * n_classes for _ in range(n_classes)]
    for j, r in zip(j_bins, r_bins):
        conf[j][r] += 1

    p_observed = sum(conf[i][i] for i in range(n_classes)) / n
    p_expected = sum(
        (sum(conf[i]) / n) * (sum(conf[k][i] for k in range(n_classes)) / n)
        for i in range(n_classes)
    )

    if math.isclose(p_expected, 1.0):
        return 1.0

    return (p_observed - p_expected) / (1.0 - p_expected)


# ---------- Bias probes ----------

def compute_position_bias(
    scores_ab: list[float],
    scores_ba: list[float],
) -> tuple[float, bool]:
    """Probe for position bias using pairwise score-difference.

    Args:
        scores_ab: Judge score for each pair in (A, B) presentation order.
        scores_ba: Judge score for same pairs in (B, A) presentation order.

    Returns:
        (mean_abs_diff, flagged)
    """
    if not scores_ab:
        return 0.0, False
    diffs = [abs(a - b) for a, b in zip(scores_ab, scores_ba)]
    mean_diff = sum(diffs) / len(diffs)
    return mean_diff, mean_diff >= POSITION_BIAS_THRESHOLD


def compute_length_bias(
    scores_short: list[float],
    scores_long: list[float],
) -> tuple[float, bool]:
    """Probe for length bias: identical-content responses of different lengths.

    Args:
        scores_short: Judge scores for short versions.
        scores_long: Judge scores for long versions (same content, padded).

    Returns:
        (mean_abs_diff, flagged)
    """
    if not scores_short:
        return 0.0, False
    diffs = [abs(s - lo) for s, lo in zip(scores_short, scores_long)]
    mean_diff = sum(diffs) / len(diffs)
    return mean_diff, mean_diff >= LENGTH_BIAS_THRESHOLD


# ---------- High-level runner ----------

async def run_calibration_experiment(
    judge: SemanticJudge,
    samples: list[CalibrationSample],
    position_bias_pairs: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
    length_bias_pairs: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> CalibrationReport:
    """Run the full calibration experiment against a set of labelled samples.

    Args:
        judge: SemanticJudge instance (must be connected to a live model).
        samples: Labelled calibration cases.
        position_bias_pairs: Optional (result_a, result_b) pairs for position-bias probe.
        length_bias_pairs: Optional (short_result, long_result) pairs for length-bias probe.

    Returns:
        CalibrationReport with computed metrics and GO/NO-GO decision.
    """

    report = CalibrationReport(n_samples=len(samples))

    # --- Agreement experiment ---
    judge_scores: list[float] = []
    reference_scores: list[float] = []

    for sample in samples:
        # Construct a minimal Task-like object for the judge.
        # CalibrationSample.task_snapshot must satisfy Task's required fields.
        task = _build_task_from_snapshot(sample.task_snapshot)
        verdict = await judge.evaluate(task, sample.execution_result, [])
        judge_scores.append(verdict.score)
        reference_scores.append(sample.reference_label)

    report.judge_scores = judge_scores
    report.reference_scores = reference_scores
    report.simple_accuracy = compute_simple_accuracy(judge_scores, reference_scores)
    report.cohens_kappa = compute_cohens_kappa(judge_scores, reference_scores)

    # --- Position bias probe ---
    if position_bias_pairs:
        ab_scores: list[float] = []
        ba_scores: list[float] = []
        for result_a, result_b in position_bias_pairs:
            # Use the first sample's task as a stub task for bias probes
            task = _build_task_from_snapshot(samples[0].task_snapshot)
            v_ab = await judge.evaluate(task, result_a, [])
            v_ba = await judge.evaluate(task, result_b, [])
            ab_scores.append(v_ab.score)
            ba_scores.append(v_ba.score)
        report.position_bias_mean, report.position_bias_flagged = compute_position_bias(
            ab_scores, ba_scores
        )

    # --- Length bias probe ---
    if length_bias_pairs:
        short_scores: list[float] = []
        long_scores: list[float] = []
        for short_result, long_result in length_bias_pairs:
            task = _build_task_from_snapshot(samples[0].task_snapshot)
            v_short = await judge.evaluate(task, short_result, [])
            v_long = await judge.evaluate(task, long_result, [])
            short_scores.append(v_short.score)
            long_scores.append(v_long.score)
        report.length_bias_mean, report.length_bias_flagged = compute_length_bias(
            short_scores, long_scores
        )

    report.compute_decision()
    return report


def _build_task_from_snapshot(snapshot: dict[str, Any]) -> Any:
    """Construct a minimal Task from a dict snapshot for calibration probes."""
    from agent_bench.core.scenarios import RefusalMode, Task

    return Task(
        task_id=snapshot.get("task_id", "calibration_probe"),
        domain=snapshot.get("domain", "unknown"),
        name=snapshot.get("name", "Calibration probe"),
        description=snapshot.get("description", "Calibration probe task."),
        input_messages=snapshot.get("input_messages", [{"role": "user", "content": "Evaluate this."}]),
        expected_final_state=dict(snapshot["expected_final_state"]) if snapshot.get("expected_final_state") else {},
        allowed_tools=snapshot.get("allowed_tools", []),
        required_capabilities=snapshot.get("required_capabilities", []),
        gold_references=snapshot.get("gold_references", []),
        expected_refusal_mode=RefusalMode(snapshot.get("expected_refusal_mode", "none")),
    )
