"""Metric computation utilities."""

from agent_bench.metrics.adaptive_cat import (
    BayesianCATEngine,
    CATSessionReport,
    CATStepTelemetry,
)
from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k, compute_task_metrics
from agent_bench.metrics.difficulty_analysis import (
    compare_difficulty_robustness,
    compute_difficulty_degradation,
    compute_difficulty_profile,
)
from agent_bench.metrics.discriminative import (
    compute_discriminative_power,
    compute_per_domain_discriminative_power,
)
from agent_bench.metrics.expanded import (
    SPRTDecision,
    compute_bootstrap_ci,
    compute_stratified_bootstrap_ci,
    compute_wilson_score_interval,
    evaluate_wald_sprt,
)
from agent_bench.metrics.irt import (
    IRTItemParameters,
    IRTModelFit,
    estimate_full_score_from_anchor,
    fisher_information_2pl,
    fit_2pl_irt,
    logistic_2pl,
    select_irt_anchor_tasks,
)
from agent_bench.metrics.pareto import (
    AutonomyTier,
    classify_autonomy_tier,
    compute_hypervolume_2d,
    compute_pareto_frontier,
)
from agent_bench.metrics.report import MetricsReport, build_metrics_report
from agent_bench.metrics.scorecard import compute_balanced_profile, compute_scorecard
from agent_bench.metrics.transfer_analysis import (
    compute_system_transfer_profiles,
    compute_transfer_gaps,
    rank_by_transfer_robustness,
)

__all__ = [
    "AutonomyTier",
    "BayesianCATEngine",
    "CATSessionReport",
    "CATStepTelemetry",
    "IRTItemParameters",
    "IRTModelFit",
    "MetricsReport",
    "SPRTDecision",
    "build_metrics_report",
    "classify_autonomy_tier",
    "compare_difficulty_robustness",
    "compute_balanced_profile",
    "compute_bootstrap_ci",
    "compute_difficulty_degradation",
    "compute_difficulty_profile",
    "compute_discriminative_power",
    "compute_hypervolume_2d",
    "compute_pareto_frontier",
    "compute_pass_hat_k",
    "compute_pass_k",
    "compute_per_domain_discriminative_power",
    "compute_scorecard",
    "compute_stratified_bootstrap_ci",
    "compute_system_transfer_profiles",
    "compute_task_metrics",
    "compute_transfer_gaps",
    "compute_wilson_score_interval",
    "estimate_full_score_from_anchor",
    "evaluate_wald_sprt",
    "fisher_information_2pl",
    "fit_2pl_irt",
    "logistic_2pl",
    "rank_by_transfer_robustness",
    "select_irt_anchor_tasks",
]
