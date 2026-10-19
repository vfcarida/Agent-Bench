"""Agent-Bench: Enterprise benchmark framework for evaluating LLM agents as complete systems."""

from agent_bench.core.adapters import (
    JudgeAdapter,
    JudgeVerdict,
    ModelAdapter,
    SafetyVerdict,
    ToolAdapter,
)
from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig, load_config
from agent_bench.core.metrics import MetricCategory, MetricResult, Scorecard
from agent_bench.core.protocols import AgentRunner, Evaluator, TaskEnvironment
from agent_bench.core.scenarios import RefusalMode, Severity, Task
from agent_bench.core.schema_v2 import Difficulty, EvalCase, Family, GradingStrategy, Split
from agent_bench.core.user_simulator import (
    HumanUserSimulator,
    RuleBasedUserSimulator,
    ScriptedUserSimulator,
    UserSimulator,
)
from agent_bench.graders.gated_evaluator import GatedEvaluator
from agent_bench.graders.safety_gate import evaluate_safety
from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k, compute_task_metrics
from agent_bench.metrics.expanded import (
    SPRTDecision,
    compute_bootstrap_ci,
    compute_wilson_score_interval,
    evaluate_wald_sprt,
)
from agent_bench.metrics.pareto import (
    AutonomyTier,
    classify_autonomy_tier,
    compute_hypervolume_2d,
    compute_pareto_frontier,
)
from agent_bench.metrics.scorecard import compute_balanced_profile, compute_scorecard
from agent_bench.runners.callable_runner import CallableAgentRunner
from agent_bench.runners.case_runner import DefaultAgentRunner, execute_task
from agent_bench.runners.scripted import ScriptedAgentRunner
from agent_bench.runners.suite_runner import run_suite

__version__ = "1.0.0"
__author__ = "Vinicius Caridá"

__all__ = [
    "__author__",
    "__version__",
    "AgentRunner",
    "AutonomyTier",
    "BenchConfig",
    "CallableAgentRunner",
    "DefaultAgentRunner",
    "Difficulty",
    "EvalCase",
    "Evaluator",
    "Family",
    "GatedEvaluator",
    "GradingStrategy",
    "HumanUserSimulator",
    "JudgeAdapter",
    "JudgeVerdict",
    "MetricCategory",
    "MetricResult",
    "ModelAdapter",
    "RefusalMode",
    "RuleBasedUserSimulator",
    "RunArtifact",
    "SPRTDecision",
    "SafetyVerdict",
    "Scorecard",
    "ScriptedAgentRunner",
    "ScriptedUserSimulator",
    "Severity",
    "Split",
    "SuiteConfig",
    "SystemConfig",
    "Task",
    "TaskEnvironment",
    "ToolAdapter",
    "TraceEvent",
    "TraceEventType",
    "UserSimulator",
    "classify_autonomy_tier",
    "compute_balanced_profile",
    "compute_bootstrap_ci",
    "compute_hypervolume_2d",
    "compute_pareto_frontier",
    "compute_pass_hat_k",
    "compute_pass_k",
    "compute_scorecard",
    "compute_task_metrics",
    "compute_wilson_score_interval",
    "evaluate_safety",
    "evaluate_wald_sprt",
    "execute_task",
    "load_config",
    "run_suite",
]

