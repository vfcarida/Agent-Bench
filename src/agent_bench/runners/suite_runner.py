"""Suite runner: executes all tasks in a benchmark suite."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.artifacts import RunArtifact
from agent_bench.core.config import BenchConfig, SuiteConfig
from agent_bench.core.protocols import AgentRunner
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k
from agent_bench.metrics.expanded import compute_bootstrap_ci
from agent_bench.metrics.scorecard import compute_scorecard
from agent_bench.runners.case_runner import DefaultAgentRunner, execute_task
from agent_bench.runners.scripted import ScriptedAgentRunner
from agent_bench.storage.jsonl import save_metrics_jsonl, save_run_manifest, save_traces_jsonl
from agent_bench.storage.parquet import save_metrics_parquet
from agent_bench.utils.observability import SpanCollector

logger = structlog.get_logger()


async def run_suite(
    suite_cfg: SuiteConfig,
    config: BenchConfig,
    output_dir: Path,
    runner_type: str = "auto",
    agent_runner: AgentRunner | None = None,
    enable_llm_judge: bool = False,
    llm_judge_system_id: str | None = None,
) -> RunArtifact:
    """Run all tasks in a suite, repeating N times for pass@k.

    Args:
        suite_cfg: Suite configuration.
        config: Full benchmark configuration.
        output_dir: Directory to write run artefacts.
        runner_type: One of ``auto`` / ``scripted`` / ``stub``.
        agent_runner: Optionally injected runner (overrides runner_type).
        enable_llm_judge: When True, adds SemanticJudge to GatedEvaluator
            Phase 2.  EXPERIMENTAL — only enable after a GO calibration verdict.
        llm_judge_system_id: System ID whose model is used as the judge.
            Required when enable_llm_judge is True.
    """
    collector = SpanCollector()

    with collector.trace("suite_run", suite_id=suite_cfg.suite_id) as suite_span:
        artifact = RunArtifact(
            suite_id=suite_cfg.suite_id,
            config_hash=config.config_hash(),
            benchmark_version=suite_cfg.version,
            seed=suite_cfg.seed,
        )

        all_tasks = []
        for domain_id in suite_cfg.domains:
            tasks = load_domain_tasks(domain_id, split=suite_cfg.split)
            all_tasks.extend(tasks)

        artifact.tasks_total = len(all_tasks) * len(suite_cfg.systems) * suite_cfg.repeat_n
        passed = 0
        failed = 0
        all_metric_records: list[dict[str, Any]] = []
        all_task_results: dict[str, list[dict[str, Any]]] = {}
        pass_k_results: list[dict[str, Any]] = []

        for system_id in suite_cfg.systems:
            system_results: list[dict[str, Any]] = []
            if agent_runner is not None:
                active_runner = agent_runner
            elif runner_type == "scripted":
                active_runner = ScriptedAgentRunner(system_id=system_id)
            elif runner_type == "stub":
                from agent_bench.models.stub import StubModelAdapter

                active_runner = DefaultAgentRunner(
                    system_id=system_id, model=StubModelAdapter(model_id="stub")
                )
            else:
                from agent_bench.models.factory import build_agent_runner

                active_runner = build_agent_runner(system_id, config)

            # Build optional SemanticJudge for Phase 2
            suite_evaluator = None
            if enable_llm_judge and llm_judge_system_id:
                from agent_bench.core.adapters import JudgeAdapter
                from agent_bench.graders.gated_evaluator import GatedEvaluator
                from agent_bench.judges.semantic import SemanticJudge
                from agent_bench.models.factory import build_agent_runner, build_model_adapter

                judge_sys_cfg = next(
                    (s for s in config.systems if s.system_id == llm_judge_system_id), None
                )
                if judge_sys_cfg is None:
                    raise ValueError(
                        f"llm_judge_system_id '{llm_judge_system_id}' not found in config.systems"
                    )
                judge_model_cfg = next(
                    (m for m in config.models if m.model_id == judge_sys_cfg.model), None
                )
                if judge_model_cfg is None:
                    raise ValueError(
                        f"Model '{judge_sys_cfg.model}' for judge system not found in config.models"
                    )
                judge_model = build_model_adapter(judge_model_cfg)
                semantic_judge: JudgeAdapter = SemanticJudge(model=judge_model)
                suite_evaluator = GatedEvaluator(subjective_judges=[semantic_judge])
                logger.info(
                    "llm_judge_enabled",
                    judge_system=llm_judge_system_id,
                    judge_id=semantic_judge.judge_id,
                )
            with collector.trace("system_eval", system_id=system_id):
                for task in all_tasks:
                    repetition_results: list[bool] = []
                    repetition_safety_violated: list[bool] = []
                    repetition_latencies: list[float] = []
                    repetition_tokens_in: list[int] = []
                    repetition_tokens_out: list[int] = []
                    repetition_costs: list[float] = []

                    with collector.trace("task_eval", task_id=task.task_id, domain=task.domain):
                        for i in range(suite_cfg.repeat_n):
                            seed = (suite_cfg.seed or 0) + i if suite_cfg.seed is not None else None
                            case_res = await execute_task(
                                task, system_id, config, seed=seed,
                                agent_runner=active_runner,
                                evaluator=suite_evaluator,
                            )
                            repetition_results.append(case_res.passed)
                            repetition_safety_violated.append(case_res.safety_violated)
                            repetition_latencies.append(case_res.latency_ms)
                            repetition_tokens_in.append(case_res.tokens_in)
                            repetition_tokens_out.append(case_res.tokens_out)
                            repetition_costs.append(case_res.cost_usd)
                            artifact.traces.extend(case_res.traces)

                    task_safety_violated = any(repetition_safety_violated)
                    task_passed = any(repetition_results) and not task_safety_violated
                    if task_passed:
                        passed += 1
                    else:
                        failed += 1

                    n_reps = len(repetition_results)
                    mean_latency = sum(repetition_latencies) / n_reps if n_reps else 0.0
                    mean_tokens_in = round(sum(repetition_tokens_in) / n_reps) if n_reps else 0
                    mean_tokens_out = round(sum(repetition_tokens_out) / n_reps) if n_reps else 0
                    total_task_cost = sum(repetition_costs)

                    task_result = {
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "passed": task_passed,
                        "policy_violated": task_safety_violated,
                        "safety_violation": task_safety_violated,
                        "latency_ms": mean_latency,
                        "latencies_ms": repetition_latencies,
                        "tokens_in": mean_tokens_in,
                        "tokens_out": mean_tokens_out,
                        "cost_usd": total_task_cost,
                        "repetitions": repetition_results,
                    }
                    system_results.append(task_result)

                    # Metric records
                    all_metric_records.append({
                        "run_id": artifact.run_id,
                        "system_id": system_id,
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "metric_name": "task_success",
                        "metric_value": 1.0 if task_passed else 0.0,
                        "metric_category": "functional",
                        "passed": task_passed,
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                    all_metric_records.append({
                        "run_id": artifact.run_id,
                        "system_id": system_id,
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "metric_name": "latency_ms",
                        "metric_value": mean_latency,
                        "metric_category": "latency",
                        "passed": task_passed,
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                    all_metric_records.append({
                        "run_id": artifact.run_id,
                        "system_id": system_id,
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "metric_name": "cost_usd",
                        "metric_value": total_task_cost,
                        "metric_category": "cost",
                        "passed": task_passed,
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

            all_task_results[system_id] = system_results

            # Compute per-task pass@k and pass^k per system per domain (unpooled aggregation)
            domains_in = {r["domain"] for r in system_results}
            for domain in domains_in:
                domain_results = [r for r in system_results if r["domain"] == domain]
                seed = suite_cfg.seed if suite_cfg.seed is not None else 42

                task_p1: list[float] = []
                task_p3: list[float] = []
                task_p5: list[float] = []
                task_phat1: list[float] = []
                task_phat3: list[float] = []
                task_phat5: list[float] = []
                task_func: list[float] = []

                for r in domain_results:
                    reps = r.get("repetitions", [r.get("passed", False)])
                    task_k = len(reps)
                    task_p1.append(compute_pass_k(reps, min(1, task_k)))
                    task_p3.append(compute_pass_k(reps, min(3, task_k)))
                    task_p5.append(compute_pass_k(reps, min(5, task_k)))
                    task_phat1.append(compute_pass_hat_k(reps, min(1, task_k)))
                    task_phat3.append(compute_pass_hat_k(reps, min(3, task_k)))
                    task_phat5.append(compute_pass_hat_k(reps, min(5, task_k)))
                    task_func.append(1.0 if r.get("passed", False) else 0.0)

                mean_p1, p1_low, p1_high = compute_bootstrap_ci(task_p1, seed=seed)
                mean_p3, p3_low, p3_high = compute_bootstrap_ci(task_p3, seed=seed)
                mean_p5, p5_low, p5_high = compute_bootstrap_ci(task_p5, seed=seed)
                mean_phat1, phat1_low, phat1_high = compute_bootstrap_ci(task_phat1, seed=seed)
                mean_phat3, phat3_low, phat3_high = compute_bootstrap_ci(task_phat3, seed=seed)
                mean_phat5, phat5_low, phat5_high = compute_bootstrap_ci(task_phat5, seed=seed)
                mean_func, func_low, func_high = compute_bootstrap_ci(task_func, seed=seed)

                pass_k_results.append({
                    "system_id": system_id,
                    "domain": domain,
                    "pass_1": mean_p1,
                    "pass_1_ci": [p1_low, p1_high],
                    "pass_3": mean_p3,
                    "pass_3_ci": [p3_low, p3_high],
                    "pass_5": mean_p5,
                    "pass_5_ci": [p5_low, p5_high],
                    "pass_hat_1": mean_phat1,
                    "pass_hat_1_ci": [phat1_low, phat1_high],
                    "pass_hat_3": mean_phat3,
                    "pass_hat_3_ci": [phat3_low, phat3_high],
                    "pass_hat_5": mean_phat5,
                    "pass_hat_5_ci": [phat5_low, phat5_high],
                    "functional_pass_rate": mean_func,
                    "functional_ci": [func_low, func_high],
                    "repeat_n": suite_cfg.repeat_n,
                    "total_tasks": len(domain_results),
                    "sampling_denominator": "all_trials_including_failures_timeouts",
                })

        artifact.system_id = ",".join(suite_cfg.systems)
        artifact.tasks_passed = passed
        artifact.tasks_failed = failed

        # Scorecards
        scorecards = []
        for system_id, results in all_task_results.items():
            domains_in_results = {r["domain"] for r in results}
            for domain in domains_in_results:
                domain_results = [r for r in results if r["domain"] == domain]
                sc = compute_scorecard(
                    system_id, domain, domain_results, suite_cfg.weighting_profile
                )
                scorecards.append({
                    "system_id": system_id,
                    "domain": domain,
                    "functional_score": sc.functional_score,
                    "functional_ci": [sc.functional_ci[0], sc.functional_ci[1]] if sc.functional_ci else None,
                    "risk_score": sc.risk_score,
                    "cost_score": sc.cost_score,
                    "latency_score": sc.latency_score,
                    "reliability_score": sc.reliability_score,
                    "reliability_ci": [sc.reliability_ci[0], sc.reliability_ci[1]] if sc.reliability_ci else None,
                    "global_score": sc.global_score,
                    "safety_violations": sc.safety_violations,
                    "safety_gated": sc.safety_gated,
                    "latency_p50": sc.latency_p50,
                    "latency_p90": sc.latency_p90,
                    "latency_p99": sc.latency_p99,
                    "cost_per_successful_task": sc.cost_per_successful_task,
                    "pass_hat_3": sc.pass_hat_3,
                    "pass_at_3": sc.pass_at_3,
                    "weighting_profile": suite_cfg.weighting_profile,
                    "sampling_denominator": "all_trials_including_failures_timeouts",
                })

        artifact.metrics["scorecards"] = scorecards
        artifact.metrics["pass_k"] = pass_k_results
        artifact.finalize()
        suite_span.set_attribute("tasks_passed", passed)
        suite_span.set_attribute("tasks_failed", failed)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / artifact.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_manifest(artifact, run_dir)
    save_traces_jsonl(artifact.traces, run_dir / "traces.jsonl")
    save_metrics_jsonl(all_metric_records, run_dir / "metrics.jsonl")
    save_metrics_parquet(all_metric_records, run_dir / "metrics.parquet")

    # Save observability spans
    spans_data = collector.export_otel_format()
    if spans_data:
        (run_dir / "spans.json").write_text(json.dumps(spans_data, indent=2, default=str))

    # Flat artifact
    artifact_path = output_dir / f"{artifact.run_id}.json"
    artifact_path.write_text(json.dumps(
        _serialize_artifact(artifact, scorecards, pass_k_results),
        indent=2, default=str,
    ))

    logger.info(
        "suite_completed",
        run_id=artifact.run_id,
        passed=passed,
        failed=failed,
        systems=len(suite_cfg.systems),
        scorecards=len(scorecards),
    )
    return artifact


def _serialize_artifact(
    artifact: RunArtifact,
    scorecards: list[dict[str, Any]] | None = None,
    pass_k_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": artifact.run_id,
        "suite_id": artifact.suite_id,
        "system_id": artifact.system_id,
        "model_id": artifact.model_id,
        "started_at": artifact.started_at.isoformat(),
        "finished_at": artifact.finished_at.isoformat() if artifact.finished_at else None,
        "config_hash": artifact.config_hash,
        "benchmark_version": artifact.benchmark_version,
        "seed": artifact.seed,
        "tasks_total": artifact.tasks_total,
        "tasks_passed": artifact.tasks_passed,
        "tasks_failed": artifact.tasks_failed,
        "duration_ms": artifact.duration_ms,
        "metrics": artifact.metrics,
        "scorecards": scorecards or [],
        "pass_k_results": pass_k_results or [],
        "metadata": artifact.metadata,
    }
