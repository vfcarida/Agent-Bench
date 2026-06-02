"""Online eval stub: ingest real traces from JSONL and evaluate offline."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig
from agent_bench.core.scenarios import Task
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.judges.composite import CompositeJudge
from agent_bench.metrics.scorecard import compute_scorecard
from agent_bench.storage.jsonl import save_metrics_jsonl, save_run_manifest, save_traces_jsonl
from agent_bench.storage.parquet import save_metrics_parquet

logger = structlog.get_logger()


def load_traces_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load interaction traces from a JSONL file.

    Expected format per line:
    {
        "trace_id": "...",
        "task_id": "...",          # maps to benchmark task
        "system_id": "...",
        "domain": "...",
        "timestamp": "...",
        "input_messages": [...],
        "response": "...",
        "tool_calls": [...],
        "retrieved_documents": [...],
        "metadata": {}
    }
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


async def run_online_eval(
    traces_path: Path,
    config: BenchConfig,
    output_dir: Path,
    domain: str | None = None,
    weighting_profile: str = "transactional_high_risk",
) -> RunArtifact:
    """Evaluate pre-collected traces against benchmark tasks.

    Matches each trace to a task by task_id, then runs judges on the
    actual response vs expected outcomes.
    """
    artifact = RunArtifact(
        suite_id=f"online_eval_{traces_path.stem}",
        config_hash="online",
        benchmark_version="1.0.0",
    )

    # Load traces
    trace_records = load_traces_from_jsonl(traces_path)
    if not trace_records:
        logger.warning("no_traces_found", path=str(traces_path))
        artifact.finalize()
        return artifact

    # Load reference tasks
    domains_to_load = set()
    if domain:
        domains_to_load.add(domain)
    else:
        domains_to_load = {str(r.get("domain", "")) for r in trace_records}

    all_tasks: dict[str, Task] = {}
    for d in domains_to_load:
        if d:
            for t in load_domain_tasks(d):
                all_tasks[t.task_id] = t

    # Evaluate each trace
    judge = CompositeJudge()
    all_traces: list[TraceEvent] = []
    metric_records: list[dict[str, Any]] = []
    task_results: dict[str, list[dict[str, Any]]] = {}  # system_id -> results

    for record in trace_records:
        task_id = record.get("task_id", "")
        system_id = record.get("system_id", "unknown")
        record_domain = record.get("domain", domain or "unknown")

        task = all_tasks.get(task_id)
        if not task:
            logger.debug("task_not_found_for_trace", task_id=task_id)
            continue

        # Build result dict from trace
        result: dict[str, Any] = {
            "response": record.get("response", ""),
            "tool_calls": record.get("tool_calls", []),
            "retrieved_documents": record.get("retrieved_documents", []),
            "final_state": record.get("final_state", {}),
            "refusal": record.get("refusal", False),
        }

        # Build trace events
        traces: list[TraceEvent] = [
            TraceEvent(
                event_type=TraceEventType.PROMPT_SENT,
                data={"messages": record.get("input_messages", []), "system_id": system_id},
            ),
            TraceEvent(
                event_type=TraceEventType.MODEL_RESPONSE,
                data={"content": result["response"]},
            ),
        ]
        if result["retrieved_documents"]:
            traces.append(TraceEvent(
                event_type=TraceEventType.RETRIEVAL_RESULT,
                data={"documents": result["retrieved_documents"]},
            ))

        # Judge
        final_verdict, _ = await judge.evaluate(task, result, traces)
        traces.append(TraceEvent(
            event_type=TraceEventType.JUDGE_DECISION,
            data={"verdict": final_verdict.passed, "score": final_verdict.score},
        ))

        all_traces.extend(traces)

        # Record result
        if system_id not in task_results:
            task_results[system_id] = []
        task_results[system_id].append({
            "task_id": task_id,
            "domain": record_domain,
            "passed": final_verdict.passed,
            "policy_violated": False,
            "latency_ms": record.get("metadata", {}).get("latency_ms", 0),
            "tokens_in": record.get("metadata", {}).get("tokens_in", 0),
            "tokens_out": record.get("metadata", {}).get("tokens_out", 0),
            "cost_usd": record.get("metadata", {}).get("cost_usd", 0),
            "repetitions": [final_verdict.passed],
        })

        metric_records.append({
            "run_id": artifact.run_id,
            "system_id": system_id,
            "task_id": task_id,
            "domain": record_domain,
            "metric_name": "task_success",
            "metric_value": final_verdict.score,
            "metric_category": "functional",
            "passed": final_verdict.passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Finalize
    total = len(metric_records)
    passed_count = sum(1 for m in metric_records if m["passed"])
    artifact.tasks_total = total
    artifact.tasks_passed = passed_count
    artifact.tasks_failed = total - passed_count
    artifact.traces = all_traces

    # Scorecards
    scorecards = []
    for sys_id, results in task_results.items():
        domains_in = set(r["domain"] for r in results)
        for d in domains_in:
            d_results = [r for r in results if r["domain"] == d]
            sc = compute_scorecard(sys_id, d, d_results, weighting_profile)
            scorecards.append({
                "system_id": sys_id,
                "domain": d,
                "functional_score": sc.functional_score,
                "risk_score": sc.risk_score,
                "cost_score": sc.cost_score,
                "latency_score": sc.latency_score,
                "reliability_score": sc.reliability_score,
                "global_score": sc.global_score,
                "weighting_profile": weighting_profile,
            })

    artifact.metrics["scorecards"] = scorecards
    artifact.finalize()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / artifact.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(artifact, run_dir)
    save_traces_jsonl(all_traces, run_dir / "traces.jsonl")
    save_metrics_jsonl(metric_records, run_dir / "metrics.jsonl")
    save_metrics_parquet(metric_records, run_dir / "metrics.parquet")

    # Flat artifact
    artifact_json = {
        "run_id": artifact.run_id,
        "suite_id": artifact.suite_id,
        "system_id": ",".join(task_results.keys()),
        "started_at": artifact.started_at.isoformat(),
        "finished_at": artifact.finished_at.isoformat() if artifact.finished_at else None,
        "config_hash": artifact.config_hash,
        "benchmark_version": artifact.benchmark_version,
        "tasks_total": artifact.tasks_total,
        "tasks_passed": artifact.tasks_passed,
        "tasks_failed": artifact.tasks_failed,
        "duration_ms": artifact.duration_ms,
        "metrics": artifact.metrics,
        "scorecards": scorecards,
        "metadata": {"source": str(traces_path), "mode": "online_eval"},
    }
    (output_dir / f"{artifact.run_id}.json").write_text(
        json.dumps(artifact_json, indent=2, default=str)
    )

    logger.info("online_eval_completed", run_id=artifact.run_id, evaluated=total, passed=passed_count)
    return artifact
