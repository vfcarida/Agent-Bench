"""Online eval stub: ingest real traces from JSONL and evaluate offline."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
            "timestamp": datetime.now(UTC).isoformat(),
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
        domains_in = {r["domain"] for r in results}
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
                "latency_p50": sc.latency_p50,
                "latency_p90": sc.latency_p90,
                "latency_p99": sc.latency_p99,
                "cost_per_successful_task": sc.cost_per_successful_task,
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


# ============================================================================
# Continuous Shadow-Traffic Online Drift & Calibration Monitor
# ============================================================================


class CUSUMDriftDetector:
    """Two-sided tabular CUSUM (Page's Cumulative Sum) sequential change-point detector.

    Monitors a continuous metric stream (e.g., latency_ms, cost_usd, or error indicator)
    against a calibrated baseline mean and standard deviation:
        S_t^+ = max(0, S_{t-1}^+ + z_t - k)
        S_t^- = max(0, S_{t-1}^- - z_t - k)
    where z_t = (x_t - mu_0) / max(sigma_0, eps).
    Triggers an alarm when S_t^+ > h or S_t^- > h.
    """

    def __init__(
        self,
        target_mean: float,
        target_std: float = 1.0,
        *,
        drift_allowance_k: float = 0.5,
        decision_threshold_h: float = 4.0,
    ) -> None:
        self._mean = float(target_mean)
        self._std = max(1e-6, float(target_std))
        self._k = max(0.0, float(drift_allowance_k))
        self._h = max(0.5, float(decision_threshold_h))
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n = 0
        self._alarm_step: int | None = None

    def reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n = 0
        self._alarm_step = None

    def update(self, value: float) -> dict[str, Any]:
        """Ingest one observation and return current CUSUM state."""
        self._n += 1
        z = (float(value) - self._mean) / self._std
        self._s_pos = max(0.0, self._s_pos + z - self._k)
        self._s_neg = max(0.0, self._s_neg - z - self._k)

        upward = self._s_pos > self._h
        downward = self._s_neg > self._h
        alarm = upward or downward
        if alarm and self._alarm_step is None:
            self._alarm_step = self._n

        direction = "upward" if upward else ("downward" if downward else "none")
        return {
            "s_pos": round(self._s_pos, 6),
            "s_neg": round(self._s_neg, 6),
            "alarm": alarm,
            "direction": direction,
            "observations": self._n,
            "first_alarm_step": self._alarm_step,
        }

    def evaluate_stream(self, values: Sequence[float]) -> dict[str, Any]:
        """Evaluate an entire sequence of observations and return final CUSUM summary."""
        self.reset()
        last_state: dict[str, Any] = {
            "s_pos": 0.0,
            "s_neg": 0.0,
            "alarm": False,
            "direction": "none",
            "observations": 0,
            "first_alarm_step": None,
        }
        any_alarm = False
        alarm_dir = "none"
        max_s_pos = 0.0
        max_s_neg = 0.0

        for v in values:
            last_state = self.update(v)
            max_s_pos = max(max_s_pos, float(last_state["s_pos"]))
            max_s_neg = max(max_s_neg, float(last_state["s_neg"]))
            if last_state["alarm"] and not any_alarm:
                any_alarm = True
                alarm_dir = str(last_state["direction"])

        return {
            "alarm": any_alarm,
            "direction": alarm_dir,
            "max_s_pos": round(max_s_pos, 6),
            "max_s_neg": round(max_s_neg, 6),
            "observations": self._n,
            "first_alarm_step": self._alarm_step,
            "target_mean": round(self._mean, 6),
            "target_std": round(self._std, 6),
        }


def compute_jensen_shannon_divergence(
    baseline_counts: Mapping[str, int | float],
    shadow_counts: Mapping[str, int | float],
) -> float:
    """Compute base-2 Jensen-Shannon Divergence D_JS(P || Q) in [0.0, 1.0]."""
    all_keys = sorted(set(baseline_counts.keys()) | set(shadow_counts.keys()))
    if not all_keys:
        return 0.0

    sum_p = sum(max(0.0, float(baseline_counts.get(k, 0.0))) for k in all_keys)
    sum_q = sum(max(0.0, float(shadow_counts.get(k, 0.0))) for k in all_keys)
    if sum_p <= 0.0 and sum_q <= 0.0:
        return 0.0
    if sum_p <= 0.0 or sum_q <= 0.0:
        return 1.0

    p = {k: max(0.0, float(baseline_counts.get(k, 0.0))) / sum_p for k in all_keys}
    q = {k: max(0.0, float(shadow_counts.get(k, 0.0))) / sum_q for k in all_keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in all_keys}

    kl_pm = sum(
        p[k] * math.log2(p[k] / m[k]) for k in all_keys if p[k] > 1e-12 and m[k] > 1e-12
    )
    kl_qm = sum(
        q[k] * math.log2(q[k] / m[k]) for k in all_keys if q[k] > 1e-12 and m[k] > 1e-12
    )
    jsd = 0.5 * (kl_pm + kl_qm)
    return round(max(0.0, min(1.0, jsd)), 6)


@dataclass
class OnlineDriftReport:
    """Structured report from comparing shadow production traffic against a baseline run."""

    baseline_run_id: str
    shadow_source: str
    tool_distribution_jsd: float
    latency_cusum: dict[str, Any]
    cost_cusum: dict[str, Any]
    baseline_safety_violation_rate: float
    shadow_safety_violation_rate: float
    safety_rate_delta: float
    drift_detected: bool
    alerts: list[str] = field(default_factory=list)
    recommended_autonomy_action: str = "maintain_tier"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OnlineDriftMonitor:
    """Monitors shadow traffic (JSONL traces or OTLP GenAI spans) against a certified baseline run."""

    def __init__(
        self,
        *,
        jsd_threshold: float = 0.20,
        safety_delta_threshold: float = 0.02,
        cusum_k: float = 0.5,
        cusum_h: float = 4.0,
    ) -> None:
        self._jsd_threshold = jsd_threshold
        self._safety_delta_threshold = safety_delta_threshold
        self._cusum_k = cusum_k
        self._cusum_h = cusum_h

    @staticmethod
    def _extract_from_otlp(otlp_payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert OTLP/JSON resourceSpans into normalized shadow observation dicts."""
        records: list[dict[str, Any]] = []
        for rs in otlp_payload.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    attrs: dict[str, Any] = {}
                    for kv in span.get("attributes", []):
                        key = str(kv.get("key", ""))
                        val_obj = kv.get("value", {})
                        for v_val in val_obj.values():
                            attrs[key] = v_val
                    op_name = str(attrs.get("gen_ai.operation.name", span.get("name", "")))
                    start_ns = int(span.get("startTimeUnixNano", 0) or 0)
                    end_ns = int(span.get("endTimeUnixNano", 0) or 0)
                    dur_ms = max(0.0, (end_ns - start_ns) / 1_000_000.0)

                    if "execute_tool" in op_name or "gen_ai.tool.name" in attrs:
                        t_name = str(attrs.get("gen_ai.tool.name", "unknown_tool"))
                        records.append({"tool_calls": [{"name": t_name}], "metadata": {}})
                    elif "evaluate_task" in op_name or "gen_ai.task.id" in attrs:
                        cost_val = float(attrs.get("gen_ai.usage.cost_usd", 0.0) or 0.0)
                        records.append(
                            {
                                "tool_calls": [],
                                "metadata": {"latency_ms": dur_ms, "cost_usd": cost_val},
                                "safety_violated": False,
                            }
                        )
                    elif "invoke_agent" in op_name:
                        safety_gated = bool(attrs.get("gen_ai.safety.gated", False))
                        if safety_gated:
                            records.append(
                                {"tool_calls": [], "metadata": {}, "safety_violated": True}
                            )
        return records

    def evaluate_shadow_traffic(
        self,
        baseline_manifest: dict[str, Any],
        shadow_data: Sequence[dict[str, Any]] | dict[str, Any],
        *,
        shadow_source: str = "shadow_stream",
    ) -> OnlineDriftReport:
        """Compare shadow traffic observations against baseline run metrics."""
        baseline_run_id = str(baseline_manifest.get("run_id", "baseline_unknown"))
        raw_metrics = baseline_manifest.get("metrics", {})
        base_tasks: list[dict[str, Any]] = raw_metrics.get("task_results", [])

        # Normalize shadow data (either list of trace dicts or OTLP resourceSpans dict)
        if isinstance(shadow_data, dict) and "resourceSpans" in shadow_data:
            shadow_records = self._extract_from_otlp(shadow_data)
        elif isinstance(shadow_data, Sequence):
            shadow_records = list(shadow_data)
        else:
            shadow_records = []

        # 1. Baseline tool counts, latency, cost, and safety rate
        base_tool_counts: dict[str, int] = {}
        base_latencies: list[float] = []
        base_costs: list[float] = []
        base_safety_violations = 0

        for tr in base_tasks:
            for tc in tr.get("tools_called", tr.get("tool_calls", [])):
                t_name = str(tc.get("name", tc)) if isinstance(tc, dict) else str(tc)
                if t_name:
                    base_tool_counts[t_name] = base_tool_counts.get(t_name, 0) + 1
            if "latency_ms" in tr:
                base_latencies.append(float(tr.get("latency_ms", 0.0)))
            if "cost_usd" in tr:
                base_costs.append(float(tr.get("cost_usd", 0.0)))
            if tr.get("policy_violated") or tr.get("safety_violated"):
                base_safety_violations += 1

        base_safety_rate = (
            base_safety_violations / len(base_tasks) if base_tasks else 0.0
        )

        # Compute baseline mean & std for latency and cost
        lat_mean = sum(base_latencies) / len(base_latencies) if base_latencies else 50.0
        lat_var = (
            sum((x - lat_mean) ** 2 for x in base_latencies) / len(base_latencies)
            if len(base_latencies) > 1
            else max(25.0, (0.25 * lat_mean) ** 2)
        )
        lat_std = max(5.0, math.sqrt(lat_var))

        cost_mean = sum(base_costs) / len(base_costs) if base_costs else 0.005
        cost_var = (
            sum((x - cost_mean) ** 2 for x in base_costs) / len(base_costs)
            if len(base_costs) > 1
            else max(1e-6, (0.25 * cost_mean) ** 2)
        )
        cost_std = max(1e-4, math.sqrt(cost_var))

        # 2. Shadow tool counts, latency stream, cost stream, and safety rate
        shadow_tool_counts: dict[str, int] = {}
        shadow_latencies: list[float] = []
        shadow_costs: list[float] = []
        shadow_safety_violations = 0

        for rec in shadow_records:
            for tc in rec.get("tool_calls", rec.get("tools_called", [])):
                t_name = str(tc.get("name", tc)) if isinstance(tc, dict) else str(tc)
                if t_name:
                    shadow_tool_counts[t_name] = shadow_tool_counts.get(t_name, 0) + 1
            meta = rec.get("metadata", {})
            lat_val = rec.get("latency_ms", meta.get("latency_ms"))
            if lat_val is not None:
                shadow_latencies.append(float(lat_val))
            cost_val = rec.get("cost_usd", meta.get("cost_usd"))
            if cost_val is not None:
                shadow_costs.append(float(cost_val))
            if rec.get("safety_violated") or rec.get("policy_violated"):
                shadow_safety_violations += 1

        shadow_safety_rate = (
            shadow_safety_violations / len(shadow_records) if shadow_records else 0.0
        )
        safety_delta = round(shadow_safety_rate - base_safety_rate, 6)

        # 3. Evaluate JSD and CUSUM detectors
        jsd = compute_jensen_shannon_divergence(base_tool_counts, shadow_tool_counts)
        lat_detector = CUSUMDriftDetector(
            lat_mean,
            lat_std,
            drift_allowance_k=self._cusum_k,
            decision_threshold_h=self._cusum_h,
        )
        cost_detector = CUSUMDriftDetector(
            cost_mean,
            cost_std,
            drift_allowance_k=self._cusum_k,
            decision_threshold_h=self._cusum_h,
        )
        lat_cusum = lat_detector.evaluate_stream(shadow_latencies)
        cost_cusum = cost_detector.evaluate_stream(shadow_costs)

        alerts: list[str] = []
        if safety_delta > self._safety_delta_threshold or shadow_safety_violations > 0:
            alerts.append(
                f"SAFETY_DRIFT: Shadow safety violation rate ({shadow_safety_rate:.1%}) "
                f"exceeded baseline ({base_safety_rate:.1%})."
            )
        if jsd > self._jsd_threshold:
            alerts.append(
                f"TOOL_DISTRIBUTION_SHIFT: Tool call JSD={jsd:.3f} > threshold {self._jsd_threshold:.3f}."
            )
        if lat_cusum["alarm"] and lat_cusum["direction"] == "upward":
            alerts.append(
                f"LATENCY_CUSUM_ALARM: Upward latency shift detected at observation {lat_cusum['first_alarm_step']}."
            )
        if cost_cusum["alarm"] and cost_cusum["direction"] == "upward":
            alerts.append(
                f"COST_CUSUM_ALARM: Upward cost shift detected at observation {cost_cusum['first_alarm_step']}."
            )

        drift_detected = len(alerts) > 0
        if any(a.startswith("SAFETY_DRIFT") for a in alerts):
            action = "halt_and_recalibrate"
        elif drift_detected:
            action = "downgrade_to_supervised"
        else:
            action = "maintain_tier"

        return OnlineDriftReport(
            baseline_run_id=baseline_run_id,
            shadow_source=shadow_source,
            tool_distribution_jsd=jsd,
            latency_cusum=lat_cusum,
            cost_cusum=cost_cusum,
            baseline_safety_violation_rate=round(base_safety_rate, 6),
            shadow_safety_violation_rate=round(shadow_safety_rate, 6),
            safety_rate_delta=safety_delta,
            drift_detected=drift_detected,
            alerts=alerts,
            recommended_autonomy_action=action,
        )

