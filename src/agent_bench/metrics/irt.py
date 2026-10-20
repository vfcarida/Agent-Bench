"""2-Parameter Logistic (2PL) Item Response Theory (IRT) & Fast-Anchor Subset Selection.

Grounded in:
- Polo et al. (ICML 2024), "tinyBenchmarks: evaluating LLMs with fewer examples" (arXiv:2402.14992)
- Baker & Kim (2004), "Item Response Theory: Parameter Estimation Techniques"

Formulation:
  For agent system j with latent ability theta_j in R and task i with discrimination a_i > 0
  and difficulty b_i in R, the probability of passing task i is:
      P(Y_{j,i} = 1 | theta_j, a_i, b_i) = sigma(a_i * (theta_j - b_i))
  where sigma(z) = 1 / (1 + exp(-z)).

  Fisher Information of task i at ability theta:
      I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IRTItemParameters:
    """Estimated 2PL IRT parameters for a single benchmark task."""

    task_id: str
    difficulty_b: float
    discrimination_a: float
    empirical_pass_rate: float
    fisher_info_at_zero: float
    integrated_information: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IRTModelFit:
    """Result of fitting a 2PL IRT model across a system-by-task response matrix."""

    item_parameters: dict[str, IRTItemParameters] = field(default_factory=dict)
    system_abilities: dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0
    converged: bool = True
    iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_parameters": {k: v.to_dict() for k, v in self.item_parameters.items()},
            "system_abilities": dict(self.system_abilities),
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "iterations": self.iterations,
        }


def logistic_2pl(theta: float, a: float, b: float) -> float:
    """Compute 2PL probability P(Y = 1 | theta, a, b) = 1 / (1 + exp(-a * (theta - b)))."""
    z = max(-30.0, min(30.0, a * (theta - b)))
    return 1.0 / (1.0 + math.exp(-z))


def fisher_information_2pl(theta: float, a: float, b: float) -> float:
    """Compute Fisher Information I_i(theta) = a^2 * P(theta) * (1 - P(theta))."""
    p = logistic_2pl(theta, a, b)
    return (a * a) * p * (1.0 - p)


def _integrated_fisher_info(a: float, b: float, grid_points: int = 9) -> float:
    """Quadrature approximation of Fisher Information over [-2.5, 2.5]."""
    step = 5.0 / max(1, grid_points - 1)
    thetas = [-2.5 + k * step for k in range(grid_points)]
    return sum(fisher_information_2pl(t, a, b) for t in thetas) / len(thetas)


def fit_2pl_irt(
    task_results: dict[str, dict[str, bool]],
    *,
    max_iter: int = 40,
    tol: float = 1e-4,
    l2_reg: float = 0.15,
) -> IRTModelFit:
    """Fit a 2-Parameter Logistic (2PL) IRT model via alternating regularized MAP / Newton-Raphson.

    Args:
        task_results: Mapping from system_id -> task_id -> passed (bool).
        max_iter: Maximum alternating optimization iterations.
        tol: Convergence threshold on parameter change.
        l2_reg: L2 regularization penalty on (theta, b, log(a)) for finite estimates on unanimous items.

    Returns:
        IRTModelFit with calibrated item parameters (a_i, b_i) and system abilities (theta_j).
    """
    if not task_results:
        return IRTModelFit()

    systems = sorted(task_results.keys())
    all_tasks_set: set[str] = set()
    for sys_map in task_results.values():
        all_tasks_set.update(sys_map.keys())
    tasks = sorted(all_tasks_set)

    if not systems or not tasks:
        return IRTModelFit()

    n_sys = len(systems)
    n_tasks = len(tasks)

    # Initialize system abilities theta_j from smoothed empirical pass rates
    theta: dict[str, float] = {}
    for s in systems:
        s_map = task_results[s]
        passed = sum(1.0 for t in tasks if s_map.get(t, False))
        p_s = (passed + 0.5) / (n_tasks + 1.0)
        theta[s] = math.log(p_s / (1.0 - p_s))

    _standardize_abilities(theta)

    # Initialize task difficulties b_i and discriminations a_i
    b: dict[str, float] = {}
    a: dict[str, float] = {}
    emp_rate: dict[str, float] = {}
    for t in tasks:
        passes = sum(1.0 for s in systems if task_results[s].get(t, False))
        emp_rate[t] = passes / n_sys
        p_t = (passes + 0.5) / (n_sys + 1.0)
        # Higher pass rate => lower difficulty b_i
        b[t] = -math.log(p_t / (1.0 - p_t))
        a[t] = 1.0

    converged = False
    iteration = 0

    if n_sys >= 2:
        for iteration in range(1, max_iter + 1):
            max_delta = 0.0

            # Step 1: Update item parameters (b_i, a_i) given theta
            for t in tasks:
                b_curr = b[t]
                a_curr = a[t]

                # Update difficulty b_i (Newton step)
                grad_b = -l2_reg * b_curr
                hess_b = l2_reg
                for s in systems:
                    y = 1.0 if task_results[s].get(t, False) else 0.0
                    p = logistic_2pl(theta[s], a_curr, b_curr)
                    grad_b += -a_curr * (y - p)
                    hess_b += (a_curr * a_curr) * p * (1.0 - p)

                step_b = grad_b / max(1e-6, hess_b)
                b_new = max(-4.0, min(4.0, b_curr + step_b))
                max_delta = max(max_delta, abs(b_new - b_curr))
                b[t] = b_new

                # Update discrimination a_i (projected gradient / Newton step on a_i in [0.25, 3.5])
                grad_a = -l2_reg * (a_curr - 1.0)
                hess_a = l2_reg
                for s in systems:
                    y = 1.0 if task_results[s].get(t, False) else 0.0
                    diff = theta[s] - b_new
                    p = logistic_2pl(theta[s], a_curr, b_new)
                    grad_a += diff * (y - p)
                    hess_a += (diff * diff) * p * (1.0 - p)

                step_a = grad_a / max(1e-6, hess_a)
                a_new = max(0.25, min(3.5, a_curr + 0.5 * step_a))
                max_delta = max(max_delta, abs(a_new - a_curr))
                a[t] = a_new

            # Step 2: Update system abilities theta_j given (a_i, b_i)
            prev_theta = dict(theta)
            for s in systems:
                t_curr = theta[s]
                grad_t = -l2_reg * t_curr
                hess_t = l2_reg
                for t in tasks:
                    y = 1.0 if task_results[s].get(t, False) else 0.0
                    p = logistic_2pl(t_curr, a[t], b[t])
                    grad_t += a[t] * (y - p)
                    hess_t += (a[t] * a[t]) * p * (1.0 - p)

                step_t = grad_t / max(1e-6, hess_t)
                t_new = max(-4.0, min(4.0, t_curr + step_t))
                theta[s] = t_new

            _standardize_abilities(theta)
            for s in systems:
                max_delta = max(max_delta, abs(theta[s] - prev_theta[s]))

            if max_delta < tol:
                converged = True
                break
    else:
        converged = True
        iteration = 1

    # Compute final log-likelihood
    log_lik = 0.0
    for s in systems:
        for t in tasks:
            y = 1.0 if task_results[s].get(t, False) else 0.0
            p = max(1e-9, min(1.0 - 1e-9, logistic_2pl(theta[s], a[t], b[t])))
            log_lik += y * math.log(p) + (1.0 - y) * math.log(1.0 - p)

    items: dict[str, IRTItemParameters] = {}
    for t in tasks:
        f_zero = fisher_information_2pl(0.0, a[t], b[t])
        int_info = _integrated_fisher_info(a[t], b[t])
        items[t] = IRTItemParameters(
            task_id=t,
            difficulty_b=round(b[t], 6),
            discrimination_a=round(a[t], 6),
            empirical_pass_rate=round(emp_rate[t], 6),
            fisher_info_at_zero=round(f_zero, 6),
            integrated_information=round(int_info, 6),
        )

    rounded_abilities = {s: round(val, 6) for s, val in theta.items()}
    return IRTModelFit(
        item_parameters=items,
        system_abilities=rounded_abilities,
        log_likelihood=round(log_lik, 6),
        converged=converged,
        iterations=iteration,
    )


def _standardize_abilities(theta: dict[str, float]) -> None:
    """Normalize theta to zero mean and unit variance when >= 2 systems exist with non-zero spread."""
    if len(theta) < 2:
        return
    vals = list(theta.values())
    mean_v = sum(vals) / len(vals)
    var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    std_v = math.sqrt(var_v)
    for k in theta:
        if std_v > 1e-6:
            theta[k] = (theta[k] - mean_v) / std_v
        else:
            theta[k] = theta[k] - mean_v


def select_irt_anchor_tasks(
    irt_fit: IRTModelFit | dict[str, Any],
    *,
    anchor_count: int | None = None,
    anchor_fraction: float = 0.25,
) -> dict[str, Any]:
    """Select an informative anchor subset of tasks via stratified Fisher Information maximization.

    Follows the tinyBenchmarks (arXiv:2402.14992) principle: partition the difficulty spectrum
    b_i into K quantiles and select the highest-information task in each stratum, assigning
    importance weights proportional to stratum representation.
    """
    raw_items = (
        irt_fit.item_parameters
        if isinstance(irt_fit, IRTModelFit)
        else irt_fit.get("item_parameters", {})
    )
    if not raw_items:
        return {
            "anchor_task_ids": [],
            "anchor_count": 0,
            "total_tasks": 0,
            "anchor_fraction_actual": 0.0,
            "information_retention_ratio": 0.0,
            "item_weights": {},
        }

    items: list[IRTItemParameters] = []
    for k, v in raw_items.items():
        if isinstance(v, IRTItemParameters):
            items.append(v)
        elif isinstance(v, dict):
            items.append(
                IRTItemParameters(
                    task_id=str(v.get("task_id", k)),
                    difficulty_b=float(v.get("difficulty_b", 0.0)),
                    discrimination_a=float(v.get("discrimination_a", 1.0)),
                    empirical_pass_rate=float(v.get("empirical_pass_rate", 0.5)),
                    fisher_info_at_zero=float(v.get("fisher_info_at_zero", 0.25)),
                    integrated_information=float(v.get("integrated_information", 0.2)),
                )
            )

    total = len(items)
    if anchor_count is None:
        clamped_frac = max(0.05, min(1.0, anchor_fraction))
        k_target = max(1, min(total, math.ceil(total * clamped_frac)))
    else:
        k_target = max(1, min(total, anchor_count))

    # Sort items by difficulty_b ascending (with task_id tie-breaker for determinism)
    sorted_by_diff = sorted(items, key=lambda x: (x.difficulty_b, x.task_id))

    selected: list[IRTItemParameters] = []
    stratum_sizes: list[int] = []

    for s_idx in range(k_target):
        start = (s_idx * total) // k_target
        end = ((s_idx + 1) * total) // k_target
        stratum = sorted_by_diff[start:end]
        if not stratum:
            continue
        # Pick item in stratum with highest integrated Fisher Information
        best_item = max(
            stratum,
            key=lambda x: (x.integrated_information, x.discrimination_a, -abs(x.difficulty_b)),
        )
        selected.append(best_item)
        stratum_sizes.append(len(stratum))

    total_info = sum(x.integrated_information for x in items)
    selected_info = sum(x.integrated_information for x in selected)
    info_ratio = selected_info / total_info if total_info > 0 else 0.0

    size_sum = sum(stratum_sizes) or 1
    weights = {
        item.task_id: round(sz / size_sum, 6)
        for item, sz in zip(selected, stratum_sizes, strict=False)
    }

    return {
        "anchor_task_ids": [x.task_id for x in selected],
        "anchor_count": len(selected),
        "total_tasks": total,
        "anchor_fraction_actual": round(len(selected) / total, 6),
        "information_retention_ratio": round(info_ratio, 6),
        "item_weights": weights,
    }


def estimate_full_score_from_anchor(
    anchor_outcomes: dict[str, bool],
    irt_fit: IRTModelFit | dict[str, Any],
    item_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate an agent's latent ability theta_hat and full-benchmark score from anchor tasks only.

    Args:
        anchor_outcomes: Mapping from anchor task_id -> passed (bool).
        irt_fit: Fitted 2PL IRT model containing item parameters for all N tasks in the suite.
        item_weights: Optional stratum weights for the anchor tasks.

    Returns:
        Dictionary containing:
        - theta_hat: MAP estimate of agent latent ability
        - ability_se: Asymptotic standard error of theta_hat
        - irt_predicted_full_score: Expected accuracy across all N suite tasks under 2PL IRT
        - weighted_anchor_score: Stratum-weighted empirical pass rate on anchor subset
        - raw_anchor_score: Unweighted pass rate on anchor subset
    """
    raw_items = (
        irt_fit.item_parameters
        if isinstance(irt_fit, IRTModelFit)
        else irt_fit.get("item_parameters", {})
    )
    items: dict[str, IRTItemParameters] = {}
    for k, v in raw_items.items():
        if isinstance(v, IRTItemParameters):
            items[k] = v
        elif isinstance(v, dict):
            items[k] = IRTItemParameters(
                task_id=str(v.get("task_id", k)),
                difficulty_b=float(v.get("difficulty_b", 0.0)),
                discrimination_a=float(v.get("discrimination_a", 1.0)),
                empirical_pass_rate=float(v.get("empirical_pass_rate", 0.5)),
                fisher_info_at_zero=float(v.get("fisher_info_at_zero", 0.25)),
                integrated_information=float(v.get("integrated_information", 0.2)),
            )

    if not anchor_outcomes:
        return {
            "theta_hat": 0.0,
            "ability_se": 1.0,
            "irt_predicted_full_score": 0.0,
            "weighted_anchor_score": 0.0,
            "raw_anchor_score": 0.0,
            "anchor_tasks_evaluated": 0,
        }

    raw_score = sum(1.0 for v in anchor_outcomes.values() if v) / len(anchor_outcomes)

    if item_weights:
        w_sum = sum(item_weights.get(tid, 0.0) for tid in anchor_outcomes)
        if w_sum > 0:
            weighted_score = (
                sum(
                    item_weights.get(tid, 0.0) * (1.0 if passed else 0.0)
                    for tid, passed in anchor_outcomes.items()
                )
                / w_sum
            )
        else:
            weighted_score = raw_score
    else:
        weighted_score = raw_score

    # Estimate latent ability theta_hat via MAP Newton-Raphson with N(0, 1) prior
    theta_hat = 0.0
    l2_prior = 0.25
    for _ in range(25):
        grad = -l2_prior * theta_hat
        hess = l2_prior
        for tid, passed in anchor_outcomes.items():
            item = items.get(tid)
            a_i = item.discrimination_a if item else 1.0
            b_i = item.difficulty_b if item else 0.0
            y = 1.0 if passed else 0.0
            p = logistic_2pl(theta_hat, a_i, b_i)
            grad += a_i * (y - p)
            hess += (a_i * a_i) * p * (1.0 - p)
        step = grad / max(1e-6, hess)
        theta_hat = max(-4.5, min(4.5, theta_hat + step))
        if abs(step) < 1e-5:
            break

    # Compute Fisher information at theta_hat for standard error
    info_sum = l2_prior
    for tid in anchor_outcomes:
        item = items.get(tid)
        a_i = item.discrimination_a if item else 1.0
        b_i = item.difficulty_b if item else 0.0
        info_sum += fisher_information_2pl(theta_hat, a_i, b_i)
    ability_se = 1.0 / math.sqrt(max(1e-6, info_sum))

    # Predict expected full-suite score across all items in the IRT model
    if items:
        irt_pred = sum(
            logistic_2pl(theta_hat, it.discrimination_a, it.difficulty_b)
            for it in items.values()
        ) / len(items)
    else:
        irt_pred = weighted_score

    return {
        "theta_hat": round(theta_hat, 6),
        "ability_se": round(ability_se, 6),
        "irt_predicted_full_score": round(irt_pred, 6),
        "weighted_anchor_score": round(weighted_score, 6),
        "raw_anchor_score": round(raw_score, 6),
        "anchor_tasks_evaluated": len(anchor_outcomes),
    }
