"""Contamination and data leakage detection between holdout and dev/synthetic datasets."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from agent_bench.validators.dedup_checker import _jaccard_similarity, _tokenize


@dataclass
class ContaminationLeak:
    """Represents an identified leakage event of a holdout case into training/dev/synthetic."""

    holdout_id: str
    leaked_in_id: str
    leak_type: str  # "verbatim" | "paraphrase"
    similarity: float
    holdout_prompt: str
    leaked_prompt: str
    holdout_domain: str = ""
    comparison_domain: str = ""


@dataclass
class ContaminationReport:
    """Aggregated contamination check report."""

    passed: bool
    total_holdout_cases: int
    total_comparison_cases: int
    leaks: list[ContaminationLeak] = field(default_factory=list)
    summary_text: str = ""


def _extract_prompt(case: dict[str, Any]) -> str:
    """Extract prompt text from an EvalCase or Task dictionary."""
    prompt = case.get("prompt_or_user_goal") or case.get("prompt")
    if prompt:
        return str(prompt).strip()

    input_messages = case.get("input_messages") or []
    user_parts = [
        str(m.get("content", ""))
        for m in input_messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    if user_parts:
        return " ".join(user_parts).strip()

    return str(case.get("name") or case.get("description") or "").strip()


def compute_case_content_hash(case: dict[str, Any]) -> str:
    """Compute SHA-256 hash of normalized prompt and initial_state."""
    prompt = _extract_prompt(case).lower()
    initial_state = case.get("initial_state") or {}
    state_str = json.dumps(initial_state, sort_keys=True, default=str)
    content = f"{prompt}|{state_str}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def detect_contamination(
    holdout_cases: list[dict[str, Any]],
    comparison_cases: list[dict[str, Any]],
    threshold: float = 0.80,
) -> list[ContaminationLeak]:
    """Detect verbatim or paraphrased leakage of holdout cases into comparison cases.

    - Verbatim check: Normalized prompt exact match or SHA-256 state+prompt hash match.
    - Paraphrase check: Token-level Jaccard similarity >= threshold with length pruning.
    """
    leaks: list[ContaminationLeak] = []

    # Pre-tokenize and pre-hash holdout cases
    prepared_holdout = []
    for h in holdout_cases:
        h_id = str(h.get("id") or h.get("task_id") or "unknown_holdout")
        h_prompt = _extract_prompt(h)
        h_norm = h_prompt.lower()
        h_hash = compute_case_content_hash(h)
        h_tokens = _tokenize(h_norm)
        h_domain = str(h.get("domain") or "")
        prepared_holdout.append({
            "id": h_id,
            "prompt": h_prompt,
            "norm": h_norm,
            "hash": h_hash,
            "tokens": h_tokens,
            "domain": h_domain,
            "case": h,
        })

    # Pre-tokenize and pre-hash comparison cases
    prepared_comparison = []
    for c in comparison_cases:
        c_id = str(c.get("id") or c.get("task_id") or "unknown_comparison")
        c_prompt = _extract_prompt(c)
        c_norm = c_prompt.lower()
        c_hash = compute_case_content_hash(c)
        c_tokens = _tokenize(c_norm)
        c_domain = str(c.get("domain") or "")
        prepared_comparison.append({
            "id": c_id,
            "prompt": c_prompt,
            "norm": c_norm,
            "hash": c_hash,
            "tokens": c_tokens,
            "domain": c_domain,
            "case": c,
        })

    for h_item in prepared_holdout:
        h_id = str(h_item["id"])
        h_norm = str(h_item["norm"])
        h_hash = str(h_item["hash"])
        h_tokens = cast(set[str], h_item["tokens"])
        len_h = len(h_tokens)

        for c_item in prepared_comparison:
            c_id = str(c_item["id"])
            c_norm = str(c_item["norm"])
            c_hash = str(c_item["hash"])
            c_tokens = cast(set[str], c_item["tokens"])
            len_c = len(c_tokens)

            # 1. Verbatim check
            if h_hash == c_hash or (len(h_norm) > 10 and h_norm == c_norm):
                leaks.append(
                    ContaminationLeak(
                        holdout_id=h_id,
                        leaked_in_id=c_id,
                        leak_type="verbatim",
                        similarity=1.0,
                        holdout_prompt=str(h_item["prompt"]),
                        leaked_prompt=str(c_item["prompt"]),
                        holdout_domain=str(h_item["domain"]),
                        comparison_domain=str(c_item["domain"]),
                    )
                )
                continue

            # 2. Paraphrase / Near-duplicate check with Jaccard
            if len_h == 0 or len_c == 0:
                continue

            # Mathematical length pruning: Jaccard(A, B) <= min(|A|,|B|) / max(|A|,|B|)
            if min(len_h, len_c) / max(len_h, len_c) < threshold:
                continue

            sim = _jaccard_similarity(h_tokens, c_tokens)
            if sim >= threshold:
                leaks.append(
                    ContaminationLeak(
                        holdout_id=h_id,
                        leaked_in_id=c_id,
                        leak_type="paraphrase",
                        similarity=round(sim, 4),
                        holdout_prompt=str(h_item["prompt"]),
                        leaked_prompt=str(c_item["prompt"]),
                        holdout_domain=str(h_item["domain"]),
                        comparison_domain=str(c_item["domain"]),
                    )
                )

    return leaks


def _load_cases_from_yaml(path: Path) -> list[dict[str, Any]]:
    """Load cases or tasks from a YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return []

    if not data:
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("cases") or data.get("tasks") or []
    return []


def check_contamination(
    holdout_dir: Path | None = None,
    dev_dir: Path | None = None,
    synthetic_dir: Path | None = None,
    threshold: float = 0.80,
) -> ContaminationReport:
    """Scan holdout vs dev/synthetic dataset directories and report leakage."""
    repo_root = Path(__file__).parents[3]
    holdout_path = holdout_dir or (repo_root / "datasets" / "gold" / "holdout")
    dev_path = dev_dir or (repo_root / "datasets" / "gold" / "dev")
    syn_path = synthetic_dir or (repo_root / "datasets" / "synthetic")

    holdout_cases: list[dict[str, Any]] = []
    if holdout_path.exists():
        for yf in sorted(holdout_path.glob("*.yaml")):
            holdout_cases.extend(_load_cases_from_yaml(yf))

    comparison_cases: list[dict[str, Any]] = []
    if dev_path.exists():
        for yf in sorted(dev_path.glob("*.yaml")):
            comparison_cases.extend(_load_cases_from_yaml(yf))

    if syn_path.exists():
        for yf in sorted(syn_path.glob("**/*.yaml")):
            comparison_cases.extend(_load_cases_from_yaml(yf))

    leaks = detect_contamination(holdout_cases, comparison_cases, threshold=threshold)
    passed = len(leaks) == 0

    lines = [
        "============================================================",
        "Contamination & Data Leakage Gate Report",
        f"  Holdout cases scanned:    {len(holdout_cases)}",
        f"  Comparison cases scanned: {len(comparison_cases)}",
        f"  Similarity threshold:     {threshold:.2f}",
        f"  Leakages detected:        {len(leaks)}",
        "============================================================",
    ]

    if leaks:
        lines.append("\n[LEAKAGE DETAILS]")
        for idx, leak in enumerate(leaks, 1):
            lines.append(
                f"  {idx}. [{leak.leak_type.upper()}] Holdout: '{leak.holdout_id}' leaked into '{leak.leaked_in_id}'"
                f" (Similarity: {leak.similarity:.2f})"
            )
            lines.append(f"     Holdout prompt: \"{leak.holdout_prompt}\"")
            lines.append(f"     Leaked prompt:  \"{leak.leaked_prompt}\"")

    summary_text = "\n".join(lines)

    return ContaminationReport(
        passed=passed,
        total_holdout_cases=len(holdout_cases),
        total_comparison_cases=len(comparison_cases),
        leaks=leaks,
        summary_text=summary_text,
    )
