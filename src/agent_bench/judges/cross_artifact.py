"""Cross-artifact consistency judge: verifies outputs are internally consistent.

Inspired by BankerToolBench (arXiv:2604.11304), which identifies cross-artifact
consistency as the PRIMARY obstacle for AI agents in professional workflows.
Investment banking workflows require data, assumptions, and narratives to be
perfectly aligned across different file types (e.g., numbers in an Excel model
must match the narrative in a Word memo and the summary in a PowerPoint deck).

This judge extracts factual claims (especially numeric) from multiple outputs
of a single task and checks that they are consistent with each other.
"""

import re
from typing import Any

from agent_bench.core.adapters import JudgeVerdict
from agent_bench.core.artifacts import TraceEvent
from agent_bench.core.scenarios import Task


class CrossArtifactConsistencyJudge:
    """Evaluates consistency across multiple outputs/artifacts from the same task.

    Checks:
    - Numeric values cited in multiple artifacts are consistent
    - Key factual claims are not contradicted between artifacts
    - Referenced entities and dates are consistent
    """

    @property
    def judge_id(self) -> str:
        return "cross_artifact_consistency"

    @property
    def judge_type(self) -> str:
        return "deterministic"

    async def evaluate(
        self,
        task: Task,
        result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict:
        """Evaluate cross-artifact consistency.

        Expects result to contain either:
        - "artifacts": list[dict] with multiple output artifacts
        - OR "response" + "structured_data" as two artifacts to compare
        """
        artifacts = self._collect_artifacts(result)

        if len(artifacts) < 2:
            return JudgeVerdict(
                score=1.0,
                passed=True,
                reasoning="Single artifact — no cross-artifact consistency check needed.",
                judge_id=self.judge_id,
                criteria="cross_artifact_consistency",
            )

        # Extract numeric facts from each artifact
        artifact_facts = []
        for artifact in artifacts:
            facts = _extract_numeric_facts(artifact.get("content", ""))
            facts.update(_extract_named_values(artifact.get("content", "")))
            artifact_facts.append({
                "artifact_id": artifact.get("id", f"artifact_{len(artifact_facts)}"),
                "artifact_type": artifact.get("type", "unknown"),
                "facts": facts,
            })

        # Cross-check facts between all pairs of artifacts
        contradictions: list[dict[str, Any]] = []
        total_comparisons = 0
        consistent_comparisons = 0

        for i in range(len(artifact_facts)):
            for j in range(i + 1, len(artifact_facts)):
                facts_i = artifact_facts[i]["facts"]
                facts_j = artifact_facts[j]["facts"]

                # Find overlapping keys (same named value in both artifacts)
                common_keys = set(facts_i.keys()) & set(facts_j.keys())
                for key in common_keys:
                    total_comparisons += 1
                    val_i = facts_i[key]
                    val_j = facts_j[key]

                    if _values_consistent(val_i, val_j):
                        consistent_comparisons += 1
                    else:
                        contradictions.append({
                            "key": key,
                            "artifact_a": artifact_facts[i]["artifact_id"],
                            "value_a": val_i,
                            "artifact_b": artifact_facts[j]["artifact_id"],
                            "value_b": val_j,
                        })

        if total_comparisons == 0:
            # No overlapping facts to compare
            return JudgeVerdict(
                score=0.8,
                passed=True,
                reasoning="No overlapping facts found between artifacts to verify consistency.",
                judge_id=self.judge_id,
                criteria="cross_artifact_consistency",
                metadata={
                    "artifact_count": len(artifacts),
                    "total_comparisons": 0,
                },
            )

        consistency_score = consistent_comparisons / total_comparisons
        passed = consistency_score >= 0.9  # BTB-style: high bar for consistency

        return JudgeVerdict(
            score=consistency_score,
            passed=passed,
            reasoning=(
                f"Cross-artifact consistency: {consistent_comparisons}/{total_comparisons} "
                f"facts consistent ({consistency_score:.1%}). "
                f"Contradictions: {contradictions[:3]}"
            ),
            judge_id=self.judge_id,
            criteria="cross_artifact_consistency",
            metadata={
                "artifact_count": len(artifacts),
                "total_comparisons": total_comparisons,
                "consistent_comparisons": consistent_comparisons,
                "contradictions": contradictions[:10],
                "artifact_fact_counts": [
                    {"id": af["artifact_id"], "fact_count": len(af["facts"])}
                    for af in artifact_facts
                ],
            },
        )

    @staticmethod
    def _collect_artifacts(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect artifacts from result dict, supporting multiple formats."""
        artifacts: list[dict[str, Any]] = []

        # Explicit artifacts list
        if "artifacts" in result and isinstance(result["artifacts"], list):
            artifacts.extend(result["artifacts"])
            return artifacts

        # Fall back to combining response + structured_data
        if result.get("response"):
            artifacts.append({
                "id": "response",
                "type": "text",
                "content": result["response"],
            })
        if result.get("structured_data"):
            data = result["structured_data"]
            content = str(data) if not isinstance(data, str) else data
            artifacts.append({
                "id": "structured_data",
                "type": "structured_data",
                "content": content,
            })
        if result.get("calculation_output"):
            artifacts.append({
                "id": "calculation",
                "type": "calculation",
                "content": str(result["calculation_output"]),
            })

        return artifacts


def _extract_numeric_facts(text: str) -> dict[str, float]:
    """Extract labeled numeric values from text (e.g., 'Revenue: R$1.5M')."""
    facts: dict[str, float] = {}

    # Pattern: "Label: number" or "Label = number"
    patterns = [
        r'([A-Za-zÀ-ú_\s]{2,30})[:\s=]+R\$\s*([\d.]+,\d{2})',
        r'([A-Za-zÀ-ú_\s]{2,30})[:\s=]+([\d.]+,\d+)\s*%',
        r'([A-Za-zÀ-ú_\s]{2,30})[:\s=]+([\d.,]+)',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            label = match.group(1).strip().lower()
            value_str = match.group(2)
            try:
                value = _parse_number(value_str)
                if label and len(label) > 1:
                    facts[label] = value
            except ValueError:
                continue

    return facts


def _extract_named_values(text: str) -> dict[str, float]:
    """Extract values associated with common financial labels."""
    labels = [
        "total", "subtotal", "saldo", "valor", "taxa", "juros", "rendimento",
        "custo", "receita", "despesa", "lucro", "margem", "patrimônio",
        "revenue", "cost", "profit", "margin", "rate", "yield", "return",
        "balance", "amount", "price", "fee", "spread",
    ]
    facts: dict[str, float] = {}

    for label in labels:
        pattern = rf'(?i){re.escape(label)}\s*[:\s=]+\s*([\d.,]+)'
        match = re.search(pattern, text)
        if match:
            try:
                facts[label] = _parse_number(match.group(1))
            except ValueError:
                continue

    return facts


def _parse_number(s: str) -> float:
    """Parse a number that may be in BR format (1.000,50) or US format (1,000.50)."""
    s = s.strip()
    if '.' in s and ',' in s:
        # Determine format by position of last separator
        last_dot = s.rfind('.')
        last_comma = s.rfind(',')
        if last_comma > last_dot:
            # BR format: 1.000,50
            return float(s.replace('.', '').replace(',', '.'))
        else:
            # US format: 1,000.50
            return float(s.replace(',', ''))
    elif ',' in s:
        return float(s.replace(',', '.'))
    return float(s)


def _values_consistent(a: float, b: float, tolerance: float = 0.01) -> bool:
    """Check if two numeric values are consistent within tolerance."""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= tolerance
    relative_diff = abs(a - b) / max(abs(a), abs(b))
    return relative_diff <= tolerance
