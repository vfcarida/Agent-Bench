"""Athena model lineage tracking for merge genealogy.

Tracks the parentage and merge configuration of Athena's merged models
(Phase 1d — SLERP, TIES, DARE). When a merged model is evaluated, its
lineage is attached to leaderboard entries so researchers can correlate
merge strategy with benchmark performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class AthenaLineage:
    """Tracks model genealogy from Athena's merge pipeline.

    Attributes:
        model_name: Name of the merged/fine-tuned model.
        merge_method: Merge algorithm used (SLERP, TIES, DARE, linear, etc.).
        parents: List of parent model names/IDs used in the merge.
        parent_weights: Weights assigned to each parent (if applicable).
        merge_config: Full merge configuration dictionary.
        athena_phase: The Athena pipeline phase that produced this model.
        training_metadata: Additional training info (epochs, lr, dataset, etc.).
        base_model: The original base model before fine-tuning.
    """

    model_name: str
    merge_method: str | None = None
    parents: list[str] = field(default_factory=list)
    parent_weights: list[float] = field(default_factory=list)
    merge_config: dict[str, Any] = field(default_factory=dict)
    athena_phase: str | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)
    base_model: str | None = None

    @property
    def is_merged(self) -> bool:
        """Whether this model is the result of a merge."""
        return self.merge_method is not None and len(self.parents) > 0

    @property
    def is_fine_tuned(self) -> bool:
        """Whether this model is fine-tuned (LoRA or full)."""
        return self.base_model is not None and not self.is_merged

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "model_name": self.model_name,
            "merge_method": self.merge_method,
            "parents": self.parents,
            "parent_weights": self.parent_weights,
            "merge_config": self.merge_config,
            "athena_phase": self.athena_phase,
            "training_metadata": self.training_metadata,
            "base_model": self.base_model,
        }


def load_athena_metadata(path: Path) -> AthenaLineage | None:
    """Load athena_metadata.json from a model directory.

    Expected JSON schema:
    {
        "model_name": "athena-7b-merged-v2",
        "merge_method": "SLERP",
        "parents": ["athena-7b-math", "athena-7b-code"],
        "parent_weights": [0.5, 0.5],
        "merge_config": { ... },
        "athena_phase": "1d_merge",
        "training_metadata": { ... },
        "base_model": "meta-llama/Llama-2-7b-hf"
    }

    Args:
        path: Path to athena_metadata.json or to the model directory
              containing athena_metadata.json.

    Returns:
        AthenaLineage instance, or None if file not found.
    """
    # Accept both direct file path and directory path
    if path.is_dir():
        metadata_file = path / "athena_metadata.json"
    else:
        metadata_file = path

    if not metadata_file.exists():
        logger.debug("athena_metadata_not_found", path=str(metadata_file))
        return None

    try:
        with open(metadata_file, encoding="utf-8") as f:
            data = json.load(f)

        lineage = AthenaLineage(
            model_name=data.get("model_name", metadata_file.parent.name),
            merge_method=data.get("merge_method"),
            parents=data.get("parents", []),
            parent_weights=data.get("parent_weights", []),
            merge_config=data.get("merge_config", {}),
            athena_phase=data.get("athena_phase"),
            training_metadata=data.get("training_metadata", {}),
            base_model=data.get("base_model"),
        )

        logger.info(
            "athena_metadata_loaded",
            model=lineage.model_name,
            merge_method=lineage.merge_method,
            parents=lineage.parents,
        )
        return lineage

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("athena_metadata_parse_error", path=str(metadata_file), error=str(e))
        return None


def format_lineage_display(lineage: AthenaLineage) -> str:
    """Format lineage as a human-readable string.

    Examples:
        "Model C = 0.5×A + 0.5×B via SLERP"
        "Model D (fine-tuned from Llama-2-7b, phase: 1c_lora)"
        "Model E (base model)"

    Args:
        lineage: AthenaLineage instance.

    Returns:
        Human-readable lineage description.
    """
    if lineage.is_merged:
        parts: list[str] = []
        for i, parent in enumerate(lineage.parents):
            weight = lineage.parent_weights[i] if i < len(lineage.parent_weights) else "?"
            parent_short = parent.split("/")[-1] if "/" in parent else parent
            parts.append(f"{weight}×{parent_short}")

        merge_str = " + ".join(parts)
        method = lineage.merge_method or "unknown"
        return f"{lineage.model_name} = {merge_str} via {method}"

    if lineage.is_fine_tuned:
        base_short = lineage.base_model.split("/")[-1] if lineage.base_model and "/" in lineage.base_model else lineage.base_model
        phase_str = f", phase: {lineage.athena_phase}" if lineage.athena_phase else ""
        return f"{lineage.model_name} (fine-tuned from {base_short}{phase_str})"

    return f"{lineage.model_name} (base model)"


def format_lineage_compact(lineage: AthenaLineage) -> str:
    """Short-form lineage for table columns.

    Examples:
        "SLERP(A,B)"
        "LoRA→Llama2-7b"
        "—" (no lineage)

    Args:
        lineage: AthenaLineage instance.

    Returns:
        Compact lineage string suitable for table display.
    """
    if lineage.is_merged:
        parent_names = [p.split("/")[-1][:10] for p in lineage.parents]
        method = lineage.merge_method or "merge"
        return f"{method}({','.join(parent_names)})"

    if lineage.is_fine_tuned:
        base_short = (
            lineage.base_model.split("/")[-1][:12]
            if lineage.base_model
            else "unknown"
        )
        return f"LoRA→{base_short}"

    return "—"
