"""Tests for Athena model lineage tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_bench.models.lineage import (
    AthenaLineage,
    format_lineage_compact,
    format_lineage_display,
    load_athena_metadata,
)


# ──────────────────────────────────────────────────────────────────
# AthenaLineage Data Model Tests
# ──────────────────────────────────────────────────────────────────


class TestAthenaLineage:
    """Test AthenaLineage data model."""

    def test_merged_model(self):
        """Merged model should report is_merged=True."""
        lineage = AthenaLineage(
            model_name="athena-7b-merged",
            merge_method="SLERP",
            parents=["model-a", "model-b"],
            parent_weights=[0.5, 0.5],
        )

        assert lineage.is_merged is True
        assert lineage.is_fine_tuned is False

    def test_fine_tuned_model(self):
        """Fine-tuned model should report is_fine_tuned=True."""
        lineage = AthenaLineage(
            model_name="athena-7b-lora",
            base_model="meta-llama/Llama-2-7b-hf",
            athena_phase="1c_lora",
        )

        assert lineage.is_fine_tuned is True
        assert lineage.is_merged is False

    def test_base_model(self):
        """Base model (no merge, no fine-tune) should be neither."""
        lineage = AthenaLineage(model_name="base-model")

        assert lineage.is_merged is False
        assert lineage.is_fine_tuned is False

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        lineage = AthenaLineage(
            model_name="test",
            merge_method="TIES",
            parents=["a", "b"],
            parent_weights=[0.7, 0.3],
            athena_phase="1d_merge",
        )
        d = lineage.to_dict()

        assert d["model_name"] == "test"
        assert d["merge_method"] == "TIES"
        assert d["parents"] == ["a", "b"]
        assert d["parent_weights"] == [0.7, 0.3]
        assert d["athena_phase"] == "1d_merge"


# ──────────────────────────────────────────────────────────────────
# load_athena_metadata Tests
# ──────────────────────────────────────────────────────────────────


class TestLoadAthenaMetadata:
    """Test loading athena_metadata.json files."""

    def test_load_from_file(self, tmp_path: Path):
        """Should load lineage from a direct JSON file path."""
        metadata = {
            "model_name": "athena-13b-merged-v2",
            "merge_method": "SLERP",
            "parents": ["athena-7b-math", "athena-7b-code"],
            "parent_weights": [0.5, 0.5],
            "athena_phase": "1d_merge",
            "base_model": "meta-llama/Llama-2-7b-hf",
        }

        meta_file = tmp_path / "athena_metadata.json"
        meta_file.write_text(json.dumps(metadata))

        lineage = load_athena_metadata(meta_file)

        assert lineage is not None
        assert lineage.model_name == "athena-13b-merged-v2"
        assert lineage.merge_method == "SLERP"
        assert lineage.parents == ["athena-7b-math", "athena-7b-code"]
        assert lineage.parent_weights == [0.5, 0.5]
        assert lineage.is_merged is True

    def test_load_from_directory(self, tmp_path: Path):
        """Should find athena_metadata.json inside a directory."""
        metadata = {
            "model_name": "test-model",
            "merge_method": "TIES",
            "parents": ["parent-a"],
        }

        meta_file = tmp_path / "athena_metadata.json"
        meta_file.write_text(json.dumps(metadata))

        lineage = load_athena_metadata(tmp_path)

        assert lineage is not None
        assert lineage.model_name == "test-model"

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """Missing metadata file should return None."""
        lineage = load_athena_metadata(tmp_path / "nonexistent")
        assert lineage is None

    def test_load_invalid_json_returns_none(self, tmp_path: Path):
        """Invalid JSON should return None gracefully."""
        meta_file = tmp_path / "athena_metadata.json"
        meta_file.write_text("not valid json {{{")

        lineage = load_athena_metadata(meta_file)
        assert lineage is None

    def test_load_minimal_metadata(self, tmp_path: Path):
        """Minimal metadata (just model_name) should load."""
        meta_file = tmp_path / "athena_metadata.json"
        meta_file.write_text(json.dumps({"model_name": "minimal"}))

        lineage = load_athena_metadata(meta_file)

        assert lineage is not None
        assert lineage.model_name == "minimal"
        assert lineage.merge_method is None
        assert lineage.parents == []


# ──────────────────────────────────────────────────────────────────
# Display Formatter Tests
# ──────────────────────────────────────────────────────────────────


class TestFormatLineageDisplay:
    """Test human-readable lineage formatting."""

    def test_merged_model_display(self):
        """Merged model should show merge formula."""
        lineage = AthenaLineage(
            model_name="Model-C",
            merge_method="SLERP",
            parents=["Model-A", "Model-B"],
            parent_weights=[0.5, 0.5],
        )

        display = format_lineage_display(lineage)

        assert "Model-C" in display
        assert "0.5" in display
        assert "Model-A" in display
        assert "Model-B" in display
        assert "SLERP" in display

    def test_fine_tuned_model_display(self):
        """Fine-tuned model should show base model."""
        lineage = AthenaLineage(
            model_name="athena-7b-lora",
            base_model="meta-llama/Llama-2-7b-hf",
            athena_phase="1c_lora",
        )

        display = format_lineage_display(lineage)

        assert "athena-7b-lora" in display
        assert "fine-tuned" in display
        assert "Llama-2-7b-hf" in display
        assert "1c_lora" in display

    def test_base_model_display(self):
        """Base model should show simple label."""
        lineage = AthenaLineage(model_name="base-model")
        display = format_lineage_display(lineage)

        assert "base-model" in display
        assert "base model" in display

    def test_hub_path_parent_names(self):
        """Hub-style paths should use basename only."""
        lineage = AthenaLineage(
            model_name="merged",
            merge_method="TIES",
            parents=["org/model-a", "org/model-b"],
            parent_weights=[0.7, 0.3],
        )

        display = format_lineage_display(lineage)
        assert "model-a" in display
        assert "model-b" in display


class TestFormatLineageCompact:
    """Test compact lineage formatting for table columns."""

    def test_merged_compact(self):
        """Merged model should show method and parent names."""
        lineage = AthenaLineage(
            model_name="merged",
            merge_method="SLERP",
            parents=["model-a", "model-b"],
        )

        compact = format_lineage_compact(lineage)

        assert "SLERP" in compact
        assert "model-a" in compact
        assert "model-b" in compact

    def test_fine_tuned_compact(self):
        """Fine-tuned should show LoRA arrow notation."""
        lineage = AthenaLineage(
            model_name="lora-model",
            base_model="Llama-2-7b",
        )

        compact = format_lineage_compact(lineage)
        assert "LoRA→" in compact
        assert "Llama-2-7b" in compact

    def test_base_model_compact(self):
        """Base model should show dash."""
        lineage = AthenaLineage(model_name="base")
        compact = format_lineage_compact(lineage)
        assert compact == "—"


# ──────────────────────────────────────────────────────────────────
# Leaderboard Integration Tests
# ──────────────────────────────────────────────────────────────────


class TestLeaderboardLineageIntegration:
    """Test lineage data flows through to leaderboard."""

    def test_update_leaderboard_with_lineage(self, tmp_path: Path):
        """Leaderboard entries should include lineage data."""
        from agent_bench.reports.leaderboard import update_leaderboard, get_leaderboard

        lineage = AthenaLineage(
            model_name="athena-merged",
            merge_method="SLERP",
            parents=["model-a", "model-b"],
            parent_weights=[0.5, 0.5],
        )

        scorecards = [{
            "system_id": "test-system",
            "domain": "test-domain",
            "global_score": 0.85,
            "functional_score": 0.9,
            "risk_score": 0.8,
            "cost_score": 0.7,
            "latency_score": 0.6,
            "reliability_score": 0.95,
        }]

        lb_path = tmp_path / "leaderboard.json"
        update_leaderboard("run-1", "suite-1", scorecards, lb_path, athena_lineage=lineage)

        entries = get_leaderboard(leaderboard_path=lb_path)
        assert len(entries) == 1
        assert "lineage" in entries[0]
        assert "SLERP" in entries[0]["lineage"]
        assert entries[0]["merge_method"] == "SLERP"
        assert entries[0]["parents"] == ["model-a", "model-b"]

    def test_update_leaderboard_without_lineage(self, tmp_path: Path):
        """Leaderboard without lineage should work as before."""
        from agent_bench.reports.leaderboard import update_leaderboard, get_leaderboard

        scorecards = [{
            "system_id": "test-system",
            "domain": "test-domain",
            "global_score": 0.85,
            "functional_score": 0.9,
            "risk_score": 0.8,
            "cost_score": 0.7,
            "latency_score": 0.6,
            "reliability_score": 0.95,
        }]

        lb_path = tmp_path / "leaderboard.json"
        update_leaderboard("run-1", "suite-1", scorecards, lb_path)

        entries = get_leaderboard(leaderboard_path=lb_path)
        assert len(entries) == 1
        assert "lineage" not in entries[0]  # No lineage data

    def test_leaderboard_markdown_with_lineage(self, tmp_path: Path):
        """Markdown renderer should include lineage column when present."""
        from agent_bench.reports.leaderboard import (
            render_leaderboard_markdown,
            update_leaderboard,
        )

        lineage = AthenaLineage(
            model_name="merged",
            merge_method="TIES",
            parents=["a", "b"],
        )

        scorecards = [{
            "system_id": "test-system",
            "domain": "test",
            "global_score": 0.9,
            "functional_score": 0.9,
            "risk_score": 0.9,
            "cost_score": 0.9,
            "latency_score": 0.9,
            "reliability_score": 0.9,
        }]

        lb_path = tmp_path / "leaderboard.json"
        update_leaderboard("run-1", "suite-1", scorecards, lb_path, athena_lineage=lineage)

        md = render_leaderboard_markdown(leaderboard_path=lb_path)
        assert "Lineage" in md
        assert "TIES" in md
