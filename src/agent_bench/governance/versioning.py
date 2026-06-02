"""Benchmark versioning: track suite/dataset versions and generate changelogs."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VersionEntry:
    version: str
    timestamp: str
    config_hash: str
    changes: list[str]
    author: str = ""
    datasets_hash: str = ""


@dataclass
class ChangelogEntry:
    version: str
    date: str
    changes: list[str]
    breaking: bool = False


class BenchmarkVersioning:
    """Manages benchmark suite versioning and changelog."""

    def __init__(self, version_file: Path | None = None):
        self._path = version_file or Path("data/governance/versions.json")
        self._entries: list[VersionEntry] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                data = json.load(f)
            self._entries = [VersionEntry(**e) for e in data.get("versions", [])]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "current_version": self.current_version,
            "versions": [self._entry_to_dict(e) for e in self._entries],
        }
        self._path.write_text(json.dumps(data, indent=2))

    @property
    def current_version(self) -> str:
        return self._entries[-1].version if self._entries else "0.0.0"

    def record_version(
        self,
        version: str,
        config_hash: str,
        changes: list[str],
        author: str = "",
        datasets_hash: str = "",
    ) -> VersionEntry:
        """Record a new benchmark version."""
        entry = VersionEntry(
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            changes=changes,
            author=author,
            datasets_hash=datasets_hash,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def get_changelog(self, since_version: str | None = None) -> list[ChangelogEntry]:
        """Get changelog entries since a given version."""
        entries = self._entries
        if since_version:
            idx = next(
                (i for i, e in enumerate(entries) if e.version == since_version),
                -1,
            )
            if idx >= 0:
                entries = entries[idx + 1:]

        return [
            ChangelogEntry(
                version=e.version,
                date=e.timestamp[:10],
                changes=e.changes,
            )
            for e in entries
        ]

    def render_changelog_markdown(self, since_version: str | None = None) -> str:
        """Render changelog as markdown."""
        entries = self.get_changelog(since_version)
        if not entries:
            return "# Changelog\n\nNo changes recorded.\n"

        lines = ["# Changelog\n"]
        for entry in reversed(entries):
            lines.append(f"## [{entry.version}] - {entry.date}\n")
            for change in entry.changes:
                lines.append(f"- {change}")
            lines.append("")

        return "\n".join(lines)

    def compute_datasets_hash(self, fixtures_dir: Path) -> str:
        """Compute a combined hash of all dataset files."""
        hasher = hashlib.sha256()
        for path in sorted(fixtures_dir.glob("*.yaml")):
            hasher.update(path.read_bytes())
        return hasher.hexdigest()[:16]

    def detect_changes(
        self, fixtures_dir: Path, configs_dir: Path
    ) -> list[str]:
        """Detect what changed since last recorded version."""
        changes = []
        current_datasets_hash = self.compute_datasets_hash(fixtures_dir)

        if self._entries:
            last = self._entries[-1]
            if last.datasets_hash and last.datasets_hash != current_datasets_hash:
                changes.append("Dataset files modified")

            # Check config changes
            config_hash = self._compute_configs_hash(configs_dir)
            if config_hash != last.config_hash:
                changes.append("Configuration modified")
        else:
            changes.append("Initial version")

        return changes

    @staticmethod
    def _compute_configs_hash(configs_dir: Path) -> str:
        hasher = hashlib.sha256()
        for path in sorted(configs_dir.rglob("*.yaml")):
            hasher.update(path.read_bytes())
        return hasher.hexdigest()[:12]

    @staticmethod
    def _entry_to_dict(e: VersionEntry) -> dict[str, Any]:
        return {
            "version": e.version,
            "timestamp": e.timestamp,
            "config_hash": e.config_hash,
            "changes": e.changes,
            "author": e.author,
            "datasets_hash": e.datasets_hash,
        }
