"""Provenance registry: track dataset origins, versions, and ownership."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DatasetRecord:
    dataset_id: str
    domain: str
    version: str
    file_path: str
    content_hash: str
    task_count: int
    owner: str = ""
    description: str = ""
    created_at: str = ""
    approved_by: str = ""
    tags: list[str] = field(default_factory=list)


class ProvenanceRegistry:
    """Registry for tracking dataset provenance and changes."""

    def __init__(self, registry_path: Path | None = None):
        self._path = registry_path or Path("data/governance/provenance.json")
        self._records: list[DatasetRecord] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                data = json.load(f)
            self._records = [DatasetRecord(**r) for r in data.get("datasets", [])]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_datasets": len(self._records),
            "datasets": [self._record_to_dict(r) for r in self._records],
        }
        self._path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        file_path: Path,
        domain: str,
        version: str,
        owner: str = "",
        description: str = "",
    ) -> DatasetRecord:
        """Register or update a dataset in the provenance registry."""
        content_hash = self._compute_hash(file_path)
        task_count = self._count_tasks(file_path)

        # Check if already registered
        existing = next(
            (r for r in self._records if r.dataset_id == f"{domain}_{version}"),
            None,
        )
        if existing:
            existing.content_hash = content_hash
            existing.task_count = task_count
            existing.file_path = str(file_path)
        else:
            record = DatasetRecord(
                dataset_id=f"{domain}_{version}",
                domain=domain,
                version=version,
                file_path=str(file_path),
                content_hash=content_hash,
                task_count=task_count,
                owner=owner,
                description=description,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records.append(record)
            existing = record

        self._save()
        return existing

    def check_integrity(self, file_path: Path, domain: str, version: str) -> bool:
        """Check if a dataset file matches its registered hash."""
        record = next(
            (r for r in self._records if r.dataset_id == f"{domain}_{version}"),
            None,
        )
        if not record:
            return False
        current_hash = self._compute_hash(file_path)
        return current_hash == record.content_hash

    def get_record(self, domain: str, version: str | None = None) -> DatasetRecord | None:
        """Get the provenance record for a dataset."""
        if version:
            return next(
                (r for r in self._records if r.dataset_id == f"{domain}_{version}"),
                None,
            )
        # Get latest version
        domain_records = [r for r in self._records if r.domain == domain]
        return domain_records[-1] if domain_records else None

    def list_all(self) -> list[DatasetRecord]:
        return list(self._records)

    def has_changed(self, file_path: Path, domain: str, version: str) -> bool:
        """Check if file content differs from registered hash."""
        record = self.get_record(domain, version)
        if not record:
            return True  # not registered = changed
        return self._compute_hash(file_path) != record.content_hash

    @staticmethod
    def _compute_hash(path: Path) -> str:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def _count_tasks(path: Path) -> int:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return len(data.get("tasks", []))

    @staticmethod
    def _record_to_dict(r: DatasetRecord) -> dict[str, Any]:
        return {
            "dataset_id": r.dataset_id,
            "domain": r.domain,
            "version": r.version,
            "file_path": r.file_path,
            "content_hash": r.content_hash,
            "task_count": r.task_count,
            "owner": r.owner,
            "description": r.description,
            "created_at": r.created_at,
            "approved_by": r.approved_by,
            "tags": r.tags,
        }
