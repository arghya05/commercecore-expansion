"""Machine-readable source/artifact registry.

Implements the source-record schema in
capability_expansion/plans/12_data_and_evidence_registry.md. Null/unverified
fields are intentional; nothing here is silently inherited from a repository
license badge.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SourceRecord:
    source_id: str
    primary_url: str
    checked_date: str
    artifact_kind: str
    revision: str | None = None
    download_sha256: str | None = None
    verification: str = "PRIMARY_PAGE_CHECKED"
    code_license: str | None = None
    weights_license: str | None = None
    annotation_license: str | None = None
    source_content_terms: str | None = None
    access_terms: str | None = None
    attribution_requirements: list[str] = field(default_factory=list)
    intended_tasks: list[str] = field(default_factory=list)
    prohibited_task_mappings: list[str] = field(default_factory=list)
    overlap_sources: list[str] = field(default_factory=list)
    acquisition_status: str = "NOT_ACQUIRED"
    ingestion_gate: str = "NOT_MEASURED"
    evidence_notes: list[str] = field(default_factory=list)


class SourceRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._records: dict[str, SourceRecord] = {}
        if path.exists():
            data = json.loads(path.read_text())
            for row in data:
                self._records[row["source_id"]] = SourceRecord(**row)

    def upsert(self, record: SourceRecord) -> None:
        self._records[record.source_id] = record

    def get(self, source_id: str) -> SourceRecord | None:
        return self._records.get(source_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(r) for r in self._records.values()]
        self.path.write_text(json.dumps(rows, indent=2, sort_keys=True))

    def require_gate(self, source_id: str, gate: str = "PASS") -> SourceRecord:
        rec = self.get(source_id)
        if rec is None:
            raise ValueError(f"unregistered source: {source_id}")
        if rec.ingestion_gate != gate:
            raise ValueError(
                f"source {source_id} ingestion_gate={rec.ingestion_gate!r}, required {gate!r}"
            )
        return rec
