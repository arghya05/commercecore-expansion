"""Understand: source evidence -> normalized facts, taxonomy nodes.

Per plans/01_commerce_understand.md and 00_expansion_core.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .common import EvidenceLocation, Fact


class TaxonomyNodeStatus(str, Enum):
    MAPPED = "mapped"
    TENTATIVE = "tentative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    source_revision: str
    locale: str
    raw_text: str
    structured_fields: dict


@dataclass(frozen=True)
class NormalizeRequest:
    source: SourceDocument
    requested_schema_version: str


@dataclass(frozen=True)
class NormalizeResponse:
    facts: tuple[Fact, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TaxonomyMapping:
    node_path: tuple[str, ...]
    status: TaxonomyNodeStatus
    evidence: EvidenceLocation | None
    taxonomy_version: str
