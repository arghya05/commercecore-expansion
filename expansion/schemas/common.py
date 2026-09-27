"""Shared record types for the CommerceCore expansion project.

Implements the common records and evidence layers specified in
capability_expansion/plans/00_expansion_core.md. This module has no
dependency on the shipped commercecore package: no import of its
RelationClass enum, no shared serializer, no shared runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObservationState(str, Enum):
    """Fact-presence layer. Distinct from RequirementStatus."""

    PRESENT = "present"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    CONFLICTING = "conflicting"


class RequirementStatus(str, Enum):
    """Requirement-verification layer. Never derived by renaming ObservationState."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ProcessingStatus(str, Enum):
    """Request-level outcome. A processing failure is never a valid empty fact set."""

    OK = "ok"
    ABSTAINED = "abstained"
    INVALID_INPUT = "invalid_input"
    PARSE_ERROR = "parse_error"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    STALE_EVIDENCE = "stale_evidence"


class SearchOutcome(str, Enum):
    """Retrieval-level result state. no_catalog_match requires an exhaustive,
    authoritative constraint query against a stated snapshot; an empty top-K
    from approximate retrieval is no_verified_candidate."""

    FOUND = "found"
    NO_VERIFIED_CANDIDATE = "no_verified_candidate"
    NO_CATALOG_MATCH = "no_catalog_match"


class GateState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_MEASURED = "not_measured"
    NOT_APPLICABLE = "not_applicable"


class ReadinessState(str, Enum):
    SPECIFIED = "specified"
    DATA_READY = "data_ready"
    BASELINED = "baselined"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    PILOT = "pilot"
    PRODUCTION = "production"


@dataclass(frozen=True)
class EvidenceLocation:
    """Half-open character offsets against immutable text, or page/region for
    PDF/image evidence."""

    source_id: str
    source_revision: str
    start: int | None = None
    end: int | None = None
    page: int | None = None
    region: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RequestContext:
    """Authenticated context. tenant_id is resolved from authentication,
    never trusted from a caller-supplied body field."""

    request_id: str
    tenant_id: str
    session_id: str | None
    locale: str
    schema_version: str
    deadline_ms: int


@dataclass(frozen=True)
class Fact:
    product_id: str | None
    variant_id: str | None
    offer_id: str | None
    field_id: str
    raw_value: Any
    canonical_value: Any
    unit: str | None
    observation_state: ObservationState
    conflict: bool
    source_revision: str
    evidence_location: EvidenceLocation | None
    derivation_rule: str | None
    observed_at: str | None


@dataclass(frozen=True)
class Requirement:
    field_id: str
    operator: str
    value: Any
    unit: str | None
    strength: str  # "hard" | "soft"
    origin_turn: str | None
    evidence_location: EvidenceLocation | None
    status: str  # "active" | "replaced"


@dataclass(frozen=True)
class RequirementDecision:
    requirement_id: str
    status: RequirementStatus
    evidence: tuple[EvidenceLocation, ...] = field(default_factory=tuple)
    note: str | None = None


@dataclass(frozen=True)
class Result:
    status: ProcessingStatus
    model_revision: str
    index_version: str | None
    satisfied_requirement_ids: tuple[str, ...] = field(default_factory=tuple)
    unresolved_requirement_ids: tuple[str, ...] = field(default_factory=tuple)
    rejected_requirement_ids: tuple[str, ...] = field(default_factory=tuple)
    requirement_decisions: tuple[RequirementDecision, ...] = field(default_factory=tuple)
