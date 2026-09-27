"""Match: query-relevance, identity, family, and compatibility predicates.

Per capability_expansion/plans/00_expansion_core.md F02: these are separate,
non-exclusive predicates. This module never extends or imports the shipped
commercecore RelationClass enum, and ESCI's relevance label E is never
converted into an identity/variant label here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryRelevance(str, Enum):
    EXACT = "exact"
    SUBSTITUTE = "substitute"
    COMPLEMENT = "complement"
    IRRELEVANT = "irrelevant"


class PairLabel(str, Enum):
    """Used for same_variant and same_family. Distinct label sets even though
    the value set is identical — never conflate the two tasks' evidence."""

    SAME = "same"
    DISTINCT = "distinct"
    UNKNOWN = "unknown"


class FunctionalRelation(str, Enum):
    SUBSTITUTE = "substitute"
    COMPLEMENT = "complement"
    UNRELATED = "unrelated"
    UNKNOWN = "unknown"


class TechnicalCompatibility(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QueryRelevanceLabel:
    query: str
    offer_id: str
    label: QueryRelevance
    source: str  # e.g. "esci_native"


@dataclass(frozen=True)
class IdentityPairLabel:
    listing_a_id: str
    listing_b_id: str
    label: PairLabel
    cluster_id_a: str | None
    cluster_id_b: str | None
    is_hard_negative: bool
    source: str  # e.g. "wdc_products_80pair"


@dataclass(frozen=True)
class FamilyPairLabel:
    product_a_id: str
    product_b_id: str
    label: PairLabel
    source: str


@dataclass(frozen=True)
class FunctionalRelationLabel:
    item_a_id: str
    item_b_id: str
    use_context: str | None
    label: FunctionalRelation
    directional: bool
    source: str


@dataclass(frozen=True)
class TechnicalCompatibilityLabel:
    item_a_id: str
    item_b_id: str
    revision: str | None
    region: str | None
    label: TechnicalCompatibility
    source: str
