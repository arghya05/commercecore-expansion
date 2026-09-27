"""Data pipeline invariant checks.

Per 00_expansion_core.md: "The object that is serialized must be the exact
object that passed validation." These checks operate on the exact record
about to be written, never on an upstream draft.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


EXCLUDED_TASK_IDS = {
    "query_parse",
    "query_constraints",
    "queryner",
    "query_entity",
}


def reject_excluded_tasks(task_id: str) -> ValidationResult:
    """Per plans/12_data_and_evidence_registry.md: standalone query-to-constraints
    extraction and QueryNER/query-entity targets are excluded from the new
    training mixture and evaluation suite."""
    if task_id.lower() in EXCLUDED_TASK_IDS:
        return ValidationResult(
            ok=False,
            errors=(f"task_id {task_id!r} is an excluded legacy-Query task",),
        )
    return ValidationResult(ok=True)


def validate_understand_record(record: dict) -> ValidationResult:
    """Schema + completeness + evidence-grounding checks for an Understand
    normalize record before it is serialized to a training file."""
    errors: list[str] = []
    required = {"source_id", "source_revision", "field_id", "raw_value", "canonical_value"}
    missing = required - record.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if "task_id" in record:
        r = reject_excluded_tasks(record["task_id"])
        if not r.ok:
            errors.extend(r.errors)
    ev = record.get("evidence_location")
    if ev is not None and "raw_value" in record:
        start, end = ev.get("start"), ev.get("end")
        text = record.get("source_text")
        if text is not None and start is not None and end is not None:
            span = text[start:end]
            if str(record["raw_value"]).strip().lower() not in span.strip().lower():
                errors.append(
                    "evidence span does not contain raw_value: "
                    f"span={span!r} raw_value={record['raw_value']!r}"
                )
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_match_record(record: dict) -> ValidationResult:
    errors: list[str] = []
    task = record.get("task")
    if task == "relevance":
        required = {"query", "offer_id", "label", "source"}
    elif task in ("identity", "family"):
        required = {"entity_a_id", "entity_b_id", "label", "source"}
    elif task == "verify":
        required = {"requirement", "evidence", "label", "source"}
    else:
        return ValidationResult(ok=False, errors=(f"unknown match task {task!r}",))
    missing = required - record.keys()
    if missing:
        errors.append(f"missing fields for task={task}: {sorted(missing)}")
    if task == "identity" and record.get("source") == "esci_relevance":
        errors.append(
            "identity label cannot be derived from ESCI relevance (E label) — F02"
        )
    return ValidationResult(ok=not errors, errors=tuple(errors))
