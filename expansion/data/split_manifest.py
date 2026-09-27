"""Split roles and exposure tracking.

Per 00_expansion_core.md: maintain TRAINING, DEVELOPMENT, CALIBRATION, and
LOCKED_FINAL roles with a global exposure registry across all capabilities.
Split product families / merchant-source groups / sessions before augmentation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class SplitRole(str, Enum):
    TRAINING = "training"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    LOCKED_FINAL = "locked_final"


@dataclass
class SplitManifest:
    dataset_id: str
    split_key_field: str  # e.g. "product_family", "cluster_id", "query_id"
    assignments: dict[str, str] = field(default_factory=dict)  # key -> SplitRole.value

    def assign(self, key: str, role: SplitRole) -> None:
        existing = self.assignments.get(key)
        if existing is not None and existing != role.value:
            raise ValueError(
                f"split key {key!r} already assigned to {existing!r}, "
                f"cannot reassign to {role.value!r} (cross-split leakage)"
            )
        self.assignments[key] = role.value

    def role_of(self, key: str) -> str | None:
        return self.assignments.get(key)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for role in self.assignments.values():
            out[role] = out.get(role, 0) + 1
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        data = json.loads(path.read_text())
        return cls(**data)
