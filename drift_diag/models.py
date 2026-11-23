"""Unified target-state and observation data models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .canonical import content_digest


FIELD_ORDER: tuple[str, ...] = (
    "replicas",
    "image",
    "resource_limit",
    "probe_ms",
    "labels",
    "parent_ref",
)
NUMERIC_FIELDS = frozenset({"replicas", "resource_limit", "probe_ms"})
FIELD_INDEX = {field: index for index, field in enumerate(FIELD_ORDER)}


@dataclass(frozen=True)
class FieldProvenance:
    source: str
    format_version: int
    defaults_version: str | None = None
    rule_id: str | None = None
    default_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "format_version": self.format_version,
            "defaults_version": self.defaults_version,
            "rule_id": self.rule_id,
            "default_value": deepcopy(self.default_value),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FieldProvenance":
        return cls(
            source=value["source"],
            format_version=value["format_version"],
            defaults_version=value.get("defaults_version"),
            rule_id=value.get("rule_id"),
            default_value=deepcopy(value.get("default_value")),
        )


@dataclass
class WorkloadTarget:
    name: str
    fields: dict[str, Any]
    provenance: dict[str, FieldProvenance]
    source_format_version: int

    def field(self, name: str) -> Any:
        return self.fields[name]

    @property
    def filled_fields(self) -> tuple[str, ...]:
        return tuple(field for field in FIELD_ORDER if self.provenance[field].source == "default")

    @property
    def fill_basis(self) -> dict[str, dict[str, Any]]:
        return {
            field: {
                "rule_id": self.provenance[field].rule_id,
                "defaults_version": self.provenance[field].defaults_version,
                "default_value": deepcopy(self.provenance[field].default_value),
            }
            for field in FIELD_ORDER
            if self.provenance[field].source == "default"
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_format_version": self.source_format_version,
            "fields": {field: deepcopy(self.fields[field]) for field in FIELD_ORDER},
            "provenance": {field: self.provenance[field].to_dict() for field in FIELD_ORDER},
            "filled_fields": list(self.filled_fields),
            "fill_basis": self.fill_basis,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkloadTarget":
        return cls(
            name=value["name"],
            fields={field: deepcopy(value["fields"][field]) for field in FIELD_ORDER},
            provenance={
                field: FieldProvenance.from_dict(value["provenance"][field]) for field in FIELD_ORDER
            },
            source_format_version=value["source_format_version"],
        )


@dataclass
class TargetSet:
    workloads: dict[str, WorkloadTarget]
    declaration_format_versions: tuple[int, ...]
    defaults_version: str
    declaration_digest: str

    @property
    def target_digest(self) -> str:
        return content_digest(self.to_dict())

    def ordered_workloads(self) -> list[WorkloadTarget]:
        return [self.workloads[name] for name in sorted(self.workloads)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "declaration_format_versions": list(self.declaration_format_versions),
            "defaults_version": self.defaults_version,
            "declaration_digest": self.declaration_digest,
            "workloads": {
                name: self.workloads[name].to_dict() for name in sorted(self.workloads)
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetSet":
        return cls(
            workloads={
                name: WorkloadTarget.from_dict(workload)
                for name, workload in value["workloads"].items()
            },
            declaration_format_versions=tuple(value["declaration_format_versions"]),
            defaults_version=value["defaults_version"],
            declaration_digest=value["declaration_digest"],
        )


@dataclass(frozen=True)
class ObservationEvent:
    seq: int
    workload: str
    field: str
    value: Any
    at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "workload": self.workload,
            "field": self.field,
            "value": deepcopy(self.value),
            "at_ms": self.at_ms,
        }
