from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from .canonical import digest_value


SUPPORTED_SOURCE_TYPES = ("baseline", "file_override", "runtime_injection")
SUPPORTED_OVERFLOW_POLICIES = ("saturate_int64",)


@dataclass(frozen=True)
class SourceDef:
    id: str
    type: str

    def payload(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type}


@dataclass(frozen=True)
class SourceBinding:
    """Per-key source constraint used after the ordered chain is built."""

    allowed_source_ids: tuple[str, ...] | None = None
    excluded_source_ids: tuple[str, ...] = ()
    allowed_source_types: tuple[str, ...] | None = None

    def allows(self, source: SourceDef) -> bool:
        if self.allowed_source_ids is not None and source.id not in self.allowed_source_ids:
            return False
        if source.id in self.excluded_source_ids:
            return False
        if self.allowed_source_types is not None and source.type not in self.allowed_source_types:
            return False
        return True

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "excluded_source_ids": list(self.excluded_source_ids),
        }
        if self.allowed_source_ids is not None:
            result["allowed_source_ids"] = list(self.allowed_source_ids)
        if self.allowed_source_types is not None:
            result["allowed_source_types"] = list(self.allowed_source_types)
        return result


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    depth: int
    available_sources: tuple[str, ...]
    source_priority: tuple[str, ...]
    bindings: Mapping[str, SourceBinding]
    entries: Mapping[str, Mapping[str, Decimal]]

    def ordered_available_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.available_sources))

    def payload(self, *, include_priority: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "depth": self.depth,
            "available_sources": sorted(self.available_sources),
            "bindings": {key: self.bindings[key].payload() for key in sorted(self.bindings)},
            "entries": {
                source_id: {key: self.entries[source_id][key] for key in sorted(self.entries[source_id])}
                for source_id in sorted(self.entries)
            },
        }
        if include_priority:
            result["source_priority"] = list(self.source_priority)
        return result

    def all_entry_keys(self) -> tuple[str, ...]:
        values = {key for source in self.entries.values() for key in source}
        return tuple(sorted(values))


@dataclass(frozen=True)
class FormulaTerm:
    key: str | None
    coefficient: Decimal
    original_order: int

    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "coefficient": self.coefficient,
            "original_order": self.original_order,
        }


@dataclass(frozen=True)
class TaskSpec:
    id: str
    formula: str
    terms: tuple[FormulaTerm, ...]
    keys: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "formula": self.formula,
            "keys": list(self.keys),
            "terms": [term.payload() for term in self.terms],
        }


@dataclass(frozen=True)
class WorkbenchSpec:
    format_version: int
    scale_power: int
    overflow_policy: str
    error_dominance_threshold: Decimal
    source_type_bias: Mapping[str, int]
    sources: Mapping[str, SourceDef]
    environments: Mapping[str, EnvironmentSpec]
    tasks: Mapping[str, TaskSpec]
    origin_path: str = ""

    def environment_order(self) -> tuple[EnvironmentSpec, ...]:
        return tuple(sorted(self.environments.values(), key=lambda env: (env.depth, env.id)))

    def task_order(self) -> tuple[TaskSpec, ...]:
        return tuple(self.tasks[key] for key in sorted(self.tasks))

    def content_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "scale_power": self.scale_power,
            "overflow_policy": self.overflow_policy,
            "error_dominance_threshold": self.error_dominance_threshold,
            "source_type_bias": dict(self.source_type_bias),
            "sources": [self.sources[key].payload() for key in sorted(self.sources)],
            "environments": [env.payload() for env in self.environment_order()],
            "tasks": [task.payload() for task in self.task_order()],
        }

    def content_digest(self) -> str:
        return digest_value(self.content_payload())

    def environment_version_payload(self) -> dict[str, Any]:
        return {
            "sources": [self.sources[key].payload() for key in sorted(self.sources)],
            "environments": [
                env.payload(include_priority=False) for env in self.environment_order()
            ],
            "tasks": [task.payload() for task in self.task_order()],
        }

    def priority_version_payload(self) -> dict[str, Any]:
        return {
            "source_type_bias": dict(self.source_type_bias),
            "sources": [self.sources[key].payload() for key in sorted(self.sources)],
            "environment_priority": [
                {
                    "id": env.id,
                    "available_sources": sorted(env.available_sources),
                    "source_priority": list(env.source_priority),
                }
                for env in self.environment_order()
            ],
        }

    def scale_version_payload(self) -> dict[str, Any]:
        return {
            "scale_power": self.scale_power,
            "overflow_policy": self.overflow_policy,
        }

    def fingerprints(self) -> dict[str, str]:
        return {
            "environment": digest_value(self.environment_version_payload()),
            "priority": digest_value(self.priority_version_payload()),
            "scale": digest_value(self.scale_version_payload()),
            "content": self.content_digest(),
        }

    def environment_fingerprint(self, environment_id: str) -> str:
        env = self.environments[environment_id]
        return digest_value(
            {
                "environment": env.payload(),
                "sources": [
                    self.sources[source_id].payload()
                    for source_id in sorted(env.available_sources)
                ],
                "tasks": [task.payload() for task in self.task_order()],
            }
        )


@dataclass(frozen=True)
class VersionState:
    environment_version: int
    priority_version: int
    scale_version: int
    environment_fingerprint: str
    priority_fingerprint: str
    scale_fingerprint: str
    spec_digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "environment_version": self.environment_version,
            "priority_version": self.priority_version,
            "scale_version": self.scale_version,
            "environment_fingerprint": self.environment_fingerprint,
            "priority_fingerprint": self.priority_fingerprint,
            "scale_fingerprint": self.scale_fingerprint,
            "spec_digest": self.spec_digest,
        }

    def version_tuple(self) -> tuple[int, int, int]:
        return (self.environment_version, self.priority_version, self.scale_version)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "VersionState":
        return cls(
            environment_version=int(payload["environment_version"]),
            priority_version=int(payload["priority_version"]),
            scale_version=int(payload["scale_version"]),
            environment_fingerprint=str(payload["environment_fingerprint"]),
            priority_fingerprint=str(payload["priority_fingerprint"]),
            scale_fingerprint=str(payload["scale_fingerprint"]),
            spec_digest=str(payload["spec_digest"]),
        )
