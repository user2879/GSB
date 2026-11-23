"""Strict loading and normalization of historical declaration formats v1-v3."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_digest, load_json_strict
from .defaults import get_default_rule, get_rule_set
from .errors import format_error
from .models import FIELD_ORDER, FieldProvenance, TargetSet, WorkloadTarget


LABEL_KEY_RE = re.compile(
    r"^(?:(?:[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)\.)*[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    r"(?:/[A-Za-z0-9](?:[-A-Za-z0-9_.]*[A-Za-z0-9])?)?$"
)

REQUIRED_FIELDS_BY_VERSION: dict[int, tuple[str, ...]] = {
    1: ("name", "replicas"),
    2: ("name", "replicas", "image", "resource_limit"),
    3: ("name", "replicas", "image", "resource_limit", "probe_ms", "labels", "parent_ref"),
}


@dataclass(frozen=True)
class RawDeclaration:
    raw: dict[str, Any]
    format_version: int
    digest: str


def validate_label_key(key: Any) -> bool:
    if not isinstance(key, str) or not key or len(key) > 253:
        return False
    return LABEL_KEY_RE.fullmatch(key) is not None


def validate_labels(labels: Any) -> str | None:
    if not isinstance(labels, dict):
        return "labels must be an object"
    for key in sorted(labels):
        if not validate_label_key(key):
            return f"invalid label key: {key!r}"
        if not isinstance(labels[key], str):
            return f"label value for {key!r} must be a string"
    return None


def _require_int(value: Any, *, field: str, item: Any, minimum: int, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise format_error(f"{field} must be an integer", item=item, field=field)
    if strict and value <= minimum:
        raise format_error(f"{field} must be greater than {minimum}", item=item, field=field)
    if not strict and value < minimum:
        raise format_error(f"{field} must be at least {minimum}", item=item, field=field)


def _validate_field_value(field: str, value: Any, item: Any) -> None:
    if field == "replicas":
        _require_int(value, field=field, item=item, minimum=0)
    elif field == "image":
        if not isinstance(value, str) or not value:
            raise format_error("image must be a non-empty string", item=item, field=field)
    elif field == "resource_limit":
        _require_int(value, field=field, item=item, minimum=0, strict=True)
    elif field == "probe_ms":
        _require_int(value, field=field, item=item, minimum=0)
    elif field == "labels":
        error = validate_labels(value)
        if error:
            raise format_error(error, item=item, field=field)
    elif field == "parent_ref":
        if value is not None and (not isinstance(value, str) or not value):
            raise format_error("parent_ref must be null or a non-empty string", item=item, field=field)


def _first_cycle(workloads: list[dict[str, Any]]) -> tuple[str, list[str]] | None:
    by_name = {workload["name"]: workload for workload in workloads}
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(name: str, path: list[str]) -> tuple[str, list[str]] | None:
        if name in visiting:
            return name, path + [name]
        if name in visited:
            return None
        visiting.add(name)
        parent = by_name[name].get("parent_ref")
        if isinstance(parent, str) and parent in by_name:
            found = walk(parent, path + [name])
            if found:
                return found
        visiting.remove(name)
        visited.add(name)
        return None

    for name in sorted(by_name):
        found = walk(name, [])
        if found:
            return found
    return None


def validate_declaration(raw: Any) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise format_error("declaration root must be an object", item=0, field="<root>")

    allowed_top_level = {"format_version", "workloads"}
    for key in raw:
        if key not in allowed_top_level:
            raise format_error(f"unknown top-level field: {key}", item=0, field=key)

    if "format_version" not in raw:
        raise format_error("missing format_version", item=0, field="format_version")
    version = raw["format_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version not in REQUIRED_FIELDS_BY_VERSION:
        raise format_error(
            f"unknown declaration format version: {version!r}",
            item=0,
            field="format_version",
            details={"supported_versions": sorted(REQUIRED_FIELDS_BY_VERSION)},
        )

    if "workloads" not in raw:
        raise format_error("missing workloads", item=0, field="workloads")
    workloads = raw["workloads"]
    if not isinstance(workloads, list):
        raise format_error("workloads must be an array", item=0, field="workloads")

    required = REQUIRED_FIELDS_BY_VERSION[version]
    required_set = set(required)
    names: dict[str, int] = {}
    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict):
            raise format_error("workload entry must be an object", item=index, field="<entry>")
        for key in workload:
            if key not in required_set:
                raise format_error(
                    f"field {key!r} is not allowed in format v{version}",
                    item=index,
                    field=key,
                )
        for field in required:
            if field not in workload:
                raise format_error("required field is missing", item=index, field=field)
        name = workload["name"]
        if not isinstance(name, str) or not name:
            raise format_error("name must be a non-empty string", item=index, field="name")
        if name in names:
            raise format_error(
                f"duplicate workload name: {name}",
                item=index,
                field="name",
                details={"first_index": names[name]},
            )
        names[name] = index
        for field in required:
            if field != "name":
                _validate_field_value(field, workload[field], index)

    if version == 3:
        for index, workload in enumerate(workloads):
            parent = workload["parent_ref"]
            if parent is not None and parent not in names:
                raise format_error(
                    f"dangling parent_ref: {parent}",
                    item=index,
                    field="parent_ref",
                    details={"referenced_workload": parent},
                )
        cycle = _first_cycle(workloads)
        if cycle:
            cycle_name, path = cycle
            index = names[cycle_name]
            raise format_error(
                "cyclic parent_ref chain",
                item=index,
                field="parent_ref",
                details={"cycle": path},
            )

    return version, workloads


def load_declaration(path: str | Path) -> RawDeclaration:
    raw = load_json_strict(path)
    version, _ = validate_declaration(raw)
    return RawDeclaration(raw=deepcopy(raw), format_version=version, digest=content_digest(raw))


def normalize_declaration(
    raw: Any,
    *,
    defaults_version: str = "defaults-v1",
) -> TargetSet:
    get_rule_set(defaults_version)
    version, workloads = validate_declaration(raw)
    normalized: dict[str, WorkloadTarget] = {}

    for workload in workloads:
        name = workload["name"]
        fields: dict[str, Any] = {}
        provenance: dict[str, FieldProvenance] = {}

        for field in FIELD_ORDER:
            if field in workload and field != "name":
                fields[field] = deepcopy(workload[field])
                provenance[field] = FieldProvenance(
                    source="declared",
                    format_version=version,
                )
            else:
                rule = get_default_rule(defaults_version, field)
                value = rule.materialize()
                fields[field] = value
                provenance[field] = FieldProvenance(
                    source="default",
                    format_version=version,
                    defaults_version=defaults_version,
                    rule_id=rule.rule_id,
                    default_value=deepcopy(value),
                )

        normalized[name] = WorkloadTarget(
            name=name,
            fields=fields,
            provenance=provenance,
            source_format_version=version,
        )

    return TargetSet(
        workloads=normalized,
        declaration_format_versions=(version,),
        defaults_version=defaults_version,
        declaration_digest=content_digest(raw),
    )
