"""Change propagation and byte-stable partial drift recomputation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_json
from .config import EvaluationConfig
from .declarations import _validate_field_value
from .defaults import get_default_rule, get_rule_set
from .drift import aggregate_all, compare_field
from .errors import format_error, reference_error
from .models import FIELD_INDEX, FIELD_ORDER, FieldProvenance, TargetSet, WorkloadTarget


@dataclass(frozen=True)
class TargetUpdate:
    workload: str
    field: str
    value: Any


@dataclass(frozen=True)
class ChangeSet:
    target_updates: tuple[TargetUpdate, ...] = ()
    format_from: int | None = None
    format_to: int | None = None
    new_defaults_version: str | None = None
    deleted_workloads: tuple[str, ...] = ()

    @property
    def has_format_upgrade(self) -> bool:
        return self.format_to is not None


@dataclass
class PropagationResult:
    target: TargetSet
    decisions: list[dict[str, Any]]
    workload_results: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


def parse_change_set(raw: Any) -> ChangeSet:
    if not isinstance(raw, dict):
        raise format_error("change set must be an object", field="<root>")
    allowed = {"target_updates", "format_upgrade", "default_rules_change", "delete_workloads"}
    for key in raw:
        if key not in allowed:
            raise format_error(f"unknown change-set field: {key}", field=key)

    updates: list[TargetUpdate] = []
    raw_updates = raw.get("target_updates", [])
    if not isinstance(raw_updates, list):
        raise format_error("target_updates must be an array", field="target_updates")
    for index, update in enumerate(raw_updates):
        if not isinstance(update, dict):
            raise format_error("target update must be an object", item=index, field="<update>")
        if set(update) != {"workload", "field", "value"}:
            missing = sorted({"workload", "field", "value"} - set(update))
            extra = sorted(set(update) - {"workload", "field", "value"})
            raise format_error(
                "target update keys are invalid",
                item=index,
                field=(extra or missing or ["<update>"])[0],
            )
        if not isinstance(update["workload"], str) or not isinstance(update["field"], str):
            raise format_error("target update workload/field must be strings", item=index)
        updates.append(TargetUpdate(update["workload"], update["field"], deepcopy(update["value"])))

    format_from: int | None = None
    format_to: int | None = None
    if "format_upgrade" in raw:
        upgrade = raw["format_upgrade"]
        if not isinstance(upgrade, dict) or set(upgrade) != {"from", "to"}:
            raise format_error("format_upgrade must contain only from and to", field="format_upgrade")
        format_from, format_to = upgrade["from"], upgrade["to"]
        if (
            isinstance(format_from, bool)
            or not isinstance(format_from, int)
            or isinstance(format_to, bool)
            or not isinstance(format_to, int)
            or format_from not in (1, 2, 3)
            or format_to not in (1, 2, 3)
        ):
            raise format_error("format versions must be integers in 1..3", field="format_upgrade")
        if format_to < format_from:
            raise format_error("format_upgrade cannot downgrade a declaration", field="format_upgrade")

    new_defaults_version: str | None = None
    if "default_rules_change" in raw:
        defaults_change = raw["default_rules_change"]
        if not isinstance(defaults_change, dict) or set(defaults_change) != {"to"}:
            raise format_error(
                "default_rules_change must contain only to", field="default_rules_change"
            )
        new_defaults_version = defaults_change["to"]
        if not isinstance(new_defaults_version, str):
            raise format_error("defaults version must be a string", field="default_rules_change")
        get_rule_set(new_defaults_version)

    deleted = raw.get("delete_workloads", [])
    if not isinstance(deleted, list) or not all(isinstance(name, str) and name for name in deleted):
        raise format_error("delete_workloads must be an array of names", field="delete_workloads")

    return ChangeSet(
        target_updates=tuple(updates),
        format_from=format_from,
        format_to=format_to,
        new_defaults_version=new_defaults_version,
        deleted_workloads=tuple(deleted),
    )


def _clone_target(target: TargetSet) -> TargetSet:
    return TargetSet.from_dict(target.to_dict())


def _validate_parent_graph(workloads: dict[str, WorkloadTarget]) -> None:
    for name in sorted(workloads):
        parent = workloads[name].fields["parent_ref"]
        if parent is not None and parent not in workloads:
            raise reference_error(
                f"parent_ref points to missing workload: {parent}",
                item=name,
                field="parent_ref",
                details={"referenced_workload": parent},
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(name: str, path: list[str]) -> list[str] | None:
        if name in visiting:
            return path + [name]
        if name in visited:
            return None
        visiting.add(name)
        parent = workloads[name].fields["parent_ref"]
        if parent is not None:
            cycle = walk(parent, path + [name])
            if cycle:
                return cycle
        visiting.remove(name)
        visited.add(name)
        return None

    for name in sorted(workloads):
        cycle = walk(name, [])
        if cycle:
            raise reference_error(
                "cyclic parent_ref chain",
                item=name,
                field="parent_ref",
                details={"cycle": cycle},
            )


def apply_change_set(
    target: TargetSet,
    actual_state: dict[str, dict[str, Any]],
    decisions: list[dict[str, Any]],
    config: EvaluationConfig,
    change_set: ChangeSet,
) -> PropagationResult:
    config.validate()
    modified = _clone_target(target)
    modified_actual = deepcopy(actual_state)
    old_decisions = {
        (decision["workload"], decision["field"]): deepcopy(decision)
        for decision in decisions
    }
    reasons: dict[tuple[str, str], list[str]] = {}

    def mark(workload: str, field: str, reason: str) -> None:
        reasons.setdefault((workload, field), []).append(reason)

    for update in change_set.target_updates:
        if update.workload not in modified.workloads:
            raise format_error(
                f"target update references unknown workload: {update.workload}",
                item=update.workload,
                field="workload",
            )
        if update.field not in FIELD_ORDER:
            raise format_error(
                f"target update references unknown field: {update.field}",
                item=update.workload,
                field=update.field,
            )
        _validate_field_value(update.field, update.value, update.workload)
        workload = modified.workloads[update.workload]
        old_value = deepcopy(workload.fields[update.field])
        workload.fields[update.field] = deepcopy(update.value)
        workload.provenance[update.field] = FieldProvenance(
            source="declared",
            format_version=workload.source_format_version,
        )
        mark(
            update.workload,
            update.field,
            f"target_update:{update.field}:{canonical_json(old_value)}->{canonical_json(update.value)}",
        )

    if change_set.has_format_upgrade:
        assert change_set.format_to is not None
        current_versions = set(modified.declaration_format_versions)
        if change_set.format_from is not None and change_set.format_from not in current_versions:
            raise format_error(
                "format_upgrade.from does not match the current declaration version",
                field="format_upgrade.from",
                details={
                    "from": change_set.format_from,
                    "current": sorted(current_versions),
                },
            )
        for workload in modified.workloads.values():
            workload.source_format_version = change_set.format_to
            for field in FIELD_ORDER:
                workload.provenance[field] = FieldProvenance(
                    source=workload.provenance[field].source,
                    format_version=change_set.format_to,
                    defaults_version=workload.provenance[field].defaults_version,
                    rule_id=workload.provenance[field].rule_id,
                    default_value=deepcopy(workload.provenance[field].default_value),
                )
                mark(
                    workload.name,
                    field,
                    f"format_upgrade:{change_set.format_from}->{change_set.format_to}",
                )
        modified.declaration_format_versions = (change_set.format_to,)

    if change_set.new_defaults_version is not None:
        new_version = change_set.new_defaults_version
        old_version = modified.defaults_version
        for workload in modified.workloads.values():
            for field in FIELD_ORDER:
                provenance = workload.provenance[field]
                if provenance.source != "default":
                    continue
                rule = get_default_rule(new_version, field)
                new_value = rule.materialize()
                workload.fields[field] = deepcopy(new_value)
                workload.provenance[field] = FieldProvenance(
                    source="default",
                    format_version=workload.source_format_version,
                    defaults_version=new_version,
                    rule_id=rule.rule_id,
                    default_value=deepcopy(new_value),
                )
                mark(
                    workload.name,
                    field,
                    f"default_rule_change:{old_version}->{new_version}:{rule.rule_id}",
                )
        modified.defaults_version = new_version

    delete_set = set(change_set.deleted_workloads)
    for workload_name in sorted(delete_set):
        if workload_name not in modified.workloads:
            raise format_error(
                f"cannot delete unknown workload: {workload_name}",
                item=workload_name,
                field="delete_workloads",
            )

    if delete_set:
        referrers = [
            {
                "workload": workload.name,
                "field": "parent_ref",
                "references": workload.fields["parent_ref"],
            }
            for workload in modified.ordered_workloads()
            if workload.name not in delete_set and workload.fields["parent_ref"] in delete_set
        ]
        if referrers:
            raise reference_error(
                "cannot delete workloads that are still referenced by parent_ref",
                item=referrers[0]["references"],
                field="parent_ref",
                details={
                    "referrers": referrers,
                    "deleted_workloads": sorted(delete_set),
                },
            )
        for workload_name in sorted(delete_set):
            for field in FIELD_ORDER:
                mark(workload_name, field, f"workload_deleted:{workload_name}")
            del modified.workloads[workload_name]
            modified_actual.pop(workload_name, None)

    if change_set.target_updates or change_set.has_format_upgrade or delete_set:
        _validate_parent_graph(modified.workloads)

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    removed_keys: set[tuple[str, str]] = set()
    for key, decision in old_decisions.items():
        if key[0] not in modified.workloads:
            removed_keys.add(key)
        elif key in reasons:
            continue
        else:
            by_key[key] = decision

    for workload in modified.ordered_workloads():
        for field in FIELD_ORDER:
            key = (workload.name, field)
            if key in reasons or key not in by_key:
                by_key[key] = compare_field(
                    workload,
                    field,
                    modified_actual,
                    scale_exp=config.scale_exp,
                )

    new_decisions = sorted(
        by_key.values(), key=lambda decision: (decision["workload"], FIELD_INDEX[decision["field"]])
    )
    workload_results = aggregate_all(modified, new_decisions, config)

    affected_fields = [
        {"workload": workload, "field": field, "reasons": reasons[(workload, field)]}
        for workload, field in sorted(reasons, key=lambda key: (key[0], FIELD_INDEX[key[1]]))
    ]
    affected_keys = set(reasons)
    unaffected_keys = sorted(
        (
            key
            for key in by_key
            if key not in affected_keys and key not in removed_keys
        ),
        key=lambda key: (key[0], FIELD_INDEX[key[1]]),
    )
    unaffected_stable = all(
        canonical_json(old_decisions[key]) == canonical_json(by_key[key])
        for key in unaffected_keys
        if key in old_decisions
    )

    report = {
        "affected_fields": affected_fields,
        "affected_decisions": [f"{workload}/{field}" for workload, field in affected_keys],
        "removed_decisions": [f"{workload}/{field}" for workload, field in sorted(removed_keys)],
        "unaffected_decisions": [f"{workload}/{field}" for workload, field in unaffected_keys],
        "unaffected_byte_stable": unaffected_stable,
        "deleted_workloads": sorted(delete_set),
        "target_digest": modified.target_digest,
        "declaration_format_versions": list(modified.declaration_format_versions),
        "defaults_version": modified.defaults_version,
    }
    report["affected_decisions"].sort(
        key=lambda path: (
            path.split("/", 1)[0],
            FIELD_INDEX[path.split("/", 1)[1]],
        )
    )
    report["removed_decisions"].sort(
        key=lambda path: (
            path.split("/", 1)[0],
            FIELD_INDEX[path.split("/", 1)[1]],
        )
    )
    return PropagationResult(
        target=modified,
        decisions=new_decisions,
        workload_results=workload_results,
        report=report,
    )
