"""Plan loading, strict validation and incremental plan changes."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .deps import build_dependency_graph
from .errors import DependencyError, PlanError, SampleError
from .models import LayerSpec, Plan, SUPPORTED_PRECISIONS, Sample, SampleSet
from .util import read_json


ALLOWED_LAYER_FIELDS = {
    "precision",
    "scale",
    "input_dim",
    "output_dim",
    "param_count",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    value: Any,
    *,
    field_name: str,
    layer_id: str = "<plan>",
    minimum: int = 1,
) -> int:
    if not _is_int(value) or value < minimum:
        raise PlanError(
            f"{field_name} must be an integer >= {minimum}",
            layer_id=layer_id,
            field=field_name,
        )
    return int(value)


def _parse_layers(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_layers = document.get("layers")
    if isinstance(raw_layers, dict):
        entries: list[dict[str, Any]] = []
        for raw_id in sorted(raw_layers):
            raw = raw_layers[raw_id]
            if not isinstance(raw, Mapping):
                raise PlanError(
                    "layer definition must be an object",
                    layer_id=str(raw_id),
                    field="layers",
                )
            entry = dict(raw)
            entry.setdefault("id", raw_id)
            entries.append(entry)
        return entries
    if not isinstance(raw_layers, list) or not raw_layers:
        raise PlanError(
            "layers must be a non-empty array or object",
            layer_id="<plan>",
            field="layers",
        )
    entries = []
    for index, raw in enumerate(raw_layers):
        if not isinstance(raw, Mapping):
            raise PlanError(
                "layer definition must be an object",
                layer_id=f"<index:{index}>",
                field="layers",
            )
        entries.append(dict(raw))
    return entries


def _normalize_reference_map(
    value: Any,
    *,
    field_name: str,
    layer_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {layer_id: () for layer_id in sorted(layer_ids)}
    if not isinstance(value, Mapping):
        raise DependencyError(
            f"{field_name} must be an object",
            layer_id="<plan>",
            field=field_name,
        )
    normalized: dict[str, tuple[str, ...]] = {
        layer_id: () for layer_id in sorted(layer_ids)
    }
    for source in sorted(value, key=str):
        source_id = str(source)
        if source_id not in layer_ids:
            raise DependencyError(
                "reference owner does not exist",
                layer_id=source_id,
                field=field_name,
                referrers=[source_id],
                missing_target=source_id,
            )
        targets = value[source]
        if not isinstance(targets, list):
            raise DependencyError(
                f"{field_name} values must be arrays",
                layer_id=source_id,
                field=field_name,
                referrers=[source_id],
            )
        target_list = [str(target) for target in targets]
        missing = sorted({target for target in target_list if target not in layer_ids})
        if missing:
            raise DependencyError(
                "reference target does not exist",
                layer_id=source_id,
                field=field_name,
                referrers=[source_id],
                missing_targets=missing,
            )
        normalized[source_id] = tuple(sorted(set(target_list)))
    return normalized


def plan_from_dict(document: Mapping[str, Any]) -> Plan:
    if not isinstance(document, Mapping):
        raise PlanError(
            "plan root must be an object",
            layer_id="<plan>",
            field="$",
        )

    format_version = _require_int(
        document.get("format_version", 1),
        field_name="format_version",
        minimum=1,
    )
    plan_version = _require_int(
        document.get("plan_version", 1),
        field_name="plan_version",
        minimum=1,
    )
    seed = _require_int(document.get("seed", 0), field_name="seed", minimum=0)
    block_size_kb = _require_int(
        document.get("block_size_kb"),
        field_name="block_size_kb",
        minimum=1,
    )
    memory_budget_mib = _require_int(
        document.get("memory_budget_mib"),
        field_name="memory_budget_mib",
        minimum=1,
    )
    max_batch_size = _require_int(
        document.get("max_batch_size"),
        field_name="max_batch_size",
        minimum=1,
    )

    raw_layers = _parse_layers(document)
    errors: list[PlanError] = []
    specs: list[LayerSpec] = []
    seen: dict[str, int] = {}

    for position, raw in enumerate(raw_layers):
        raw_id = raw.get("id")
        layer_id = str(raw_id) if raw_id is not None else f"<index:{position}>"
        try:
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise PlanError(
                    "layer id must be a non-empty string",
                    layer_id=layer_id,
                    field="id",
                )
            if raw_id in seen:
                raise PlanError(
                    "layer id must be unique",
                    layer_id=raw_id,
                    field="id",
                    duplicate_of_index=seen[raw_id],
                )
            input_dim = _require_int(
                raw.get("input_dim"),
                field_name="input_dim",
                layer_id=raw_id,
            )
            output_dim = _require_int(
                raw.get("output_dim"),
                field_name="output_dim",
                layer_id=raw_id,
            )
            param_count = _require_int(
                raw.get("param_count"),
                field_name="param_count",
                layer_id=raw_id,
            )
            precision = raw.get("precision")
            if precision not in SUPPORTED_PRECISIONS:
                raise PlanError(
                    f"precision must be one of {list(SUPPORTED_PRECISIONS)}",
                    layer_id=raw_id,
                    field="precision",
                )
            scale = raw.get("scale")
            if not _is_int(scale) or not 0 <= scale <= 6:
                raise PlanError(
                    "scale must be an integer in [0, 6]",
                    layer_id=raw_id,
                    field="scale",
                )
            seen[raw_id] = position
            specs.append(
                LayerSpec(
                    layer_id=raw_id,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    param_count=param_count,
                    precision=str(precision),
                    scale=int(scale),
                )
            )
        except PlanError as error:
            errors.append(error)

    if errors:
        errors.sort(key=lambda item: (str(item.details.get("layer_id")), str(item.details.get("field"))))
        raise errors[0]
    if not specs:
        raise PlanError("plan must contain at least one layer", field="layers")

    specs.sort(key=lambda layer: layer.layer_id)
    layer_ids = {layer.layer_id for layer in specs}
    consumers = _normalize_reference_map(
        document.get("consumers", {}),
        field_name="consumers",
        layer_ids=layer_ids,
    )
    memory_refs = _normalize_reference_map(
        document.get("memory_derived_from", {}),
        field_name="memory_derived_from",
        layer_ids=layer_ids,
    )
    plan = Plan(
        format_version=format_version,
        plan_version=plan_version,
        seed=seed,
        block_size_kb=block_size_kb,
        memory_budget_mib=memory_budget_mib,
        max_batch_size=max_batch_size,
        layers=tuple(specs),
        consumers=consumers,
        memory_derived_from=memory_refs,
    )
    build_dependency_graph(plan)
    return plan


def load_plan(path: str | Path) -> Plan:
    try:
        document = read_json(path)
    except (OSError, ValueError) as error:
        raise PlanError(
            f"cannot read plan: {error}",
            layer_id="<plan>",
            field="file",
        ) from error
    return plan_from_dict(document)


def sample_set_from_dict(document: Mapping[str, Any]) -> SampleSet:
    if not isinstance(document, Mapping):
        raise SampleError("sample root must be an object")
    format_version = document.get("format_version", 1)
    seed = document.get("seed", 0)
    dim = document.get("dim")
    if not _is_int(format_version) or format_version < 1:
        raise SampleError("sample format_version must be a positive integer")
    if not _is_int(seed) or seed < 0:
        raise SampleError("sample seed must be a non-negative integer")
    if not _is_int(dim) or dim <= 0:
        raise SampleError("sample dim must be a positive integer")
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise SampleError("samples must be a non-empty array")
    parsed: list[Sample] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise SampleError(f"sample {index} must be an object", sample_index=index)
        sample_id = raw.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SampleError("sample id must be a non-empty string", sample_index=index)
        if sample_id in seen:
            raise SampleError("sample id must be unique", sample_id=sample_id)
        values = raw.get("values")
        if not isinstance(values, list) or len(values) != dim:
            raise SampleError(
                "sample values length must equal dim",
                sample_id=sample_id,
                expected_dim=dim,
            )
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SampleError(
                    "sample values must be finite numbers",
                    sample_id=sample_id,
                )
            float_value = float(value)
            if not (-3.4028235e38 <= float_value <= 3.4028235e38):
                raise SampleError(
                    "sample value is outside float32 range",
                    sample_id=sample_id,
                )
            normalized.append(float_value)
        seen.add(sample_id)
        parsed.append(Sample(sample_id=sample_id, values=tuple(normalized)))
    return SampleSet(
        format_version=int(format_version),
        seed=int(seed),
        dim=int(dim),
        samples=tuple(parsed),
    )


def load_samples(path: str | Path) -> SampleSet:
    try:
        document = read_json(path)
    except (OSError, ValueError) as error:
        raise SampleError(f"cannot read samples: {error}") from error
    return sample_set_from_dict(document)


def apply_plan_changes(plan: Plan, changes: Mapping[str, Any]) -> tuple[Plan, dict[str, str], list[dict[str, Any]]]:
    """Apply layer/reference changes and return normalized plan and trigger map."""

    if not isinstance(changes, Mapping):
        raise PlanError("changes must be an object", field="$")
    document = plan.to_dict()
    raw_by_id = {layer["id"]: layer for layer in document["layers"]}
    changed: dict[str, str] = {}
    change_details: list[dict[str, Any]] = []

    updates = changes.get("layer_updates", {})
    if updates is None:
        updates = {}
    if not isinstance(updates, Mapping):
        raise PlanError("layer_updates must be an object", field="layer_updates")
    for raw_id in sorted(updates, key=str):
        layer_id = str(raw_id)
        update = updates[raw_id]
        if layer_id not in raw_by_id:
            raise DependencyError(
                "cannot update a missing layer",
                layer_id=layer_id,
                missing_targets=[layer_id],
                referrers=[],
            )
        if not isinstance(update, Mapping):
            raise PlanError(
                "layer update must be an object",
                layer_id=layer_id,
                field="layer_updates",
            )
        for field_name in sorted(update, key=str):
            if field_name not in ALLOWED_LAYER_FIELDS:
                raise PlanError(
                    f"field {field_name} is not incrementally updatable",
                    layer_id=layer_id,
                    field=str(field_name),
                )
            old_value = raw_by_id[layer_id].get(field_name)
            new_value = update[field_name]
            if old_value != new_value:
                raw_by_id[layer_id][field_name] = new_value
                changed[layer_id] = f"{field_name}_change"
                change_details.append(
                    {
                        "layer_id": layer_id,
                        "field": str(field_name),
                        "old": old_value,
                        "new": new_value,
                        "reason": f"{field_name}_change",
                    }
                )

    additions = changes.get("add_layers", [])
    if additions is None:
        additions = []
    if not isinstance(additions, list):
        raise PlanError("add_layers must be an array", field="add_layers")
    for raw in additions:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise PlanError("added layer must be an object with string id", field="add_layers")
        layer_id = str(raw["id"])
        if layer_id in raw_by_id:
            raise PlanError(
                "cannot add a duplicate layer",
                layer_id=layer_id,
                field="id",
            )
        entry = dict(raw)
        raw_by_id[layer_id] = entry
        changed[layer_id] = "layer_added"
        change_details.append({"layer_id": layer_id, "field": "*", "reason": "layer_added"})

    for key, reason in (("consumer_updates", "reference_change"), ("memory_updates", "memory_reference_change")):
        reference_updates = changes.get(key, {})
        if reference_updates is None:
            reference_updates = {}
        if not isinstance(reference_updates, Mapping):
            raise PlanError(f"{key} must be an object", field=key)
        target_map = document["consumers"] if key == "consumer_updates" else document["memory_derived_from"]
        for raw_id in sorted(reference_updates, key=str):
            source_id = str(raw_id)
            if source_id not in raw_by_id:
                raise DependencyError(
                    "reference owner does not exist",
                    layer_id=source_id,
                    referrers=[source_id],
                    missing_targets=[source_id],
                )
            new_targets = reference_updates[raw_id]
            if not isinstance(new_targets, list):
                raise PlanError(
                    f"{key} values must be arrays",
                    layer_id=source_id,
                    field=key,
                )
            normalized_targets = sorted({str(item) for item in new_targets})
            old_targets = sorted(target_map.get(source_id, []))
            if normalized_targets != old_targets:
                target_map[source_id] = normalized_targets
                changed[source_id] = reason
                change_details.append(
                    {
                        "layer_id": source_id,
                        "field": key,
                        "old": old_targets,
                        "new": normalized_targets,
                        "reason": reason,
                    }
                )

    delete_ids = changes.get("delete_layers", [])
    if delete_ids is None:
        delete_ids = []
    if not isinstance(delete_ids, list):
        raise PlanError("delete_layers must be an array", field="delete_layers")
    delete_set = {str(item) for item in delete_ids}
    missing_deletes = sorted(delete_set - set(raw_by_id))
    if missing_deletes:
        raise DependencyError(
            "cannot delete a missing layer",
            layer_id=missing_deletes[0],
            missing_targets=missing_deletes,
            referrers=[],
        )
    if delete_set:
        referrers: dict[str, list[str]] = {layer_id: [] for layer_id in sorted(delete_set)}
        for producer in sorted(document["consumers"]):
            if producer in delete_set:
                continue
            for target in document["consumers"].get(producer, []):
                if target in delete_set:
                    referrers[target].append(producer)
        for owner in sorted(document["memory_derived_from"]):
            if owner in delete_set:
                continue
            for target in document["memory_derived_from"].get(owner, []):
                if target in delete_set:
                    referrers[target].append(owner)
        blocked = {layer_id: sorted(set(names)) for layer_id, names in referrers.items() if names}
        if blocked:
            first = sorted(blocked)[0]
            raise DependencyError(
                "layer is still referenced and cannot be deleted",
                layer_id=first,
                referrers=blocked[first],
                blocked_deletions=blocked,
            )
        for layer_id in sorted(delete_set):
            raw_by_id.pop(layer_id, None)
            document["consumers"].pop(layer_id, None)
            document["memory_derived_from"].pop(layer_id, None)
            changed[layer_id] = "layer_deleted"
            change_details.append({"layer_id": layer_id, "field": "*", "reason": "layer_deleted"})

    document["layers"] = [raw_by_id[layer_id] for layer_id in sorted(raw_by_id)]
    requested_version = changes.get("plan_version", plan.plan_version + 1)
    if not _is_int(requested_version) or requested_version <= plan.plan_version:
        raise PlanError(
            "incremental plan_version must be an integer greater than the base version",
            layer_id="<plan>",
            field="plan_version",
        )
    document["plan_version"] = int(requested_version)
    new_plan = plan_from_dict(document)
    return new_plan, {key: changed[key] for key in sorted(changed)}, change_details


def with_batch_size(plan: Plan, max_batch_size: int) -> Plan:
    document = plan.to_dict()
    document["max_batch_size"] = max_batch_size
    return plan_from_dict(document)
