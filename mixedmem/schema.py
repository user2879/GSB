"""Plan declaration parsing and strict, deterministic validation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, dependency_error, schema_error
from .util import digest_value


SCHEMA_VERSION = 1
PRECISIONS = ("float32", "fixed")
SUPPORTED_FIELDS = {
    "id",
    "input_dim",
    "output_dim",
    "param_count",
    "precision",
    "scale",
    "depends_on",
    "memory_derivation",
}


@dataclass(frozen=True)
class MemoryTerm:
    layer_id: str
    coefficient: int

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer_id, "coefficient": self.coefficient}


@dataclass(frozen=True)
class MemoryDerivation:
    base_bytes: int = 0
    terms: tuple[MemoryTerm, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_bytes": self.base_bytes,
            "terms": [term.to_dict() for term in self.terms],
        }


@dataclass(frozen=True)
class Layer:
    id: str
    input_dim: int
    output_dim: int
    param_count: int
    precision: str
    scale: int
    depends_on: tuple[str, ...] = ()
    memory_derivation: MemoryDerivation = MemoryDerivation()

    @property
    def weight_bytes(self) -> int:
        return 4 if self.precision == "float32" else 2

    @property
    def activation_bytes(self) -> int:
        return 4 if self.precision == "float32" else 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "param_count": self.param_count,
            "precision": self.precision,
            "scale": self.scale,
            "depends_on": list(self.depends_on),
            "memory_derivation": self.memory_derivation.to_dict(),
        }


@dataclass(frozen=True)
class Plan:
    version: int
    name: str
    seed: int
    supported_quant_scales: tuple[int, ...]
    quantization_scale_version: int
    block_size_bytes: int
    memory_budget_bytes: int
    batch_size_max: int
    lease_ttl_ms: int
    layers: tuple[Layer, ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def layer_by_id(self) -> dict[str, Layer]:
        return {layer.id: layer for layer in self.layers}

    @property
    def layer_index(self) -> dict[str, int]:
        return {layer.id: index for index, layer in enumerate(self.layers)}

    @property
    def block_count(self) -> int:
        return self.memory_budget_bytes // self.block_size_bytes

    @property
    def content_digest(self) -> str:
        return digest_value(self.to_dict())

    @property
    def declaration_digest(self) -> str:
        payload = self.to_dict()
        payload.pop("version", None)
        return digest_value(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "name": self.name,
            "seed": self.seed,
            "supported_quant_scales": list(self.supported_quant_scales),
            "quantization_scale_version": self.quantization_scale_version,
            "block_size_bytes": self.block_size_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "batch_size_max": self.batch_size_max,
            "lease_ttl_ms": self.lease_ttl_ms,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def with_layer_change(self, layer_id: str, updates: dict[str, Any]) -> "Plan":
        return change_layer(self, layer_id, updates)


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise schema_error("must be a JSON object", field=field)
    return value


def _require_int(
    value: Any,
    field: str,
    *,
    layer_id: str | None = None,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise schema_error("must be an integer", layer_id=layer_id, field=field)
    if minimum is not None and value < minimum:
        raise schema_error(
            f"must be >= {minimum}", layer_id=layer_id, field=field
        )
    return value


def _require_string(value: Any, field: str, *, layer_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise schema_error(
            "must be a non-empty string", layer_id=layer_id, field=field
        )
    if value != value.strip():
        raise schema_error(
            "must not have surrounding whitespace", layer_id=layer_id, field=field
        )
    return value


def _parse_memory_derivation(
    value: Any, layer_id: str
) -> MemoryDerivation:
    if value is None:
        return MemoryDerivation()
    obj = _require_object(value, "memory_derivation")
    unknown = sorted(set(obj) - {"base_bytes", "terms"})
    if unknown:
        raise schema_error(
            f"unknown derivation field: {unknown[0]}",
            layer_id=layer_id,
            field=f"memory_derivation.{unknown[0]}",
        )
    base = _require_int(
        obj.get("base_bytes", 0),
        "memory_derivation.base_bytes",
        layer_id=layer_id,
        minimum=0,
    )
    raw_terms = obj.get("terms", [])
    if not isinstance(raw_terms, list):
        raise schema_error(
            "must be an array",
            layer_id=layer_id,
            field="memory_derivation.terms",
        )
    terms: list[MemoryTerm] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_terms):
        term = _require_object(raw, f"memory_derivation.terms[{index}]")
        unknown_term = sorted(set(term) - {"layer", "coefficient"})
        if unknown_term:
            raise schema_error(
                f"unknown term field: {unknown_term[0]}",
                layer_id=layer_id,
                field=f"memory_derivation.terms[{index}].{unknown_term[0]}",
            )
        target = _require_string(
            term.get("layer"),
            f"memory_derivation.terms[{index}].layer",
            layer_id=layer_id,
        )
        coefficient = _require_int(
            term.get("coefficient", 1),
            f"memory_derivation.terms[{index}].coefficient",
            layer_id=layer_id,
            minimum=0,
        )
        if target in seen:
            raise schema_error(
                f"duplicate memory term for {target}",
                layer_id=layer_id,
                field=f"memory_derivation.terms[{index}].layer",
            )
        if target == layer_id:
            raise schema_error(
                "layer memory derivation cannot reference itself",
                layer_id=layer_id,
                field=f"memory_derivation.terms[{index}].layer",
            )
        seen.add(target)
        terms.append(MemoryTerm(target, coefficient))
    return MemoryDerivation(base, tuple(terms))


def _parse_layers(raw_layers: Any) -> tuple[Layer, ...]:
    if not isinstance(raw_layers, list) or not raw_layers:
        raise schema_error("layers must be a non-empty array", field="layers")
    layers: list[Layer] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_layers):
        if not isinstance(raw, dict):
            raise schema_error(
                "layer must be a JSON object", field=f"layers[{index}]"
            )
        layer_id_raw = raw.get("id")
        layer_id = layer_id_raw if isinstance(layer_id_raw, str) else f"layers[{index}]"
        unknown = sorted(set(raw) - SUPPORTED_FIELDS)
        if unknown:
            raise schema_error(
                f"unknown layer field: {unknown[0]}",
                layer_id=layer_id,
                field=unknown[0],
            )
        layer_id = _require_string(raw.get("id"), "id", layer_id=layer_id)
        if layer_id in ids:
            raise schema_error("duplicate layer id", layer_id=layer_id, field="id")
        ids.add(layer_id)
        input_dim = _require_int(
            raw.get("input_dim"), "input_dim", layer_id=layer_id, minimum=1
        )
        output_dim = _require_int(
            raw.get("output_dim"), "output_dim", layer_id=layer_id, minimum=1
        )
        param_count = _require_int(
            raw.get("param_count"), "param_count", layer_id=layer_id, minimum=1
        )
        precision = raw.get("precision")
        if precision not in PRECISIONS:
            raise schema_error(
                f"must be one of {list(PRECISIONS)}",
                layer_id=layer_id,
                field="precision",
            )
        scale = _require_int(
            raw.get("scale"), "scale", layer_id=layer_id, minimum=1
        )
        depends_raw = raw.get("depends_on", [])
        if not isinstance(depends_raw, list):
            raise schema_error(
                "must be an array", layer_id=layer_id, field="depends_on"
            )
        depends: list[str] = []
        for dep_index, dep in enumerate(depends_raw):
            dep_id = _require_string(
                dep, f"depends_on[{dep_index}]", layer_id=layer_id
            )
            if dep_id in depends:
                raise schema_error(
                    "duplicate dependency",
                    layer_id=layer_id,
                    field=f"depends_on[{dep_index}]",
                )
            if dep_id == layer_id:
                raise schema_error(
                    "layer cannot depend on itself",
                    layer_id=layer_id,
                    field=f"depends_on[{dep_index}]",
                )
            depends.append(dep_id)
        derivation = _parse_memory_derivation(
            raw.get("memory_derivation"), layer_id
        )
        layers.append(
            Layer(
                id=layer_id,
                input_dim=input_dim,
                output_dim=output_dim,
                param_count=param_count,
                precision=precision,
                scale=scale,
                depends_on=tuple(depends),
                memory_derivation=derivation,
            )
        )

    by_id = {layer.id: layer for layer in layers}
    for layer in layers:
        for index, dep_id in enumerate(layer.depends_on):
            if dep_id not in by_id:
                raise schema_error(
                    f"reference target does not exist: {dep_id}",
                    layer_id=layer.id,
                    field=f"depends_on[{index}]",
                    missing=dep_id,
                )
        for index, term in enumerate(layer.memory_derivation.terms):
            if term.layer_id not in by_id:
                raise schema_error(
                    f"derivation target does not exist: {term.layer_id}",
                    layer_id=layer.id,
                    field=f"memory_derivation.terms[{index}].layer",
                    missing=term.layer_id,
                )
        if layer.depends_on:
            supplied = sum(by_id[dep].output_dim for dep in layer.depends_on)
            if supplied != layer.input_dim:
                raise schema_error(
                    "input_dim must equal the sum of referenced output dimensions",
                    layer_id=layer.id,
                    field="input_dim",
                    expected=supplied,
                    actual=layer.input_dim,
                    references=list(layer.depends_on),
                )
    return tuple(layers)


def _validate_acyclic(layers: tuple[Layer, ...]) -> None:
    by_id = {layer.id: layer for layer in layers}
    rank = {layer.id: index for index, layer in enumerate(layers)}
    adjacency: dict[str, set[str]] = {layer.id: set() for layer in layers}
    indegree: dict[str, int] = {layer.id: 0 for layer in layers}
    for layer in layers:
        targets = list(layer.depends_on) + [
            term.layer_id for term in layer.memory_derivation.terms
        ]
        for target in targets:
            if layer.id != target and layer.id not in adjacency[target]:
                adjacency[target].add(layer.id)
                indegree[layer.id] += 1
    ready = sorted(
        (layer_id for layer_id, degree in indegree.items() if degree == 0),
        key=lambda item: (rank[item], item),
    )
    visited: list[str] = []
    while ready:
        current = ready.pop(0)
        visited.append(current)
        for dependent in sorted(adjacency[current], key=lambda item: (rank[item], item)):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
        ready.sort(key=lambda item: (rank[item], item))
    if len(visited) != len(layers):
        remaining = sorted(
            (layer_id for layer_id, degree in indegree.items() if degree > 0),
            key=lambda item: (rank[item], item),
        )
        first = remaining[0]
        raise schema_error(
            "dependency graph contains a cycle",
            layer_id=first,
            field="dependency_graph",
            cycle_members=remaining,
        )


def parse_plan(raw: Any) -> Plan:
    """Parse and validate a plan, rejecting the whole declaration on failure."""

    obj = _require_object(raw, "$")
    required = {
        "schema_version",
        "version",
        "name",
        "seed",
        "supported_quant_scales",
        "quantization_scale_version",
        "block_size_bytes",
        "memory_budget_bytes",
        "batch_size_max",
        "lease_ttl_ms",
        "layers",
    }
    missing = sorted(required - set(obj))
    if missing:
        raise schema_error(f"missing field: {missing[0]}", field=missing[0])
    unknown = sorted(set(obj) - required)
    if unknown:
        raise schema_error(f"unknown field: {unknown[0]}", field=unknown[0])
    schema_version = _require_int(
        obj["schema_version"], "schema_version", minimum=1
    )
    if schema_version != SCHEMA_VERSION:
        raise schema_error(
            f"unsupported schema_version: {schema_version}",
            field="schema_version",
            supported=SCHEMA_VERSION,
        )
    version = _require_int(obj["version"], "version", minimum=1)
    name = _require_string(obj["name"], "name")
    seed = _require_int(obj["seed"], "seed", minimum=0)
    scale_version = _require_int(
        obj["quantization_scale_version"],
        "quantization_scale_version",
        minimum=1,
    )
    raw_scales = obj["supported_quant_scales"]
    if not isinstance(raw_scales, list) or not raw_scales:
        raise schema_error(
            "must be a non-empty array", field="supported_quant_scales"
        )
    scales: list[int] = []
    for index, raw_scale in enumerate(raw_scales):
        scale = _require_int(
            raw_scale, f"supported_quant_scales[{index}]", minimum=1
        )
        if scale in scales:
            raise schema_error(
                "duplicate supported scale", field=f"supported_quant_scales[{index}]"
            )
        scales.append(scale)
    if tuple(scales) != tuple(sorted(scales)):
        raise schema_error(
            "supported_quant_scales must be ascending",
            field="supported_quant_scales",
        )
    block_size = _require_int(
        obj["block_size_bytes"], "block_size_bytes", minimum=1
    )
    budget = _require_int(
        obj["memory_budget_bytes"], "memory_budget_bytes", minimum=block_size
    )
    batch_max = _require_int(obj["batch_size_max"], "batch_size_max", minimum=1)
    ttl = _require_int(obj["lease_ttl_ms"], "lease_ttl_ms", minimum=1)
    layers = _parse_layers(obj["layers"])
    by_id = {layer.id: layer for layer in layers}
    for layer in layers:
        if layer.scale not in scales:
            raise schema_error(
                "scale must belong to supported_quant_scales",
                layer_id=layer.id,
                field="scale",
                supported=scales,
            )
    _validate_acyclic(layers)
    return Plan(
        version=version,
        name=name,
        seed=seed,
        supported_quant_scales=tuple(scales),
        quantization_scale_version=scale_version,
        block_size_bytes=block_size,
        memory_budget_bytes=budget,
        batch_size_max=batch_max,
        lease_ttl_ms=ttl,
        layers=layers,
        schema_version=schema_version,
    )


def load_plan(path: str | Path) -> Plan:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise schema_error(
            f"invalid JSON at line {exc.lineno} column {exc.colno}", field="$"
        ) from exc
    except OSError as exc:
        raise DiagnosticError(
            "E_IO", f"cannot read plan: {exc}", field="plan_path"
        ) from exc
    return parse_plan(raw)


def next_version(plan: Plan) -> Plan:
    return replace(plan, version=plan.version + 1)


def change_layers(plan: Plan, edits: dict[str, dict[str, Any]]) -> Plan:
    """Atomically apply one or more layer edits and increment version once."""

    if not isinstance(edits, dict) or not edits:
        raise schema_error("edits must be a non-empty object", field="edits")
    raw = plan.to_dict()
    indexes = {
        layer["id"]: index for index, layer in enumerate(raw["layers"])
    }
    for layer_id, updates in edits.items():
        if not isinstance(updates, dict) or not updates:
            raise schema_error(
                "each layer edit must be a non-empty object",
                layer_id=layer_id,
                field="edits",
            )
        if layer_id not in indexes:
            raise dependency_error(
                "layer does not exist", layer_id=layer_id, field="id"
            )
        target = raw["layers"][indexes[layer_id]]
        unknown = sorted(set(updates) - SUPPORTED_FIELDS - {"id"})
        if unknown:
            raise schema_error(
                f"unknown layer field: {unknown[0]}", layer_id=layer_id, field=unknown[0]
            )
        if "id" in updates and updates["id"] != layer_id:
            raise schema_error(
                "renaming a layer is unsupported", layer_id=layer_id, field="id"
            )
        target.update(copy.deepcopy(updates))
    raw["version"] = plan.version + 1
    return parse_plan(raw)


def change_layer(plan: Plan, layer_id: str, updates: dict[str, Any]) -> Plan:
    """Apply a single declaration change and increment the plan version once."""

    return change_layers(plan, {layer_id: updates})


def remove_layer(plan: Plan, layer_id: str) -> Plan:
    """Delete a layer unless activation or memory references still target it."""

    if layer_id not in plan.layer_by_id:
        raise dependency_error(
            "layer does not exist", layer_id=layer_id, field="id"
        )
    referrers: set[str] = set()
    for layer in plan.layers:
        if layer_id in layer.depends_on:
            referrers.add(layer.id)
        if any(term.layer_id == layer_id for term in layer.memory_derivation.terms):
            referrers.add(layer.id)
    referrers.discard(layer_id)
    if referrers:
        raise dependency_error(
            "cannot delete a layer that is still referenced",
            layer_id=layer_id,
            field="depends_on",
            referrers=sorted(referrers),
        )
    raw = plan.to_dict()
    raw["layers"] = [
        layer for layer in raw["layers"] if layer["id"] != layer_id
    ]
    if not raw["layers"]:
        raise dependency_error(
            "cannot delete the last layer", layer_id=layer_id, field="layers"
        )
    raw["version"] = plan.version + 1
    return parse_plan(raw)


def referenced_by(plan: Plan, layer_id: str) -> list[str]:
    result: list[str] = []
    for layer in plan.layers:
        if layer.id == layer_id:
            continue
        if layer_id in layer.depends_on or any(
            term.layer_id == layer_id for term in layer.memory_derivation.terms
        ):
            result.append(layer.id)
    return result


def layer_memory_bytes(
    plan: Plan,
    layer_id: str,
    batch_size: int,
    *,
    _stack: tuple[str, ...] = (),
) -> int:
    """Derive aligned layer occupancy for a concrete batch size."""

    if batch_size <= 0:
        raise schema_error("batch_size must be positive", field="batch_size")
    by_id = plan.layer_by_id
    if layer_id not in by_id:
        raise dependency_error(
            "layer does not exist", layer_id=layer_id, field="id"
        )
    if layer_id in _stack:
        raise schema_error(
            "memory derivation cycle", layer_id=layer_id, field="memory_derivation"
        )
    layer = by_id[layer_id]
    own = (
        layer.memory_derivation.base_bytes
        + layer.param_count * layer.weight_bytes
        + layer.output_dim * batch_size * layer.activation_bytes
    )
    derived = 0
    for term in layer.memory_derivation.terms:
        derived += term.coefficient * layer_memory_bytes(
            plan, term.layer_id, batch_size, _stack=_stack + (layer_id,)
        )
    raw = own + derived
    block_size = plan.block_size_bytes
    aligned = ((raw + block_size - 1) // block_size) * block_size
    return max(block_size, aligned)


def layer_block_count(plan: Plan, layer_id: str, batch_size: int) -> int:
    return layer_memory_bytes(plan, layer_id, batch_size) // plan.block_size_bytes


def memory_profile(plan: Plan, batch_size: int) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for layer in plan.layers:
        size = layer_memory_bytes(plan, layer.id, batch_size)
        result[layer.id] = {
            "bytes": size,
            "blocks": size // plan.block_size_bytes,
        }
    return result


def ensure_samples_within_batch_max(plan: Plan, sample_count: int) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise schema_error("sample_count must be an integer", field="sample_count")
    if sample_count <= 0:
        raise schema_error("sample_count must be positive", field="sample_count")


def plan_diff(left: Plan, right: Plan) -> list[dict[str, Any]]:
    """Stable field-level plan differences used by snapshot replacement."""

    ld, rd = left.to_dict(), right.to_dict()
    keys = sorted(set(ld) | set(rd))
    changes: list[dict[str, Any]] = []
    for key in keys:
        if key == "layers":
            continue
        if ld.get(key) != rd.get(key):
            changes.append(
                {
                    "item": key,
                    "left": ld.get(key, "<missing>"),
                    "right": rd.get(key, "<missing>"),
                }
            )
    left_layers = {item["id"]: item for item in ld["layers"]}
    right_layers = {item["id"]: item for item in rd["layers"]}
    for layer_id in sorted(set(left_layers) | set(right_layers)):
        if layer_id not in left_layers:
            changes.append(
                {"item": f"layers.{layer_id}", "left": "<missing>", "right": right_layers[layer_id]}
            )
            continue
        if layer_id not in right_layers:
            changes.append(
                {"item": f"layers.{layer_id}", "left": left_layers[layer_id], "right": "<missing>"}
            )
            continue
        lv, rv = left_layers[layer_id], right_layers[layer_id]
        for field in sorted(set(lv) | set(rv)):
            if lv.get(field) != rv.get(field):
                changes.append(
                    {
                        "item": f"layers.{layer_id}.{field}",
                        "left": lv.get(field, "<missing>"),
                        "right": rv.get(field, "<missing>"),
                    }
                )
    return changes
