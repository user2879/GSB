"""JSON loading with strong, deterministic validation.

The loader never partially accepts a document: the first invalid field raises
:class:`SpecError` (``E_SPEC``) with a JSON-like location path. Dictionaries are
walked in sorted key order and lists in declared order, so the reported first
failure is independent of hash randomization and file layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .constants import TAINT_EFFECTS, WEIGHT_BASIS
from .errors import SpecError
from .models import (
    LabelRequirement,
    Node,
    NodeAffinity,
    Pod,
    PodAntiAffinity,
    PreferredTerm,
    Taint,
    Toleration,
)

_LABEL_OPS = frozenset({"In", "NotIn", "Exists", "DoesNotExist"})
_TOL_OPS = frozenset({"Exists", "Equal"})
_LABEL_NAME_MAX = 317
_PREFIX_MAX = 253
_NAME_MAX = 63


# --------------------------------------------------------------------------- #
# Low level type / format checks
# --------------------------------------------------------------------------- #


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; it must never be accepted as an integer.
    return isinstance(value, int) and not isinstance(value, bool)


def _require_type(data: Any, expected: type, path: str) -> None:
    if not isinstance(data, expected):
        raise SpecError(
            f"expected {expected.__name__}, got {type(data).__name__}", path
        )


def _require_object(data: Any, path: str) -> Dict[str, Any]:
    _require_type(data, dict, path)
    return data


def _require_list(data: Any, path: str) -> List[Any]:
    _require_type(data, list, path)
    return data


def _require_positive_int(data: Any, path: str) -> int:
    if not _is_int(data):
        raise SpecError(f"expected positive integer, got {type(data).__name__}", path)
    if data <= 0:
        raise SpecError(f"expected positive integer, got {data}", path)
    return data


def _is_dns_label(token: str) -> bool:
    if not token or len(token) > _NAME_MAX:
        return False
    if not token[0].isalnum() or not token[-1].isalnum():
        return False
    return all(ch.isalnum() or ch == "-" for ch in token)


def _is_qualified_name(name: str) -> bool:
    if not name or len(name) > _LABEL_NAME_MAX:
        return False
    if "/" in name:
        prefix, short = name.split("/", 1)
        if len(prefix) > _PREFIX_MAX:
            return False
        if "." not in prefix and prefix not in {"kubernetes.io"}:
            # bare prefixes must still look like a DNS label
            if not _is_dns_label(prefix):
                return False
        else:
            labels = prefix.split(".")
            if any(not _is_dns_label(part) for part in labels):
                return False
        name = short
    if name == "*":
        return True
    return _is_dns_label(name)


def _is_label_value(value: str) -> bool:
    if value == "":
        return True
    if len(value) > _NAME_MAX:
        return False
    if not value[0].isalnum() or not value[-1].isalnum():
        return False
    return all(ch.isalnum() or ch in "-_." for ch in value)


def _validate_labels(labels: Any, path: str) -> Dict[str, str]:
    obj = _require_object(labels, path)
    for key in sorted(obj):
        key_path = f"{path}.{key}"
        if not isinstance(key, str) or not _is_qualified_name(key):
            raise SpecError(f"illegal label key {key!r}", key_path)
        value = obj[key]
        if not isinstance(value, str) or not _is_label_value(value):
            raise SpecError(f"illegal label value {value!r}", key_path)
    return {str(k): str(v) for k, v in obj.items()}


def _validate_requirement(data: Any, path: str) -> LabelRequirement:
    obj = _require_object(data, path)
    key = obj.get("key")
    if not isinstance(key, str) or not key or not _is_qualified_name(key):
        raise SpecError(f"illegal requirement key {key!r}", f"{path}.key")
    operator = obj.get("operator")
    if operator not in _LABEL_OPS:
        raise SpecError(
            f"operator must be one of {sorted(_LABEL_OPS)}, got {operator!r}",
            f"{path}.operator",
        )
    raw_values = obj.get("values", [])
    values = _require_list(raw_values, f"{path}.values")
    if operator in {"In", "NotIn"} and not values:
        raise SpecError(f"{operator} requires a non-empty values list", f"{path}.values")
    if operator in {"Exists", "DoesNotExist"} and values:
        raise SpecError(f"{operator} must not carry values", f"{path}.values")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not _is_label_value(value):
            raise SpecError(f"illegal label value {value!r}", f"{path}.values[{index}]")
    return LabelRequirement(key=key, operator=operator, values=tuple(values))


def _validate_preferred(data: Any, path: str) -> PreferredTerm:
    obj = _require_object(data, path)
    weight = obj.get("weight")
    if not _is_int(weight) or weight <= 0:
        raise SpecError(f"preferred weight must be a positive integer, got {weight!r}",
                        f"{path}.weight")
    requirement = _validate_requirement(obj.get("requirement"), f"{path}.requirement")
    return PreferredTerm(weight=weight, requirement=requirement)


# --------------------------------------------------------------------------- #
# Node / pod document validation
# --------------------------------------------------------------------------- #


def _validate_taint(data: Any, path: str) -> Taint:
    obj = _require_object(data, path)
    key = obj.get("key")
    if not isinstance(key, str) or not key or not _is_qualified_name(key):
        raise SpecError(f"illegal taint key {key!r}", f"{path}.key")
    value = obj.get("value", "")
    if not isinstance(value, str) or not _is_label_value(value):
        raise SpecError(f"illegal taint value {value!r}", f"{path}.value")
    effect = obj.get("effect")
    if effect not in TAINT_EFFECTS:
        raise SpecError(
            f"effect must be one of {sorted(TAINT_EFFECTS)}, got {effect!r}",
            f"{path}.effect",
        )
    return Taint(key=key, value=value, effect=effect)


def _validate_toleration(data: Any, path: str) -> Toleration:
    obj = _require_object(data, path)
    operator = obj.get("operator", "Equal")
    if operator not in _TOL_OPS:
        raise SpecError(
            f"toleration operator must be one of {sorted(_TOL_OPS)}, got {operator!r}",
            f"{path}.operator",
        )
    key = obj.get("key", "")
    if not isinstance(key, str):
        raise SpecError(f"illegal toleration key {key!r}", f"{path}.key")
    if operator == "Equal" and not key:
        raise SpecError("Equal toleration requires a key", f"{path}.key")
    value = obj.get("value", "")
    if not isinstance(value, str):
        raise SpecError(f"illegal toleration value {value!r}", f"{path}.value")
    if operator == "Exists" and value:
        raise SpecError("Exists toleration must not carry a value", f"{path}.value")
    effect = obj.get("effect", "")
    if effect not in ("", *TAINT_EFFECTS):
        raise SpecError(f"illegal toleration effect {effect!r}", f"{path}.effect")
    return Toleration(key=key, operator=operator, value=value, effect=effect)


def load_nodes(source: Any) -> List[Node]:
    """Validate and build nodes from a dict or a JSON file path."""
    data = _read_source(source)
    root = _require_object(data, "$")
    raw_nodes = _require_list(root.get("nodes"), "$.nodes")
    if not raw_nodes:
        raise SpecError("at least one node is required", "$.nodes")
    nodes: List[Node] = []
    seen_names: Dict[str, int] = {}
    for index, raw in enumerate(raw_nodes):
        path = f"$.nodes[{index}]"
        obj = _require_object(raw, path)
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            raise SpecError(f"illegal node name {name!r}", f"{path}.name")
        if name in seen_names:
            raise SpecError(
                f"duplicate node name {name!r} (first at $.nodes[{seen_names[name]}])",
                f"{path}.name",
            )
        seen_names[name] = index
        labels = _validate_labels(obj.get("labels", {}), f"{path}.labels")
        cpu = _require_positive_int(obj.get("cpu"), f"{path}.cpu")
        memory = _require_positive_int(obj.get("memory"), f"{path}.memory")
        taints_raw = _require_list(obj.get("taints", []), f"{path}.taints")
        taints = tuple(
            _validate_taint(item, f"{path}.taints[{i}]")
            for i, item in enumerate(taints_raw)
        )
        nodes.append(Node(name=name, labels=labels, cpu=cpu, memory=memory, taints=taints))
    return nodes


def _validate_node_affinity(data: Any, path: str) -> NodeAffinity:
    if data is None:
        return NodeAffinity()
    obj = _require_object(data, path)
    required_raw = _require_list(obj.get("required", []), f"{path}.required")
    required = tuple(
        _validate_requirement(item, f"{path}.required[{i}]")
        for i, item in enumerate(required_raw)
    )
    preferred_raw = _require_list(obj.get("preferred", []), f"{path}.preferred")
    preferred = tuple(
        _validate_preferred(item, f"{path}.preferred[{i}]")
        for i, item in enumerate(preferred_raw)
    )
    return NodeAffinity(required=required, preferred=preferred)


def _validate_pod_anti_affinity(data: Any, path: str) -> PodAntiAffinity:
    if data is None:
        return PodAntiAffinity()
    obj = _require_object(data, path)
    required_raw = _require_list(obj.get("required", []), f"{path}.required")
    required = tuple(
        _validate_requirement(item, f"{path}.required[{i}]")
        for i, item in enumerate(required_raw)
    )
    topology = obj.get("required_topology_key")
    if required and (not isinstance(topology, str) or not topology):
        raise SpecError(
            "hard pod anti-affinity requires required_topology_key",
            f"{path}.required_topology_key",
        )
    if topology is not None and (not isinstance(topology, str) or not topology):
        raise SpecError("illegal topology key", f"{path}.required_topology_key")
    preferred_raw = _require_list(obj.get("preferred", []), f"{path}.preferred")
    preferred = tuple(
        _validate_preferred(item, f"{path}.preferred[{i}]")
        for i, item in enumerate(preferred_raw)
    )
    pref_topology = obj.get("preferred_topology_key")
    if preferred and (not isinstance(pref_topology, str) or not pref_topology):
        raise SpecError(
            "soft pod anti-affinity requires preferred_topology_key",
            f"{path}.preferred_topology_key",
        )
    if pref_topology is not None and (not isinstance(pref_topology, str) or not pref_topology):
        raise SpecError("illegal topology key", f"{path}.preferred_topology_key")
    return PodAntiAffinity(
        required=required,
        required_topology_key=topology,
        preferred=preferred,
        preferred_topology_key=pref_topology,
    )


def load_pods(source: Any, nodes: Optional[Sequence[Node]] = None) -> List[Pod]:
    """Validate and build the pod queue, optionally cross-checking references."""
    data = _read_source(source)
    root = _require_object(data, "$")
    raw_pods = _require_list(root.get("pods"), "$.pods")
    if not raw_pods:
        raise SpecError("at least one pod is required", "$.pods")

    pods: List[Pod] = []
    seen_pods: Dict[str, int] = {}
    for index, raw in enumerate(raw_pods):
        path = f"$.pods[{index}]"
        obj = _require_object(raw, path)
        pod_id = obj.get("pod_id")
        if not isinstance(pod_id, str) or not pod_id:
            raise SpecError(f"illegal pod id {pod_id!r}", f"{path}.pod_id")
        if pod_id in seen_pods:
            raise SpecError(
                f"duplicate pod id {pod_id!r} (first at $.pods[{seen_pods[pod_id]}])",
                f"{path}.pod_id",
            )
        seen_pods[pod_id] = index
        cpu_request = _require_positive_int(obj.get("cpu_request"), f"{path}.cpu_request")
        memory_request = _require_positive_int(
            obj.get("memory_request"), f"{path}.memory_request"
        )
        labels = _validate_labels(obj.get("labels", {}), f"{path}.labels")
        affinity = _validate_node_affinity(obj.get("node_affinity"), f"{path}.node_affinity")
        anti = _validate_pod_anti_affinity(
            obj.get("pod_anti_affinity"), f"{path}.pod_anti_affinity"
        )
        tolerations_raw = _require_list(obj.get("tolerations", []), f"{path}.tolerations")
        tolerations = tuple(
            _validate_toleration(item, f"{path}.tolerations[{i}]")
            for i, item in enumerate(tolerations_raw)
        )
        pods.append(
            Pod(
                pod_id=pod_id,
                cpu_request=cpu_request,
                memory_request=memory_request,
                labels=labels,
                node_affinity=affinity,
                pod_anti_affinity=anti,
                tolerations=tolerations,
            )
        )

    if nodes is not None:
        _validate_references(pods, nodes)
    return pods


def _validate_references(pods: Sequence[Pod], nodes: Sequence[Node]) -> None:
    node_label_keys = set()
    node_label_pairs = set()
    for node in nodes:
        for key, value in node.labels.items():
            node_label_keys.add(key)
            node_label_pairs.add((key, value))
    pod_label_keys = set()
    pod_label_pairs = set()
    for pod in pods:
        for key, value in pod.labels.items():
            pod_label_keys.add(key)
            pod_label_pairs.add((key, value))

    for pod_index, pod in enumerate(pods):
        base = f"$.pods[{pod_index}]"
        affinity = pod.node_affinity
        for req_index, req in enumerate(affinity.required):
            _check_node_ref(req, f"{base}.node_affinity.required[{req_index}]",
                            node_label_keys, node_label_pairs)
        for term_index, term in enumerate(affinity.preferred):
            _check_node_ref(
                term.requirement,
                f"{base}.node_affinity.preferred[{term_index}].requirement",
                node_label_keys,
                node_label_pairs,
            )
        anti = pod.pod_anti_affinity
        topology_paths = (
            ("required", anti.required, anti.required_topology_key),
        )
        for req_index, req in enumerate(anti.required):
            _check_pod_ref(
                req, f"{base}.pod_anti_affinity.required[{req_index}]",
                pod_label_keys, pod_label_pairs,
            )
        if anti.required_topology_key and anti.required_topology_key not in node_label_keys:
            raise SpecError(
                f"topology key {anti.required_topology_key!r} is not a node label",
                f"{base}.pod_anti_affinity.required_topology_key",
            )
        for term_index, term in enumerate(anti.preferred):
            _check_pod_ref(
                term.requirement,
                f"{base}.pod_anti_affinity.preferred[{term_index}].requirement",
                pod_label_keys,
                pod_label_pairs,
            )
        if anti.preferred_topology_key and anti.preferred_topology_key not in node_label_keys:
            raise SpecError(
                f"topology key {anti.preferred_topology_key!r} is not a node label",
                f"{base}.pod_anti_affinity.preferred_topology_key",
            )
        # Keep the tuple referenced so the declared gate order stays self documenting.
        _ = topology_paths


def _check_node_ref(req: LabelRequirement, path: str,
                    keys: set, pairs: set) -> None:
    # Negative operators legitimately reference keys that never exist.
    if req.operator in {"In", "Exists"} and req.key not in keys:
        raise SpecError(f"node label key {req.key!r} does not exist on any node",
                        f"{path}.key")
    if req.operator == "In":
        for value_index, value in enumerate(req.values):
            if (req.key, value) not in pairs:
                raise SpecError(
                    f"node label {req.key}={value!r} does not exist on any node",
                    f"{path}.values[{value_index}]",
                )


def _check_pod_ref(req: LabelRequirement, path: str,
                   keys: set, pairs: set) -> None:
    if req.operator in {"In", "Exists"} and req.key not in keys:
        raise SpecError(f"pod label key {req.key!r} does not exist on any pod",
                        f"{path}.key")
    if req.operator == "In":
        for value_index, value in enumerate(req.values):
            if (req.key, value) not in pairs:
                raise SpecError(
                    f"pod label {req.key}={value!r} does not exist on any pod",
                    f"{path}.values[{value_index}]",
                )


# --------------------------------------------------------------------------- #
# Session parameters
# --------------------------------------------------------------------------- #


def validate_weights(weights: Dict[str, int]) -> Dict[str, int]:
    """Validate basis-point weights; all values are non-negative and sum to 10000."""
    from .constants import PLUGIN_ORDER

    if set(weights) != set(PLUGIN_ORDER):
        raise SpecError(
            f"weights must declare exactly {list(PLUGIN_ORDER)}",
            "$.weights",
        )
    normalized: Dict[str, int] = {}
    for plugin in PLUGIN_ORDER:
        value = weights[plugin]
        if not _is_int(value) or value < 0 or value > WEIGHT_BASIS:
            raise SpecError(
                f"weight must be an integer in [0, {WEIGHT_BASIS}], got {value!r}",
                f"$.weights.{plugin}",
            )
        normalized[plugin] = value
    total = sum(normalized.values())
    if total != WEIGHT_BASIS:
        raise SpecError(
            f"weights must sum to {WEIGHT_BASIS} basis points, got {total}",
            "$.weights",
        )
    return normalized


def validate_scale_factor(scale_factor: Any) -> int:
    if not _is_int(scale_factor) or scale_factor < 1 or scale_factor > 100_000:
        raise SpecError(
            f"scale factor must be a positive integer <= 100000, got {scale_factor!r}",
            "$.scale_factor",
        )
    return int(scale_factor)


def _read_source(source: Any) -> Any:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise SpecError(f"file does not exist: {path}", str(path))
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid JSON: {exc.msg} at line {exc.lineno}", str(path)) from exc
    return source
