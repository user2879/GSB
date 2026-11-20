"""Manifest loading and strict validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_MANIFEST, E_REF
from .model import (
    DEFAULT_PLUGINS,
    PLUGIN_NAMES,
    SELECTOR_OPERATORS,
    TAINT_EFFECTS,
    AntiRule,
    Expression,
    Node,
    PluginConfig,
    Pod,
    Selector,
    SoftRule,
    Taint,
    Toleration,
    valid_label_key,
    valid_label_value,
    valid_node_name,
    valid_pod_id,
)
from .util import read_json_file


def _ptr(token: str | int) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _loc(path: str | Path, *tokens: str | int) -> str:
    pointer = "/".join(_ptr(token) for token in tokens)
    return f"{Path(path).name}#/{pointer}" if pointer else f"{Path(path).name}#/"


def _fail(code: str, message: str, path: str | Path, *tokens: str | int) -> None:
    raise BusinessError(code=code, message=message, location=_loc(path, *tokens))


def _require_object(value: Any, path: str | Path, *tokens: str | int) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(E_MANIFEST, "expected a JSON object", path, *tokens)
    return value


def _require_list(value: Any, path: str | Path, *tokens: str | int) -> list[Any]:
    if not isinstance(value, list):
        _fail(E_MANIFEST, "expected a JSON array", path, *tokens)
    return value


def _require_string(value: Any, path: str | Path, *tokens: str | int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        _fail(E_MANIFEST, "expected a non-empty string", path, *tokens)
    return value


def _require_positive_int(value: Any, path: str | Path, *tokens: str | int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(E_MANIFEST, "expected a positive integer", path, *tokens)
    return value


def _reject_unknown(obj: dict[str, Any], allowed: set[str], path: str | Path, *tokens: str | int) -> None:
    for key in sorted(obj):
        if key not in allowed:
            _fail(E_MANIFEST, f"unknown field '{key}'", path, *tokens, key)


def _load_json(path: str | Path) -> Any:
    try:
        return read_json_file(path)
    except FileNotFoundError:
        raise BusinessError(E_MANIFEST, "manifest file does not exist", str(path)) from None
    except json.JSONDecodeError as exc:
        raise BusinessError(
            E_MANIFEST,
            f"invalid JSON: {exc.msg}",
            f"{Path(path).name}#/",
            {"line": exc.lineno, "column": exc.colno},
        ) from None
    except OSError as exc:
        raise BusinessError(E_MANIFEST, f"cannot read manifest: {exc}", str(path)) from None


def _parse_labels(value: Any, path: str | Path, *tokens: str | int) -> dict[str, str]:
    labels = _require_object(value, path, *tokens)
    result: dict[str, str] = {}
    for key in sorted(labels):
        if not valid_label_key(key):
            _fail(E_MANIFEST, f"invalid label key '{key}'", path, *tokens, key)
        label_value = labels[key]
        if not isinstance(label_value, str):
            _fail(E_MANIFEST, "label value must be a string", path, *tokens, key)
        if not valid_label_value(label_value):
            _fail(E_MANIFEST, f"invalid label value '{label_value}'", path, *tokens, key)
        result[key] = label_value
    return result


def _parse_selector(value: Any, path: str | Path, *tokens: str | int) -> Selector:
    selector = _require_object(value, path, *tokens)
    _reject_unknown(selector, {"matchLabels", "matchExpressions"}, path, *tokens)
    match_labels: dict[str, str] = {}
    if "matchLabels" in selector:
        match_labels = _parse_labels(selector["matchLabels"], path, *tokens, "matchLabels")
    expressions: list[Expression] = []
    if "matchExpressions" in selector:
        raw_expressions = _require_list(selector["matchExpressions"], path, *tokens, "matchExpressions")
        for index, raw in enumerate(raw_expressions):
            expr_obj = _require_object(raw, path, *tokens, "matchExpressions", index)
            _reject_unknown(expr_obj, {"key", "operator", "values"}, path, *tokens, "matchExpressions", index)
            if "key" not in expr_obj:
                _fail(E_MANIFEST, "missing selector key", path, *tokens, "matchExpressions", index, "key")
            key = _require_string(expr_obj["key"], path, *tokens, "matchExpressions", index, "key")
            if not valid_label_key(key):
                _fail(E_MANIFEST, f"invalid label key '{key}'", path, *tokens, "matchExpressions", index, "key")
            operator = _require_string(
                expr_obj.get("operator"), path, *tokens, "matchExpressions", index, "operator"
            )
            if operator not in SELECTOR_OPERATORS:
                _fail(
                    E_MANIFEST,
                    f"selector operator must be one of {sorted(SELECTOR_OPERATORS)}",
                    path,
                    *tokens,
                    "matchExpressions",
                    index,
                    "operator",
                )
            raw_values = expr_obj.get("values", [])
            values_list = _require_list(raw_values, path, *tokens, "matchExpressions", index, "values")
            values: list[str] = []
            for value_index, raw_value in enumerate(values_list):
                if not isinstance(raw_value, str) or not valid_label_value(raw_value):
                    _fail(
                        E_MANIFEST,
                        "selector value must be a valid label value",
                        path,
                        *tokens,
                        "matchExpressions",
                        index,
                        "values",
                        value_index,
                    )
                values.append(raw_value)
            if operator in {"In", "NotIn"}:
                if not values:
                    _fail(
                        E_MANIFEST,
                        f"selector operator {operator} requires at least one value",
                        path,
                        *tokens,
                        "matchExpressions",
                        index,
                        "values",
                    )
                if len(values) != len(set(values)):
                    _fail(
                        E_MANIFEST,
                        "selector values must be unique",
                        path,
                        *tokens,
                        "matchExpressions",
                        index,
                        "values",
                    )
            elif values:
                _fail(
                    E_MANIFEST,
                    f"selector operator {operator} does not accept values",
                    path,
                    *tokens,
                    "matchExpressions",
                    index,
                    "values",
                )
            expressions.append(Expression(key=key, operator=operator, values=tuple(values)))
    return Selector(match_labels=match_labels, match_expressions=tuple(expressions))


def _parse_weight(value: Any, path: str | Path, *tokens: str | int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        _fail(E_MANIFEST, "soft-constraint weight must be an integer from 1 to 100", path, *tokens)
    return value


def _parse_anti_rule(value: Any, path: str | Path, *, soft: bool, tokens: tuple[str | int, ...]) -> AntiRule:
    rule = _require_object(value, path, *tokens)
    _reject_unknown(rule, {"selector", "topologyKey", "weight"}, path, *tokens)
    if "selector" not in rule:
        _fail(E_MANIFEST, "anti-affinity rule requires a selector", path, *tokens, "selector")
    if "topologyKey" not in rule:
        _fail(E_MANIFEST, "anti-affinity rule requires topologyKey", path, *tokens, "topologyKey")
    selector = _parse_selector(rule["selector"], path, *tokens, "selector")
    topology_key = _require_string(rule["topologyKey"], path, *tokens, "topologyKey")
    if not valid_label_key(topology_key):
        _fail(E_MANIFEST, f"invalid topologyKey '{topology_key}'", path, *tokens, "topologyKey")
    weight: int | None = None
    if soft:
        if "weight" in rule:
            weight = _parse_weight(rule["weight"], path, *tokens, "weight")
        else:
            _fail(E_MANIFEST, "soft anti-affinity rule requires weight", path, *tokens, "weight")
    elif "weight" in rule:
        _fail(E_MANIFEST, "hard anti-affinity rule does not accept weight", path, *tokens, "weight")
    return AntiRule(selector=selector, topology_key=topology_key, weight=weight)


def _parse_taunt(value: Any, path: str | Path, node_index: int, index: int) -> Taint:
    taint = _require_object(value, path, "nodes", node_index, "taints", index)
    _reject_unknown(taint, {"key", "value", "effect"}, path, "nodes", node_index, "taints", index)
    if "key" not in taint:
        _fail(E_MANIFEST, "taint requires key", path, "nodes", node_index, "taints", index, "key")
    key = _require_string(taint["key"], path, "nodes", node_index, "taints", index, "key")
    if not valid_label_key(key):
        _fail(E_MANIFEST, f"invalid taint key '{key}'", path, "nodes", node_index, "taints", index, "key")
    raw_value = taint.get("value", "")
    if not isinstance(raw_value, str) or not valid_label_value(raw_value):
        _fail(E_MANIFEST, "invalid taint value", path, "nodes", node_index, "taints", index, "value")
    effect = _require_string(taint.get("effect"), path, "nodes", node_index, "taints", index, "effect")
    if effect not in TAINT_EFFECTS:
        _fail(
            E_MANIFEST,
            f"taint effect must be one of {sorted(TAINT_EFFECTS)}",
            path,
            "nodes",
            node_index,
            "taints",
            index,
            "effect",
        )
    return Taint(key=key, value=raw_value, effect=effect)


def _parse_toleration(value: Any, path: str | Path, pod_index: int, index: int) -> Toleration:
    toleration = _require_object(value, path, "pods", pod_index, "tolerations", index)
    _reject_unknown(toleration, {"key", "operator", "value", "effect"}, path, "pods", pod_index, "tolerations", index)
    key = toleration.get("key", "")
    if not isinstance(key, str) or (key and not valid_label_key(key)):
        _fail(E_MANIFEST, "invalid toleration key", path, "pods", pod_index, "tolerations", index, "key")
    operator = toleration.get("operator", "Equal")
    if operator not in {"Equal", "Exists"}:
        _fail(E_MANIFEST, "toleration operator must be Equal or Exists", path, "pods", pod_index, "tolerations", index, "operator")
    raw_value = toleration.get("value", "")
    if not isinstance(raw_value, str) or not valid_label_value(raw_value):
        _fail(E_MANIFEST, "invalid toleration value", path, "pods", pod_index, "tolerations", index, "value")
    effect = toleration.get("effect")
    if effect is not None and effect not in TAINT_EFFECTS:
        _fail(E_MANIFEST, f"toleration effect must be one of {sorted(TAINT_EFFECTS)}", path, "pods", pod_index, "tolerations", index, "effect")
    if operator == "Equal" and not key:
        _fail(E_MANIFEST, "Equal toleration requires key", path, "pods", pod_index, "tolerations", index, "key")
    if operator == "Exists" and raw_value:
        _fail(E_MANIFEST, "Exists toleration does not accept value", path, "pods", pod_index, "tolerations", index, "value")
    return Toleration(key=key, operator=operator, value=raw_value, effect=effect)


def _parse_node(value: Any, path: str | Path, index: int) -> Node:
    node = _require_object(value, path, "nodes", index)
    _reject_unknown(node, {"name", "labels", "allocatable", "taints", "images"}, path, "nodes", index)
    name = _require_string(node.get("name"), path, "nodes", index, "name")
    if not valid_node_name(name):
        _fail(E_MANIFEST, f"invalid node name '{name}'", path, "nodes", index, "name")
    if "labels" not in node:
        _fail(E_MANIFEST, "node requires labels", path, "nodes", index, "labels")
    labels = _parse_labels(node["labels"], path, "nodes", index, "labels")
    allocatable = _require_object(node.get("allocatable"), path, "nodes", index, "allocatable")
    _reject_unknown(allocatable, {"cpu", "memory"}, path, "nodes", index, "allocatable")
    cpu = _require_positive_int(allocatable.get("cpu"), path, "nodes", index, "allocatable", "cpu")
    memory = _require_positive_int(allocatable.get("memory"), path, "nodes", index, "allocatable", "memory")
    taints: list[Taint] = []
    for taint_index, raw_taint in enumerate(_require_list(node.get("taints", []), path, "nodes", index, "taints")):
        taints.append(_parse_taunt(raw_taint, path, index, taint_index))
    taint_seen: set[tuple[str, str]] = set()
    for taint_index, taint in enumerate(taints):
        marker = (taint.key, taint.effect)
        if marker in taint_seen:
            _fail(E_MANIFEST, "duplicate taint key/effect", path, "nodes", index, "taints", taint_index)
        taint_seen.add(marker)
    images: list[str] = []
    raw_images = _require_list(node.get("images", []), path, "nodes", index, "images")
    for image_index, raw_image in enumerate(raw_images):
        image = _require_string(raw_image, path, "nodes", index, "images", image_index)
        if image in images:
            _fail(E_MANIFEST, "duplicate node image", path, "nodes", index, "images", image_index)
        images.append(image)
    return Node(name=name, labels=labels, cpu=cpu, memory=memory, taints=tuple(taints), images=tuple(images))


def _parse_pod(value: Any, path: str | Path, index: int) -> Pod:
    pod = _require_object(value, path, "pods", index)
    _reject_unknown(
        pod,
        {
            "id",
            "labels",
            "resources",
            "image",
            "nodeAffinity",
            "antiAffinity",
            "tolerations",
        },
        path,
        "pods",
        index,
    )
    pod_id = _require_string(pod.get("id"), path, "pods", index, "id")
    if not valid_pod_id(pod_id):
        _fail(E_MANIFEST, f"invalid pod id '{pod_id}'", path, "pods", index, "id")
    labels = _parse_labels(pod.get("labels", {}), path, "pods", index, "labels")
    resources = _require_object(pod.get("resources"), path, "pods", index, "resources")
    _reject_unknown(resources, {"cpu", "memory"}, path, "pods", index, "resources")
    cpu = _require_positive_int(resources.get("cpu"), path, "pods", index, "resources", "cpu")
    memory = _require_positive_int(resources.get("memory"), path, "pods", index, "resources", "memory")
    image: str | None = None
    if "image" in pod:
        image = _require_string(pod["image"], path, "pods", index, "image")

    node_hard: Selector | None = None
    node_soft: list[SoftRule] = []
    if "nodeAffinity" in pod:
        affinity = _require_object(pod["nodeAffinity"], path, "pods", index, "nodeAffinity")
        _reject_unknown(affinity, {"hard", "soft"}, path, "pods", index, "nodeAffinity")
        if "hard" in affinity:
            node_hard = _parse_selector(affinity["hard"], path, "pods", index, "nodeAffinity", "hard")
        for soft_index, raw_rule in enumerate(
            _require_list(affinity.get("soft", []), path, "pods", index, "nodeAffinity", "soft")
        ):
            rule = _require_object(raw_rule, path, "pods", index, "nodeAffinity", "soft", soft_index)
            _reject_unknown(rule, {"selector", "weight"}, path, "pods", index, "nodeAffinity", "soft", soft_index)
            if "selector" not in rule:
                _fail(E_MANIFEST, "soft node-affinity rule requires selector", path, "pods", index, "nodeAffinity", "soft", soft_index)
            if "weight" not in rule:
                _fail(E_MANIFEST, "soft node-affinity rule requires weight", path, "pods", index, "nodeAffinity", "soft", soft_index)
            selector = _parse_selector(
                rule["selector"], path, "pods", index, "nodeAffinity", "soft", soft_index, "selector"
            )
            weight = _parse_weight(rule["weight"], path, "pods", index, "nodeAffinity", "soft", soft_index, "weight")
            node_soft.append(SoftRule(selector=selector, weight=weight))

    anti_hard: list[AntiRule] = []
    anti_soft: list[AntiRule] = []
    if "antiAffinity" in pod:
        anti = _require_object(pod["antiAffinity"], path, "pods", index, "antiAffinity")
        _reject_unknown(anti, {"hard", "soft"}, path, "pods", index, "antiAffinity")
        for hard_index, raw_rule in enumerate(
            _require_list(anti.get("hard", []), path, "pods", index, "antiAffinity", "hard")
        ):
            anti_hard.append(
                _parse_anti_rule(
                    raw_rule,
                    path,
                    soft=False,
                    tokens=("pods", index, "antiAffinity", "hard", hard_index),
                )
            )
        for soft_index, raw_rule in enumerate(
            _require_list(anti.get("soft", []), path, "pods", index, "antiAffinity", "soft")
        ):
            anti_soft.append(
                _parse_anti_rule(
                    raw_rule,
                    path,
                    soft=True,
                    tokens=("pods", index, "antiAffinity", "soft", soft_index),
                )
            )

    tolerations: list[Toleration] = []
    for toleration_index, raw_toleration in enumerate(
        _require_list(pod.get("tolerations", []), path, "pods", index, "tolerations")
    ):
        tolerations.append(_parse_toleration(raw_toleration, path, index, toleration_index))
    return Pod(
        pod_id=pod_id,
        labels=labels,
        cpu=cpu,
        memory=memory,
        image=image,
        node_affinity_hard=node_hard,
        node_affinity_soft=tuple(node_soft),
        anti_affinity_hard=tuple(anti_hard),
        anti_affinity_soft=tuple(anti_soft),
        tolerations=tuple(tolerations),
    )


def _selector_reference_locations(pod: Pod, path: str | Path, pod_index: int) -> list[tuple[str, str | Path, tuple[str | int, ...]]]:
    refs: list[tuple[str, str | Path, tuple[str | int, ...]]] = []
    if pod.node_affinity_hard is not None:
        for key in sorted(pod.node_affinity_hard.referenced_keys()):
            refs.append((key, path, ("pods", pod_index, "nodeAffinity", "hard")))
    for rule_index, rule in enumerate(pod.node_affinity_soft):
        for key in sorted(rule.selector.referenced_keys()):
            refs.append((key, path, ("pods", pod_index, "nodeAffinity", "soft", rule_index, "selector")))
    for rule_index, rule in enumerate(pod.anti_affinity_hard):
        refs.append((rule.topology_key, path, ("pods", pod_index, "antiAffinity", "hard", rule_index, "topologyKey")))
    for rule_index, rule in enumerate(pod.anti_affinity_soft):
        refs.append((rule.topology_key, path, ("pods", pod_index, "antiAffinity", "soft", rule_index, "topologyKey")))
    return refs


def _anti_reference_locations(pod: Pod, path: str | Path, pod_index: int) -> list[tuple[str, str | Path, tuple[str | int, ...]]]:
    refs: list[tuple[str, str | Path, tuple[str | int, ...]]] = []
    for rule_index, rule in enumerate(pod.anti_affinity_hard):
        for key in sorted(rule.selector.referenced_keys()):
            refs.append((key, path, ("pods", pod_index, "antiAffinity", "hard", rule_index, "selector")))
    for rule_index, rule in enumerate(pod.anti_affinity_soft):
        for key in sorted(rule.selector.referenced_keys()):
            refs.append((key, path, ("pods", pod_index, "antiAffinity", "soft", rule_index, "selector")))
    return refs


def _parse_plugins(value: Any, path: str | Path) -> PluginConfig:
    plugins = _require_object(value, path)
    _reject_unknown(plugins, {"schemaVersion", "version", "weights"}, path)
    if plugins.get("schemaVersion") != 1:
        _fail(E_MANIFEST, "plugin schemaVersion must be 1", path, "schemaVersion")
    version = _require_positive_int(plugins.get("version"), path, "version")
    weights_obj = _require_object(plugins.get("weights"), path, "weights")
    _reject_unknown(weights_obj, set(PLUGIN_NAMES), path, "weights")
    weights: dict[str, int] = {}
    for name in PLUGIN_NAMES:
        if name not in weights_obj:
            _fail(E_MANIFEST, f"missing plugin weight '{name}'", path, "weights", name)
        raw_weight = weights_obj[name]
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, int) or raw_weight < 0:
            _fail(E_MANIFEST, "plugin weight must be a non-negative integer", path, "weights", name)
        weights[name] = raw_weight
    if sum(weights.values()) <= 0:
        _fail(E_MANIFEST, "at least one plugin weight must be positive", path, "weights")
    return PluginConfig(weights=weights, version=version)


def load_manifests(
    nodes_path: str | Path,
    pods_path: str | Path,
    plugins_path: str | Path | None = None,
) -> tuple[list[Node], list[Pod], PluginConfig]:
    """Load and validate all manifests atomically.

    Structural/type/range problems use E_MANIFEST. A syntactically valid
    selector or topology key that never occurs in the node label universe uses
    E_REF. No partial object is returned when validation fails.
    """
    nodes_doc = _require_object(_load_json(nodes_path), nodes_path)
    _reject_unknown(nodes_doc, {"schemaVersion", "nodes"}, nodes_path)
    if nodes_doc.get("schemaVersion") != 1:
        _fail(E_MANIFEST, "nodes schemaVersion must be 1", nodes_path, "schemaVersion")
    raw_nodes = _require_list(nodes_doc.get("nodes"), nodes_path, "nodes")
    if not raw_nodes:
        _fail(E_MANIFEST, "node list must not be empty", nodes_path, "nodes")

    nodes: list[Node] = []
    node_names: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        node = _parse_node(raw_node, nodes_path, index)
        if node.name in node_names:
            _fail(E_MANIFEST, f"duplicate node name '{node.name}'", nodes_path, "nodes", index, "name")
        node_names.add(node.name)
        nodes.append(node)
    max_cpu = max(node.cpu for node in nodes)
    max_memory = max(node.memory for node in nodes)
    node_label_keys = {key for node in nodes for key in node.labels}

    pods_doc = _require_object(_load_json(pods_path), pods_path)
    _reject_unknown(pods_doc, {"schemaVersion", "pods"}, pods_path)
    if pods_doc.get("schemaVersion") != 1:
        _fail(E_MANIFEST, "pods schemaVersion must be 1", pods_path, "schemaVersion")
    raw_pods = _require_list(pods_doc.get("pods"), pods_path, "pods")
    if not raw_pods:
        _fail(E_MANIFEST, "pod queue must not be empty", pods_path, "pods")

    pods: list[Pod] = []
    pod_ids: set[str] = set()
    for index, raw_pod in enumerate(raw_pods):
        pod = _parse_pod(raw_pod, pods_path, index)
        if pod.pod_id in pod_ids:
            _fail(E_MANIFEST, f"duplicate pod id '{pod.pod_id}'", pods_path, "pods", index, "id")
        pod_ids.add(pod.pod_id)
        if pod.cpu > max_cpu:
            _fail(
                E_MANIFEST,
                f"pod CPU request {pod.cpu} exceeds maximum single-node allocatable CPU {max_cpu}",
                pods_path,
                "pods",
                index,
                "resources",
                "cpu",
            )
        if pod.memory > max_memory:
            _fail(
                E_MANIFEST,
                f"pod memory request {pod.memory} exceeds maximum single-node allocatable memory {max_memory}",
                pods_path,
                "pods",
                index,
                "resources",
                "memory",
            )
        pods.append(pod)

    pod_label_keys = {key for pod in pods for key in pod.labels}
    for pod_index, pod in enumerate(pods):
        for key, path, tokens in _selector_reference_locations(pod, pods_path, pod_index):
            if key not in node_label_keys:
                raise BusinessError(
                    E_REF,
                    f"label key '{key}' is referenced by a pod but does not occur in any node",
                    _loc(path, *tokens),
                    {"key": key, "pod": pod.pod_id},
                )
        for key, path, tokens in _anti_reference_locations(pod, pods_path, pod_index):
            if key not in pod_label_keys:
                raise BusinessError(
                    E_REF,
                    f"label key '{key}' is used by anti-affinity but does not occur in any pod labels",
                    _loc(path, *tokens),
                    {"key": key, "pod": pod.pod_id},
                )

    plugins = DEFAULT_PLUGINS
    if plugins_path is not None:
        plugins = _parse_plugins(_load_json(plugins_path), plugins_path)
    return nodes, pods, plugins.normalized()
