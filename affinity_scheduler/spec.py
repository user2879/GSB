from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import SpecError
from .model import (
    PLUGIN_NAMES,
    TAINT_EFFECTS,
    LabelRequirement,
    Node,
    Pod,
    PodSelectorRequirement,
    ServiceConfig,
    ServiceGroup,
    Spec,
    Taint,
    Toleration,
)


SUPPORTED_LABEL_OPERATORS = frozenset({"In", "NotIn", "Exists", "DoesNotExist"})
SUPPORTED_TOLERATION_OPERATORS = frozenset({"Equal", "Exists"})


class _JsonPositions:
    """Small JSON tokenizer that records the source line of every path."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.path_lines: dict[tuple[Any, ...], int] = {}
        self._scan_value(())

    def line_of_path(self, path: str) -> int:
        tokens = _split_path(path)
        for end in range(len(tokens), 0, -1):
            key = tuple(tokens[:end])
            line = self.path_lines.get(key)
            if line is not None:
                return line
        return 1

    def _line(self, position: int | None = None) -> int:
        if position is None:
            position = self.index
        return self.text.count("\n", 0, position) + 1

    def _skip_ws(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _scan_string(self) -> tuple[str, int]:
        start = self.index
        if self.text[self.index] != '"':
            raise ValueError("expected string")
        self.index += 1
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return json.loads(self.text[start : self.index]), start
        raise ValueError("unterminated string")

    def _scan_value(self, path: tuple[Any, ...]) -> None:
        self._skip_ws()
        self.path_lines.setdefault(path, self._line())
        if self.index >= len(self.text):
            return
        char = self.text[self.index]
        if char == "{":
            self._scan_object(path)
        elif char == "[":
            self._scan_array(path)
        elif char == '"':
            self.index += 1
            escaped = False
            while self.index < len(self.text):
                current = self.text[self.index]
                self.index += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
        else:
            while (
                self.index < len(self.text)
                and self.text[self.index] not in ",]} \t\r\n"
            ):
                self.index += 1

    def _scan_object(self, path: tuple[Any, ...]) -> None:
        self.index += 1
        self._skip_ws()
        if self.index < len(self.text) and self.text[self.index] == "}":
            self.index += 1
            return
        while self.index < len(self.text):
            self._skip_ws()
            key, key_position = self._scan_string()
            child = (*path, key)
            self.path_lines.setdefault(child, self._line(key_position))
            self._skip_ws()
            if self.text[self.index] != ":":
                raise ValueError("expected colon")
            self.index += 1
            self._scan_value(child)
            self._skip_ws()
            if self.text[self.index] == ",":
                self.index += 1
                continue
            if self.text[self.index] == "}":
                self.index += 1
                return
            raise ValueError("expected comma or closing brace")

    def _scan_array(self, path: tuple[Any, ...]) -> None:
        self.index += 1
        self._skip_ws()
        if self.index < len(self.text) and self.text[self.index] == "]":
            self.index += 1
            return
        item_index = 0
        while self.index < len(self.text):
            self._skip_ws()
            child = (*path, item_index)
            self.path_lines.setdefault(child, self._line())
            self._scan_value(child)
            item_index += 1
            self._skip_ws()
            if self.text[self.index] == ",":
                self.index += 1
                continue
            if self.text[self.index] == "]":
                self.index += 1
                return
            raise ValueError("expected comma or closing bracket")


def _split_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for chunk in path.split("."):
        if "[" not in chunk:
            tokens.append(chunk)
            continue
        name, rest = chunk.split("[", 1)
        tokens.append(name)
        for part in rest.split("["):
            tokens.append(int(part.rstrip("]")))
    return tokens


class _Validator:
    def __init__(self, data: Any, text: str, source_path: str) -> None:
        self.data = data
        self.source_path = source_path
        self.positions = _JsonPositions(text)

    def fail(self, path: str, message: str) -> None:
        raise SpecError(
            message,
            field_path=path,
            source_line=self.positions.line_of_path(path),
            source_path=self.source_path,
        )

    def obj(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.fail(path, "must be a JSON object")
        return value

    def array(self, value: Any, path: str) -> list[Any]:
        if not isinstance(value, list):
            self.fail(path, "must be a JSON array")
        return value

    def string(self, value: Any, path: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str) or (not allow_empty and not value):
            self.fail(path, "must be a non-empty string")
        return value

    def integer(
        self,
        value: Any,
        path: str,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        allowed: frozenset[int] | set[int] | None = None,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.fail(path, "must be an integer")
        if minimum is not None and value < minimum:
            self.fail(path, f"must be >= {minimum}")
        if maximum is not None and value > maximum:
            self.fail(path, f"must be <= {maximum}")
        if allowed is not None and value not in allowed:
            self.fail(path, f"must be one of {sorted(allowed)}")
        return value

    def string_map(self, value: Any, path: str) -> dict[str, str]:
        mapping = self.obj(value, path)
        result: dict[str, str] = {}
        for key, item in mapping.items():
            self.string(key, f"{path}.{key}")
            result[key] = self.string(item, f"{path}.{key}")
        return result

    def validate(self) -> Spec:
        root = self.obj(self.data, "$")
        parameter_version = self.integer(
            root.get("parameter_version"), "parameter_version", minimum=1
        )
        scale_exponent = self.integer(
            root.get("scale_exponent"), "scale_exponent", minimum=0, maximum=3
        )

        plugins = self.obj(root.get("plugins"), "plugins")
        weights: dict[str, int] = {}
        for plugin in PLUGIN_NAMES:
            plugin_path = f"plugins.{plugin}"
            declaration = self.obj(plugins.get(plugin), plugin_path)
            weights[plugin] = self.integer(
                declaration.get("weight_myriad"),
                f"{plugin_path}.weight_myriad",
                minimum=0,
                maximum=10000,
            )
        for extra in sorted(set(plugins) - set(PLUGIN_NAMES)):
            self.fail(f"plugins.{extra}", "unknown scoring plugin")
        if sum(weights.values()) != 10000:
            self.fail(
                "plugins",
                f"weight_myriad sum must equal 10000, got {sum(weights.values())}",
            )

        raw_nodes = self.array(root.get("nodes"), "nodes")
        if not raw_nodes:
            self.fail("nodes", "must contain at least one node")
        nodes: list[Node] = []
        node_names: set[str] = set()
        for node_index, raw_node in enumerate(raw_nodes):
            base = f"nodes[{node_index}]"
            node_obj = self.obj(raw_node, base)
            name = self.string(node_obj.get("name"), f"{base}.name")
            if name in node_names:
                self.fail(f"{base}.name", f"duplicate node name {name!r}")
            node_names.add(name)
            labels = self.string_map(node_obj.get("labels", {}), f"{base}.labels")
            allocatable = self.obj(
                node_obj.get("allocatable"), f"{base}.allocatable"
            )
            cpu = self.integer(
                allocatable.get("cpu"), f"{base}.allocatable.cpu", minimum=1
            )
            memory = self.integer(
                allocatable.get("memory"),
                f"{base}.allocatable.memory",
                minimum=1,
            )
            raw_taints = self.array(node_obj.get("taints", []), f"{base}.taints")
            taints: list[Taint] = []
            for taint_index, raw_taint in enumerate(raw_taints):
                taint_path = f"{base}.taints[{taint_index}]"
                taint_obj = self.obj(raw_taint, taint_path)
                key = self.string(taint_obj.get("key"), f"{taint_path}.key")
                value = self.string(
                    taint_obj.get("value", ""),
                    f"{taint_path}.value",
                    allow_empty=True,
                )
                effect = self.integer(
                    taint_obj.get("effect"),
                    f"{taint_path}.effect",
                    allowed=frozenset(TAINT_EFFECTS),
                )
                taints.append(Taint(key=key, value=value, effect=effect))
            nodes.append(
                Node(
                    name=name,
                    labels=labels,
                    cpu=cpu,
                    memory=memory,
                    taints=tuple(taints),
                )
            )

        raw_pods = self.array(root.get("pods"), "pods")
        if not raw_pods:
            self.fail("pods", "must contain at least one pod")
        pods: list[Pod] = []
        pod_ids: set[str] = set()
        for pod_index, raw_pod in enumerate(raw_pods):
            base = f"pods[{pod_index}]"
            pod_obj = self.obj(raw_pod, base)
            pod_id = self.string(pod_obj.get("id"), f"{base}.id")
            if pod_id in pod_ids:
                self.fail(f"{base}.id", f"duplicate pod id {pod_id!r}")
            pod_ids.add(pod_id)
            labels = self.string_map(pod_obj.get("labels", {}), f"{base}.labels")
            requests = self.obj(pod_obj.get("requests"), f"{base}.requests")
            request_cpu = self.integer(
                requests.get("cpu"), f"{base}.requests.cpu", minimum=1
            )
            request_memory = self.integer(
                requests.get("memory"), f"{base}.requests.memory", minimum=1
            )
            node_affinity = self.obj(
                pod_obj.get("node_affinity", {}), f"{base}.node_affinity"
            )
            node_hard = self._label_requirements(
                node_affinity.get("hard", []), f"{base}.node_affinity.hard"
            )
            node_soft = self._label_requirements(
                node_affinity.get("soft", []), f"{base}.node_affinity.soft"
            )
            anti_affinity = self.obj(
                pod_obj.get("pod_anti_affinity", {}),
                f"{base}.pod_anti_affinity",
            )
            anti_hard = self._pod_selector_requirements(
                anti_affinity.get("hard", []),
                f"{base}.pod_anti_affinity.hard",
            )
            anti_soft = self._pod_selector_requirements(
                anti_affinity.get("soft", []),
                f"{base}.pod_anti_affinity.soft",
            )
            raw_tolerations = self.array(
                pod_obj.get("tolerations", []), f"{base}.tolerations"
            )
            tolerations: list[Toleration] = []
            for tol_index, raw_toleration in enumerate(raw_tolerations):
                tol_path = f"{base}.tolerations[{tol_index}]"
                tol_obj = self.obj(raw_toleration, tol_path)
                key = self.string(
                    tol_obj.get("key"), f"{tol_path}.key", allow_empty=True
                )
                operator = self.string(
                    tol_obj.get("operator", "Equal"), f"{tol_path}.operator"
                )
                if operator not in SUPPORTED_TOLERATION_OPERATORS:
                    self.fail(
                        f"{tol_path}.operator",
                        f"unsupported operator {operator!r}",
                    )
                value = self.string(
                    tol_obj.get("value", ""),
                    f"{tol_path}.value",
                    allow_empty=True,
                )
                effect = self.integer(
                    tol_obj.get("effect", -1),
                    f"{tol_path}.effect",
                    allowed=frozenset({-1, *TAINT_EFFECTS}),
                )
                tolerations.append(
                    Toleration(
                        key=key, operator=operator, value=value, effect=effect
                    )
                )
            pods.append(
                Pod(
                    pod_id=pod_id,
                    labels=labels,
                    request_cpu=request_cpu,
                    request_memory=request_memory,
                    node_hard=tuple(node_hard),
                    node_soft=tuple(node_soft),
                    anti_hard=tuple(anti_hard),
                    anti_soft=tuple(anti_soft),
                    tolerations=tuple(tolerations),
                )
            )

        service = self._service(root.get("service", {}), node_names)
        return Spec(
            source_path=self.source_path,
            parameter_version=parameter_version,
            scale_exponent=scale_exponent,
            weights_myriad=weights,
            nodes=tuple(nodes),
            pods=tuple(pods),
            service=service,
        )

    def _label_requirements(self, value: Any, path: str) -> list[LabelRequirement]:
        requirements: list[LabelRequirement] = []
        for index, raw_requirement in enumerate(self.array(value, path)):
            item_path = f"{path}[{index}]"
            item = self.obj(raw_requirement, item_path)
            key = self.string(item.get("key"), f"{item_path}.key")
            operator = self.string(
                item.get("operator", "In"), f"{item_path}.operator"
            )
            if operator not in SUPPORTED_LABEL_OPERATORS:
                self.fail(
                    f"{item_path}.operator",
                    f"unsupported operator {operator!r}",
                )
            values: list[str] = []
            if operator in {"In", "NotIn"}:
                raw_values = self.array(item.get("values"), f"{item_path}.values")
                if not raw_values:
                    self.fail(f"{item_path}.values", "must not be empty")
                values = [
                    self.string(entry, f"{item_path}.values[{entry_index}]")
                    for entry_index, entry in enumerate(raw_values)
                ]
            weight = self.integer(
                item.get("weight", 1),
                f"{item_path}.weight",
                minimum=1,
                maximum=100,
            )
            requirements.append(
                LabelRequirement(
                    key=key,
                    operator=operator,
                    values=tuple(values),
                    weight=weight,
                )
            )
        return requirements

    def _pod_selector_requirements(
        self, value: Any, path: str
    ) -> list[PodSelectorRequirement]:
        requirements: list[PodSelectorRequirement] = []
        for index, raw_requirement in enumerate(self.array(value, path)):
            item_path = f"{path}[{index}]"
            item = self.obj(raw_requirement, item_path)
            key = self.string(
                item.get("label_key", item.get("key")), f"{item_path}.label_key"
            )
            operator = self.string(
                item.get("operator", "In"), f"{item_path}.operator"
            )
            if operator not in SUPPORTED_LABEL_OPERATORS:
                self.fail(
                    f"{item_path}.operator",
                    f"unsupported operator {operator!r}",
                )
            values: list[str] = []
            if operator in {"In", "NotIn"}:
                raw_values = self.array(item.get("values"), f"{item_path}.values")
                if not raw_values:
                    self.fail(f"{item_path}.values", "must not be empty")
                values = [
                    self.string(entry, f"{item_path}.values[{entry_index}]")
                    for entry_index, entry in enumerate(raw_values)
                ]
            topology_key = self.string(
                item.get("topology_key"), f"{item_path}.topology_key"
            )
            weight = self.integer(
                item.get("weight", 1),
                f"{item_path}.weight",
                minimum=1,
                maximum=100,
            )
            requirements.append(
                PodSelectorRequirement(
                    key=key,
                    operator=operator,
                    values=tuple(values),
                    topology_key=topology_key,
                    weight=weight,
                )
            )
        return requirements

    def _service(self, value: Any, node_names: set[str]) -> ServiceConfig:
        config = self.obj(value, "service")
        host = self.string(config.get("host", "127.0.0.1"), "service.host")
        if host != "127.0.0.1":
            self.fail("service.host", "must be 127.0.0.1 for loopback-only scoring")
        port_base = self.integer(
            config.get("port_base", 39000),
            "service.port_base",
            minimum=1024,
            maximum=65000,
        )
        timeout_ms = self.integer(
            config.get("timeout_ms", 2000),
            "service.timeout_ms",
            minimum=50,
            maximum=60000,
        )
        raw_groups = self.array(config.get("groups", []), "service.groups")
        groups: list[ServiceGroup] = []
        if not raw_groups:
            groups.append(
                ServiceGroup(
                    name="default",
                    node_names=tuple(sorted(node_names)),
                    port=port_base,
                )
            )
        else:
            group_names: set[str] = set()
            assigned: set[str] = set()
            for index, raw_group in enumerate(raw_groups):
                base = f"service.groups[{index}]"
                group_obj = self.obj(raw_group, base)
                name = self.string(group_obj.get("name"), f"{base}.name")
                if name in group_names:
                    self.fail(f"{base}.name", f"duplicate group name {name!r}")
                group_names.add(name)
                raw_nodes = self.array(group_obj.get("nodes"), f"{base}.nodes")
                if not raw_nodes:
                    self.fail(f"{base}.nodes", "must not be empty")
                names: list[str] = []
                for node_index, raw_name in enumerate(raw_nodes):
                    node_name = self.string(
                        raw_name, f"{base}.nodes[{node_index}]"
                    )
                    if node_name not in node_names:
                        self.fail(
                            f"{base}.nodes[{node_index}]",
                            f"unknown node {node_name!r}",
                        )
                    if node_name in assigned:
                        self.fail(
                            f"{base}.nodes[{node_index}]",
                            f"node {node_name!r} appears in multiple groups",
                        )
                    assigned.add(node_name)
                    names.append(node_name)
                groups.append(
                    ServiceGroup(
                        name=name,
                        node_names=tuple(names),
                        port=port_base + index,
                    )
                )
            missing = sorted(node_names - assigned)
            if missing:
                self.fail(
                    "service.groups",
                    f"nodes missing from scoring groups: {', '.join(missing)}",
                )
        if port_base + len(groups) - 1 > 65535:
            self.fail("service.port_base", "port range exceeds 65535")
        return ServiceConfig(
            host=host,
            port_base=port_base,
            timeout_ms=timeout_ms,
            groups=tuple(groups),
        )


def load_spec(path: str | Path) -> Spec:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            field_path="$",
            source_line=exc.lineno,
            source_path=str(source),
        ) from exc
    return _Validator(data, text, str(source)).validate()
