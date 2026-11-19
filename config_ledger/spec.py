"""Strict loading and canonicalisation of local JSON declarations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .errors import ErrorIssue, SpecValidationError


LOOPBACK_HOST = "127.0.0.1"
ARBITRATION_RULES = ("majority_then_latest",)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_value_type(value: Any) -> bool:
    return _is_int(value) or isinstance(value, str)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class NodeSpec:
    id: str
    host: str
    port: int
    quota: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "quota": self.quota,
        }


@dataclass(frozen=True)
class KeySpec:
    name: str
    expected: int | str
    quorum_bp: int

    def as_dict(self) -> dict[str, Any]:
        return {"expected": self.expected, "quorum_bp": self.quorum_bp}


@dataclass(frozen=True)
class Spec:
    name: str
    parameter_version: int
    global_concurrency: int
    observation_timeout_ms: int
    ui_latency_budget_ms: int
    observation_window: str
    arbitration_rule: str
    keys: dict[str, KeySpec]
    nodes: tuple[NodeSpec, ...]

    @property
    def sorted_node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(node.id for node in self.nodes))

    @property
    def sorted_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.keys))

    @property
    def nodes_by_id(self) -> dict[str, NodeSpec]:
        return {node.id: node for node in sorted(self.nodes, key=lambda item: item.id)}

    @property
    def key_names(self) -> tuple[str, ...]:
        return self.sorted_keys

    def parameter_payload(self) -> dict[str, Any]:
        """Semantic inputs whose change requires a monotonically new version."""

        return {
            "global_concurrency": self.global_concurrency,
            "observation_timeout_ms": self.observation_timeout_ms,
            "ui_latency_budget_ms": self.ui_latency_budget_ms,
            "observation_window": self.observation_window,
            "arbitration_rule": self.arbitration_rule,
            "keys": {
                key: self.keys[key].as_dict() for key in sorted(self.keys)
            },
            "nodes": [node.as_dict() for node in sorted(self.nodes, key=lambda n: n.id)],
        }

    @property
    def fingerprint(self) -> str:
        payload = _canonical_json(self.parameter_payload()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def with_parameter_version(self, version: int) -> "Spec":
        return replace(self, parameter_version=version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_version": self.parameter_version,
            "global_concurrency": self.global_concurrency,
            "observation_timeout_ms": self.observation_timeout_ms,
            "ui_latency_budget_ms": self.ui_latency_budget_ms,
            "observation_window": self.observation_window,
            "arbitration_rule": self.arbitration_rule,
            "keys": {key: self.keys[key].as_dict() for key in self.sorted_keys},
            "nodes": [node.as_dict() for node in sorted(self.nodes, key=lambda n: n.id)],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


_TOP_LEVEL_FIELDS = {
    "name",
    "parameter_version",
    "global_concurrency",
    "observation_timeout_ms",
    "ui_latency_budget_ms",
    "observation_window",
    "arbitration_rule",
    "keys",
    "nodes",
}
_NODE_FIELDS = {"id", "host", "port", "quota"}
_KEY_FIELDS = {"expected", "quorum_bp"}


def _issue(node_id: str, field_path: str, message: str) -> ErrorIssue:
    return ErrorIssue(node_id=node_id, field_path=field_path, message=message)


def _validate_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    base_path: str,
    node_id: str,
    issues: list[ErrorIssue],
) -> None:
    for field in sorted(set(value) - allowed):
        issues.append(
            _issue(node_id, f"{base_path}.{field}" if base_path else field, "unknown field")
        )


def parse_spec_data(data: Any, *, source: str = "<memory>") -> Spec:
    """Validate a decoded declaration without mutating any caller state.

    All schema issues are collected. A ``Spec`` is returned only when every
    issue list is empty; callers therefore never observe a half-loaded spec.
    """

    issues: list[ErrorIssue] = []
    if not isinstance(data, dict):
        raise SpecValidationError(
            [_issue("<file>", "$", f"{source}: top-level JSON value must be an object")]
        )

    _validate_unknown_fields(data, _TOP_LEVEL_FIELDS, "", "<global>", issues)

    name = data.get("name", "")
    if not isinstance(name, str) or not name.strip():
        issues.append(_issue("<global>", "name", "must be a non-empty string"))

    parameter_version = data.get("parameter_version", 1)
    if not _is_int(parameter_version) or parameter_version < 1:
        issues.append(_issue("<global>", "parameter_version", "must be an integer >= 1"))
        parameter_version = 1

    global_concurrency = data.get("global_concurrency")
    if not _is_int(global_concurrency) or global_concurrency < 1:
        issues.append(
            _issue("<global>", "global_concurrency", "must be an integer >= 1")
        )
        global_concurrency = 1

    timeout_ms = data.get("observation_timeout_ms")
    if not _is_int(timeout_ms) or timeout_ms < 1 or timeout_ms > 600000:
        issues.append(
            _issue(
                "<global>",
                "observation_timeout_ms",
                "must be an integer between 1 and 600000",
            )
        )
        timeout_ms = 1

    latency_budget_ms = data.get("ui_latency_budget_ms")
    if not _is_int(latency_budget_ms) or latency_budget_ms < 0:
        issues.append(
            _issue(
                "<global>",
                "ui_latency_budget_ms",
                "must be a non-negative integer",
            )
        )
        latency_budget_ms = 0

    observation_window = data.get("observation_window")
    if not isinstance(observation_window, str) or not observation_window.strip():
        issues.append(
            _issue(
                "<global>",
                "observation_window",
                "must be a non-empty string",
            )
        )
        observation_window = "invalid"

    arbitration_rule = data.get("arbitration_rule", "majority_then_latest")
    if arbitration_rule not in ARBITRATION_RULES:
        issues.append(
            _issue(
                "<global>",
                "arbitration_rule",
                "must be one of: " + ", ".join(ARBITRATION_RULES),
            )
        )
        arbitration_rule = ARBITRATION_RULES[0]

    keys_raw = data.get("keys")
    keys: dict[str, KeySpec] = {}
    if not isinstance(keys_raw, dict) or not keys_raw:
        issues.append(_issue("<global>", "keys", "must be a non-empty object"))
    else:
        for key_name in sorted(keys_raw):
            key_path = f'keys[{json.dumps(key_name, ensure_ascii=False)}]'
            raw_key = keys_raw[key_name]
            if not isinstance(key_name, str) or not key_name:
                issues.append(_issue("<global>", key_path, "key name must be non-empty"))
                continue
            if not isinstance(raw_key, dict):
                issues.append(_issue("<global>", key_path, "must be an object"))
                continue
            _validate_unknown_fields(raw_key, _KEY_FIELDS, key_path, "<keys>", issues)
            if "expected" not in raw_key:
                issues.append(
                    _issue("<keys>", f"{key_path}.expected", "required field is missing")
                )
                expected: int | str = ""
            else:
                expected = raw_key["expected"]
                if not _is_value_type(expected):
                    issues.append(
                        _issue(
                            "<keys>",
                            f"{key_path}.expected",
                            "must be an integer (not boolean) or a string",
                        )
                    )
            quorum_bp = raw_key.get("quorum_bp")
            if not _is_int(quorum_bp) or quorum_bp < 0 or quorum_bp > 10000:
                issues.append(
                    _issue(
                        "<keys>",
                        f"{key_path}.quorum_bp",
                        "must be an integer between 0 and 10000",
                    )
                )
                quorum_bp = 0
            if _is_value_type(expected):
                keys[key_name] = KeySpec(
                    name=key_name, expected=expected, quorum_bp=quorum_bp
                )

    nodes_raw = data.get("nodes")
    nodes: list[NodeSpec] = []
    if not isinstance(nodes_raw, list) or not nodes_raw:
        issues.append(_issue("<global>", "nodes", "must be a non-empty array"))
    else:
        seen_ids: set[str] = set()
        seen_ports: dict[int, int] = {}
        for index, raw_node in enumerate(nodes_raw):
            path = f"nodes[{index}]"
            if not isinstance(raw_node, dict):
                issues.append(_issue(f"<nodes[{index}]>", path, "must be an object"))
                continue
            raw_id = raw_node.get("id")
            node_id = raw_id if isinstance(raw_id, str) and raw_id else f"<nodes[{index}]>"
            _validate_unknown_fields(raw_node, _NODE_FIELDS, path, node_id, issues)
            node_ok = True

            if not isinstance(raw_id, str) or not raw_id.strip():
                issues.append(_issue(node_id, f"{path}.id", "must be a non-empty string"))
                node_ok = False
            elif raw_id in seen_ids:
                issues.append(_issue(node_id, f"{path}.id", "duplicate node id"))
                node_ok = False
            else:
                seen_ids.add(raw_id)

            host = raw_node.get("host")
            host_ok = host == LOOPBACK_HOST
            if not host_ok:
                issues.append(
                    _issue(
                        node_id,
                        f"{path}.host",
                        f"must be the exact loopback literal {LOOPBACK_HOST!r}",
                    )
                )

            port = raw_node.get("port")
            port_ok = _is_int(port) and 1 <= port <= 65535
            if not port_ok:
                issues.append(
                    _issue(
                        node_id,
                        f"{path}.port",
                        "must be an integer between 1 and 65535",
                    )
                )
            elif port in seen_ports:
                issues.append(
                    _issue(
                        node_id,
                        f"{path}.port",
                        f"duplicate port; already used by nodes[{seen_ports[port]}]",
                    )
                )
                port_ok = False
            else:
                seen_ports[port] = index

            quota = raw_node.get("quota", 1)
            quota_ok = _is_int(quota) and quota >= 1
            if not quota_ok:
                issues.append(
                    _issue(node_id, f"{path}.quota", "must be an integer >= 1")
                )

            if node_ok and host_ok and port_ok and quota_ok:
                nodes.append(NodeSpec(id=raw_id, host=host, port=port, quota=quota))

    if issues:
        raise SpecValidationError(issues)

    return Spec(
        name=name,
        parameter_version=parameter_version,
        global_concurrency=global_concurrency,
        observation_timeout_ms=timeout_ms,
        ui_latency_budget_ms=latency_budget_ms,
        observation_window=observation_window,
        arbitration_rule=arbitration_rule,
        keys=keys,
        nodes=tuple(sorted(nodes, key=lambda node: node.id)),
    )


def load_spec(path: str | Path) -> Spec:
    """Load a declaration from disk; invalid JSON is reported as E_SPEC."""

    spec_path = Path(path)
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecValidationError(
            [
                _issue(
                    "<file>",
                    "$",
                    f"cannot read {spec_path}: {exc.__class__.__name__}: {exc}",
                )
            ]
        ) from exc
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except _DuplicateJsonKey as exc:
        raise SpecValidationError(
            [_issue("<file>", "$", f"duplicate JSON object key: {exc}")]
        ) from exc
    except json.JSONDecodeError as exc:
        raise SpecValidationError(
            [
                _issue(
                    "<file>",
                    "$",
                    f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
                )
            ]
        ) from exc
    return parse_spec_data(data, source=str(spec_path))
